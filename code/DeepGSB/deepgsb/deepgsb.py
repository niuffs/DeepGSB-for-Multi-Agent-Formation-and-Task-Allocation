import gc
import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from easydict import EasyDict as edict
from torch.optim import SGD, Adagrad, Adam, AdamW, RMSprop, lr_scheduler

from mfg import MFG, MFGPolicy
from options import Options

from . import eval_metrics, loss_lib, sb_policy, util
from .replay_buffer import Buffer

log = logging.getLogger(__file__)

OptSchedPair = Tuple[torch.optim.Optimizer, lr_scheduler._LRScheduler]


def build_optimizer_sched(opt: Options, policy) -> OptSchedPair:
    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
        "lr": opt.lr,
        'weight_decay': opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    if policy.param == "actor-critic":
        optimizer = optim_name([
            {'params': policy.Znet.parameters()},
            {'params': policy.Ynet.parameters(), 'lr': opt.lr_y},
        ], **optim_dict)
    elif policy.param == "critic":
        optimizer = optim_name(policy.parameters(), **optim_dict)
    else:
        raise ValueError(f"Expected either actor-critic or critic, got {policy.param}")

    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    else:
        sched = None

    return optimizer, sched


def get_grad_loss_norm(opt: Options) -> str:
    return "huber"


class DeepGSB:
    def __init__(self, opt: Options, mfg: MFG):
        super(DeepGSB, self).__init__()

        self.start_time = time.time()
        self.mfg = mfg

        # Build forward (z_f) and backward (z_b) policies (single-task, no segments)
        self.z_f = sb_policy.build(opt, mfg.sde, 'forward')
        self.z_b = sb_policy.build(opt, mfg.sde, 'backward')
        if mfg.uses_mf_drift():
            mfg.initialize_mf_drift(self.z_f)

        self.optimizer_f, self.sched_f = build_optimizer_sched(opt, self.z_f)
        self.optimizer_b, self.sched_b = build_optimizer_sched(opt, self.z_b)

        self.buffer_f = Buffer(opt, 'forward') if opt.use_rb_loss else None
        self.buffer_b = Buffer(opt, 'backward') if opt.use_rb_loss else None

        self.it_f = self.it_b = 0

        # Loss history for plotting
        self.loss_hist_f = []  # list of float dicts per iteration
        self.loss_hist_b = []
        if opt.log_tb:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=opt.log_dir)

        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)

    @property
    def is_critic_param(self) -> bool:
        return self.z_f.param == self.z_b.param == "critic"

    @property
    def is_actor_critic_param(self) -> bool:
        return self.z_f.param == self.z_b.param == "actor-critic"

    def get_count(self, direction: str) -> int:
        return self.it_f if direction == "forward" else self.it_b

    def update_count(self, direction: str) -> int:
        if direction == 'forward':
            self.it_f += 1
            return self.it_f
        elif direction == 'backward':
            self.it_b += 1
            return self.it_b
        else:
            raise RuntimeError()

    def get_optimizer_sched(self, z: MFGPolicy) -> OptSchedPair:
        if z == self.z_f:
            return self.optimizer_f, self.sched_f
        elif z == self.z_b:
            return self.optimizer_b, self.sched_b
        else:
            raise RuntimeError()

    @torch.no_grad()
    def sample_train_data(self, opt: Options, train_direction: str) -> edict:
        policy_opt, policy_impt = {
            'forward': [self.z_f, self.z_b],
            'backward': [self.z_b, self.z_f],
        }.get(train_direction)

        train_ts = self.mfg.ts.detach()

        update_mf_drift = (self.mfg.uses_mf_drift() and policy_impt.direction == 'forward')

        ema, ema_impt = policy_opt.get_ema(), policy_impt.get_ema()
        with ema.average_parameters(), ema_impt.average_parameters():
            policy_impt.freeze()
            policy_opt.freeze()

            xs, zs, ws, _ = self.mfg.sample_traj(policy_impt, update_mf_drift=update_mf_drift)
            train_xs = xs.detach().cpu()
            train_zs = zs.detach().cpu()
            train_ws = ws.detach().cpu()

        log.info('Generated train data from sampling!')

        assert train_xs.shape[0] == opt.samp_bs
        assert train_xs.shape[1] == len(train_ts)
        assert train_xs.shape == train_zs.shape
        gc.collect()

        return edict(xs=train_xs, zs=train_zs, ws=train_ws, ts=train_ts)

    def train_stage(self, opt: Options, stage: int, train_direction: str, datas: Optional[edict] = None) -> None:
        policy_opt, policy_impt = {
            'forward': [self.z_f, self.z_b],
            'backward': [self.z_b, self.z_f],
        }.get(train_direction)

        buffer_impt, buffer_opt = {
            'forward': [self.buffer_f, self.buffer_b],
            'backward': [self.buffer_b, self.buffer_f],
        }.get(policy_impt.direction)

        if datas is None:
            datas = self.sample_train_data(opt, train_direction)

        if opt.log_tb:
            t1 = time.time()
            self.log_validate_metrics(opt, train_direction, datas)
            log.info("Done logging validate metrics! Took {:.1f} s!".format(time.time() - t1))

        if opt.use_rb_loss:
            buffer_impt.append(datas)

        self.train_ep(opt, stage, train_direction, datas, policy_opt, policy_impt, buffer_opt, buffer_impt)

    def train_ep(self, opt, stage, direction, datas, policy, policy_impt, buffer_opt, buffer_impt):
        train_xs, train_zs, train_ws, train_ts = datas.xs, datas.zs, datas.ws, datas.ts

        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert train_ts.shape[0] == opt.interval
        assert direction == policy.direction

        cross_boundary_indices = None

        optimizer, sched = self.get_optimizer_sched(policy)
        optimizer_impt, _ = self.get_optimizer_sched(policy_impt)

        policy.activate()
        policy_impt.freeze()

        if stage > 0 and opt.use_rb_loss:
            assert len(buffer_opt) > 0 and len(buffer_impt) > 0

        mfg = self.mfg
        samp_direction = policy_impt.direction
        for it in range(opt.num_itr):
            step = self.update_count(direction)

            samp_x_idx = torch.randint(opt.samp_bs, (opt.train_bs_x,), device='cpu')
            dim01 = [opt.train_bs_x, opt.interval]

            ts = train_ts.detach()
            xs = train_xs[samp_x_idx].to(opt.device)
            zs_impt = train_zs[samp_x_idx].to(opt.device)
            dw = train_ws[samp_x_idx].to(opt.device)

            if mfg.uses_xs_all():
                samp_x_idx2 = torch.randint(opt.samp_bs, (opt.train_bs_x,), device='cpu')
                xs_all = train_xs[samp_x_idx2].to(opt.device)
            else:
                xs_all = None

            optimizer.zero_grad()
            optimizer_impt.zero_grad()

            loss_kl, zs, kl, _ = loss_lib.compute_kl_loss(
                opt, dim01, mfg, samp_direction,
                ts.detach(), xs.detach(), zs_impt.detach(),
                policy, return_all=True
            )

            loss_bsde_td = loss_lib.compute_bsde_td_loss(
                opt, mfg, samp_direction,
                ts.detach(), xs.detach(), zs.detach(), dw.detach(), kl.detach(),
                policy, policy_impt, xs_all,
                cross_boundary_indices=cross_boundary_indices,
            )

            loss_boundary = loss_lib.compute_boundary_loss(
                opt, mfg, ts.detach(), xs.detach(), policy_impt, policy,
            )

            loss_grad = torch.tensor(0.0, device=opt.device)
            if self.is_actor_critic_param:
                norm = get_grad_loss_norm(opt)
                loss_grad = loss_lib.compute_grad_loss(opt, ts.detach(), xs.detach(), mfg.sde, policy, norm)

            loss_bsde_td_rb = torch.tensor(0.0, device=opt.device)
            if opt.use_rb_loss:
                loss_bsde_td_rb = loss_lib.compute_bsde_td_loss_from_buffer(
                    opt, mfg, buffer_impt, ts.detach(), policy, policy_impt, xs_all,
                    cross_boundary_indices=cross_boundary_indices,
                )

            if self.is_critic_param:
                if opt.weighted_loss:
                    w_kl, w_nkl = opt.weights['kl'], opt.weights['non-kl']
                    loss = w_kl * loss_kl + w_nkl * (loss_bsde_td + loss_boundary + loss_bsde_td_rb)
                else:
                    loss = loss_kl + loss_bsde_td + loss_boundary + loss_bsde_td_rb
            elif self.is_actor_critic_param:
                if opt.weighted_loss:
                    w_kl, w_nkl = opt.weights['kl'], opt.weights['non-kl']
                    loss = w_kl * loss_kl + w_nkl * (loss_boundary + loss_bsde_td + loss_grad + loss_bsde_td_rb)
                else:
                    loss = loss_kl + loss_boundary + loss_bsde_td + loss_grad + loss_bsde_td_rb
            else:
                raise RuntimeError("")

            assert not torch.isnan(loss)
            loss.backward()
            optimizer.step()
            policy.update_ema()

            if sched is not None:
                sched.step()

            loss = edict(boundary=loss_boundary, kl=loss_kl, grad=loss_grad, bsde_td=loss_bsde_td)

            # Store loss history (as plain floats) for later plotting
            hist = self.loss_hist_f if direction == "forward" else self.loss_hist_b
            hist.append({k: v.detach().cpu().item() for k, v in loss.items()})

            if it % 20 == 0:
                self.log_train(opt, it, stage, loss, optimizer, direction)


    def train(self, opt: Options) -> None:
        # Plot scenario setup
        try:
            from mfg.plotting import plot_scenario_setup
            plot_scenario_setup(opt)
            log.info("Saved scenario setup plot.")
        except Exception as e:
            log.warning("Could not plot scenario setup: %s", e)

        # Single-task training (no segments)
        self.evaluate(opt, 0)
        train_start = time.time()
        for stage in range(opt.num_stage):
            stage_start = time.time()
            if opt.samp_method == 'jacobi':
                datas1 = self.sample_train_data(opt, 'forward')
                datas2 = self.sample_train_data(opt, 'backward')
                self.train_stage(opt, stage, 'forward', datas=datas1)
                self.train_stage(opt, stage, 'backward', datas=datas2)
            elif opt.samp_method == 'gauss':
                self.train_stage(opt, stage, 'forward')
                self.train_stage(opt, stage, 'backward')

            self.evaluate(opt, stage + 1)
            stage_time = (time.time() - stage_start) / 60.0
            elapsed = (time.time() - train_start) / 60.0
            log.info("Stage %d/%d done --- %.1f min (total %.1f min)",
                     stage + 1, opt.num_stage, stage_time, elapsed)

        total_h = (time.time() - train_start) / 3600.0
        log.info("Training complete --- total time: %.2f hours", total_h)

        if opt.log_tb:
            self.writer.close()

        self._save_final_models(opt)
        try:
            self._run_evaluator(opt)
        except Exception as e:
            log.warning("Evaluator failed: %s", e)
        try:
            self._plot_sparse_trajectories(opt)
        except Exception as e:
            log.warning("Trajectory plotting failed: %s", e)
        try:
            self._plot_loss_curves(opt)
        except Exception as e:
            log.warning("Loss curve plotting failed: %s", e)


    def _save_final_models(self, opt: Options) -> None:
        """Save the two trained neural network models (forward & backward policies)."""
        import torch
        torch.save(self.z_f.state_dict(), os.path.join(opt.model_path, "forward_policy.pt"))
        torch.save(self.z_b.state_dict(), os.path.join(opt.model_path, "backward_policy.pt"))

    def _run_evaluator(self, opt: Options) -> None:
        """Run DeepGSBEvaluator on the trained model and log cost summary."""
        from evaluator import DeepGSBEvaluator
        from envs import get_env_class

        device = opt.device
        self.z_f.freeze()
        self.z_b.freeze()

        env_class = get_env_class(opt.problem_name)
        scenario = env_class(device=device, d=opt.x_dim)

        evaluator = DeepGSBEvaluator.__new__(DeepGSBEvaluator)
        evaluator.deepgsb = self
        evaluator.policy_f = self.z_f
        evaluator.policy_b = self.z_b
        evaluator.g = self.z_f.g
        evaluator.mfg = self.mfg
        evaluator.device = device
        evaluator.scenario = scenario
        evaluator.lambda_L = getattr(scenario, 'lambda_L', 0.01)
        evaluator.lambda_E = getattr(scenario, 'lambda_E', 1.0)
        evaluator.lambda_P = getattr(scenario, 'lambda_P', 500.0)
        evaluator.dim = getattr(scenario, 'dim', 2)
        evaluator.n_geom = 3 if 'SphereShell' in opt.problem_name else min(evaluator.dim, 2)

        # Build obstacle list
        evaluator.obstacles = []
        for center, radius in getattr(scenario, 'obstacles', []):
            c = center.to(device) if isinstance(center, torch.Tensor) else torch.tensor(center, device=device)
            evaluator.obstacles.append({"center": c, "radius": radius})

        save_path = os.path.join(opt.run_dir, "deepgsb_eval")
        all_costs, summary = evaluator.evaluate(
            save_path, n_train_samples=500, n_viz_samples=500, num_repeats=5
        )

    @torch.no_grad()
    def _plot_sparse_trajectories(self, opt: Options) -> None:
        """Plot forward PF ODE trajectory (single-task, no segments)."""
        from plot_trajectory import (
            plot_pf_ode_trajectory, _drift_pf, _default_formation,
            _draw_star_connection, _draw_circles_patches,
            _draw_circle_connection, _draw_rings_patches, _draw_rings_connections,
        )
        import torch, math, os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D

        # Dispatch to env-specific custom trajectory if available
        from envs import get_env_class
        env_cls = get_env_class(opt.problem_name)
        if hasattr(env_cls, 'plot_trajectory_custom'):
            env = env_cls(device=opt.device, d=opt.x_dim)
            env.plot_trajectory_custom(self, opt)
            return

        save_path = os.path.join(opt.eval_path, f"{opt.problem_name}_d{opt.x_dim}_trajectory.pdf")
        plot_pf_ode_trajectory(self, opt, save_path, title=f"{opt.problem_name} (d={opt.x_dim})")

    @torch.no_grad()
    def _plot_loss_curves(self, opt: Options) -> None:
        """Plot loss curves: init-KL, constraint KL(s), end-KL."""
        if not self.loss_hist_f and not self.loss_hist_b:
            return

        def extract(hist, key):
            return [step.get(key, 0.0) for step in hist]

        # Detect constraint-kl keys present in history (p1-kl, p2-kl, ...)
        all_keys = set()
        for hist in [self.loss_hist_f, self.loss_hist_b]:
            for step in hist:
                all_keys.update(step.keys())
        constraint_keys = sorted(
            k for k in all_keys
            if k.endswith('-kl') and k != 'end-kl'
            and extract(self.loss_hist_f or self.loss_hist_b, k)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        for direction, hist, color, ls in [
            ('Forward', self.loss_hist_f, '#2196F3', '-'),
            ('Backward', self.loss_hist_b, '#FF5722', '--'),
        ]:
            if not hist:
                continue
            iters = list(range(1, len(hist) + 1))

            # end-KL
            end_vals = extract(hist, 'kl')
            if any(abs(v) > 1e-8 for v in end_vals):
                ax.plot(iters, end_vals, color=color, linestyle=ls,
                        linewidth=1.0, alpha=0.8,
                        label=f'{direction} KL')

            # Individual constraint KLs (p1-kl, p2-kl, ...)
            cons_colors = ['#66BB6A', '#FFA726', '#AB47BC', '#26C6DA', '#EF5350']
            for ci, ck in enumerate(constraint_keys):
                c_vals = extract(hist, ck)
                if any(abs(v) > 1e-8 for v in c_vals):
                    cc = cons_colors[ci % len(cons_colors)]
                    ax.plot(iters, c_vals, color=cc, linestyle=ls,
                            linewidth=0.6, alpha=0.4, dashes=[1, 3],
                            label=f'{direction} {ck}')

        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('Loss', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f'{opt.problem_name} (d={opt.x_dim})', fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('symlog')
        if opt.problem_name == 'MixGauss2SphereShell_Obs3':
            ax.set_yscale('log')

        plt.tight_layout()
        save_path = os.path.join(opt.run_dir, 'plots', f'{opt.problem_name}_d{opt.x_dim}_loss_curves.pdf')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        log.info("Loss curves saved to %s", save_path)


    def _plot_trajectory_2d(self, ax, traj, ts, opt, title):
        """2D trajectory with rainbow-colored snapshots at 5 time points + obstacles."""
        import numpy as np
        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D

        n_steps = traj.shape[0]
        n_particles = traj.shape[1]

        n_snap = 5
        snap_indices = np.linspace(0, n_steps - 1, n_snap).astype(int)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_snap))
        snap_times = [r'$t=0$', r'$t=0.25T$', r'$t=0.50T$', r'$t=0.75T$', r'$t=T$']

        # Draw obstacles from the env
        from envs import get_env_class
        env_cls = get_env_class(opt.problem_name)
        try:
            env = env_cls(device='cpu')
            for center, radius in getattr(env, 'obstacles', []):
                cx, cy = center[0].item(), center[1].item()
                circle = Circle((cx, cy), radius,
                                facecolor='darkgray', edgecolor='black', alpha=0.5, zorder=0)
                ax.add_patch(circle)
        except Exception:
            pass

        # Draw formation circles + red triangle markers (e.g. Rings encirclement targets)
        try:
            for i, (center, radius) in enumerate(getattr(env, 'formation_circles', [])):
                cx, cy = center[0].item(), center[1].item()
                circle = Circle((cx, cy), radius, facecolor='none', edgecolor="#FDE508",
                                linestyle='--', linewidth=1.8, alpha=0.6, zorder=4)
                ax.add_patch(circle)
                label = 'encirclement target' if i == 0 else ''
                ax.scatter(cx, cy, marker='^', color='red', s=100, zorder=5, label=label)
        except Exception:
            pass

        # Trajectory lines (thin, low alpha)
        for i in range(n_particles):
            ax.plot(traj[:, i, 0].cpu(), traj[:, i, 1].cpu(), 'k-', alpha=0.2, linewidth=0.5, zorder=1)

        # Rainbow snapshots
        for sidx, color in zip(snap_indices, colors):
            ax.scatter(traj[sidx, :, 0].cpu(), traj[sidx, :, 1].cpu(),
                       color=color, s=15, alpha=0.8, zorder=2)

        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                                  markersize=8, label=snap_times[i]) for i in range(n_snap)]
        ax.legend(handles=legend_elements, fontsize=13, loc='upper right')
        ax.set_aspect('equal')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_title(f"{title}\n{opt.problem_name}", fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel('$x_1$', fontsize=18); ax.set_ylabel('$x_2$', fontsize=18)
        ax.grid(True, alpha=0.3)

    def _plot_trajectory_3d(self, ax, traj, ts, opt, title):
        """3D trajectory with rainbow snapshots + cylindrical obstacles."""
        import numpy as np

        n_steps = traj.shape[0]
        n_particles = traj.shape[1]

        n_snap = 5
        snap_indices = np.linspace(0, n_steps - 1, n_snap).astype(int)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_snap))
        snap_times = [r'$t=0$', r'$t=0.25T$', r'$t=0.50T$', r'$t=0.75T$', r'$t=T$']

        # Draw cylindrical obstacles
        from envs import get_env_class
        env_cls = get_env_class(opt.problem_name)
        try:
            env = env_cls(device='cpu')
            for center, radius in getattr(env, 'obstacles', []):
                cx, cy = center[0].item(), center[1].item()
                # Draw cylinder as a circle extruded in z: use plot_surface
                z_vals = np.linspace(-6, 6, 30)
                theta = np.linspace(0, 2 * np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z_vals)
                x_grid = cx + radius * np.cos(theta_grid)
                y_grid = cy + radius * np.sin(theta_grid)
                ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.25, edgecolor='none')
        except Exception:
            pass

        # Trajectory lines
        for i in range(n_particles):
            ax.plot(traj[:, i, 0].cpu(), traj[:, i, 1].cpu(), traj[:, i, 2].cpu(),
                    'b-', alpha=0.3, linewidth=0.6)

        # Rainbow snapshots
        for sidx, color in zip(snap_indices, colors):
            ax.scatter(traj[sidx, :, 0].cpu(), traj[sidx, :, 1].cpu(), traj[sidx, :, 2].cpu(),
                       color=color, s=20, alpha=0.8)

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                                  markersize=8, label=snap_times[i]) for i in range(n_snap)]
        ax.legend(handles=legend_elements, fontsize=13, loc='upper right')
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_zlabel('z', fontsize=16)
        ax.set_title(f"{title}\n{opt.problem_name}", fontsize=18)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        ax.view_init(elev=25, azim=-45)

    @torch.no_grad()
    def evaluate(self, opt: Options, stage: int) -> None:
        snapshot, ckpt = util.evaluate_stage(opt, stage)
        if snapshot:
            self.z_f.freeze()
            self.z_b.freeze()
            self.mfg.save_snapshot(self.z_f, self.z_b, stage)

        if ckpt and stage > 0:
            keys = ['z_f', 'optimizer_f', 'z_b', 'optimizer_b', "it_f", "it_b"]
            util.save_checkpoint(opt, self, keys, stage)

    def compute_validate_metrics(self, opt: Options, train_direction: str, datas: edict) -> Dict[str, torch.Tensor]:
        xs, zs, ts = datas.xs, datas.zs, datas.ts

        b, T, nx = xs.shape
        assert zs.shape == (b, T, nx)
        assert ts.shape == (T,)

        mfg = self.mfg
        dt = mfg.dt
        metrics = {}

        metrics.update(eval_metrics.compute_sinkhorn_metrics(opt, mfg, xs, train_direction))

        if mfg.uses_xs_all():
            xs_all = xs.to(opt.device)
            xs_all = xs_all[torch.randperm(opt.samp_bs)]
        else:
            xs_all = None

        est_mf_cost, logp = eval_metrics.compute_est_mf_cost(
            opt, self, xs, ts, dt, xs_all, return_logp=True
        )
        s_cost, mf_cost = eval_metrics.compute_state_cost(opt, mfg, xs, ts, xs_all, logp, dt)
        del logp, xs_all
        control_cost = eval_metrics.compute_control_cost(opt, zs, dt)

        mean_s_cost, mean_control_cost, mean_mf_cost = s_cost.mean(), control_cost.mean(), mf_cost.mean()
        mean_nonmf_cost = mean_s_cost + mean_control_cost
        mean_total_cost = mean_nonmf_cost + mean_mf_cost

        metrics["est_mf_cost"] = est_mf_cost
        metrics["state_cost"] = mean_s_cost
        metrics["nonmf_cost"] = mean_nonmf_cost
        metrics["mf_cost"] = mean_mf_cost
        metrics["control_cost"] = mean_control_cost
        metrics["total_cost"] = mean_total_cost

        return metrics

    @torch.no_grad()
    def log_validate_metrics(self, opt: Options, direction: str, datas: edict) -> None:
        def tag(name: str) -> str:
            return os.path.join(f"{direction}-loss", name)
        step = self.get_count(direction)
        metrics = self.compute_validate_metrics(opt, direction, datas)
        for key in metrics:
            self.writer.add_scalar(tag(key), metrics[key], global_step=step)

    def log_train(self, opt, it, stage, loss, optimizer, direction):
        time_elapsed = util.get_time(time.time() - self.start_time)
        lr = optimizer.param_groups[0]['lr']
        log.info("[SB {0}] stage {1}/{2} | itr {3}/{4} | lr {5} | loss {6} | time {7}".format(
            "fwd" if direction == "forward" else "bwd",
            str(1 + stage).zfill(2),
            opt.num_stage,
            str(1 + it).zfill(3),
            opt.num_itr,
            "{:.2e}".format(lr),
            util.get_loss_str(loss),
            "{0}:{1:02d}:{2:05.2f}".format(*time_elapsed),
        ))

        step = self.get_count(direction)
        if opt.log_tb:
            for key, val in loss.items():
                self.writer.add_scalar(
                    os.path.join(f'{direction}-loss', f'{key}'), val.detach(), global_step=step
                )
            self.writer.add_scalar(os.path.join(f"{direction}-loss", "stage"), stage, global_step=step)
            self.writer.add_scalar(os.path.join(f"{direction}-opt", "lr"), lr, global_step=step)
