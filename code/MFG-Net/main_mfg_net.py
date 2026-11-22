"""MFG-Net: train and evaluate on a given scenario.

Usage:
    python main_mfg_net.py --env Gauss2Gauss_Obs3 [--device cpu|cuda] [--max-iters 100000]
"""
import sys
import os
import argparse
import json
import datetime
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from envs import env_dict
from evaluator import MFGEvaluatorBase
from models.mfg import MeanFieldGame
from models.costs import Fp, Fe, Fcomb, Gpref


def _get_time_string():
    out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
    return out[:10] + '__' + out[11:]


def _create_experiment_dir(env_name, dim, base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(PROJECT_ROOT, 'Experiments')
    run_name = f'{env_name}_d{dim}_{_get_time_string()}'
    exp_dir = os.path.join(base_dir, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    for sub in ['models', 'plots', 'mfg_eval']:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    return exp_dir


def _save_config(exp_dir, env_name, args_dict, env_info):
    config = {'experiment_args': args_dict, 'env_info': env_info}
    with open(os.path.join(exp_dir, f'{env_name}_config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)


class MFGNetEvaluator(MFGEvaluatorBase):
    """Evaluator using MFGNet ODE integration for trajectory generation."""

    def __init__(self, mfg_solver, env, lambda_L, lambda_E, lambda_P, device='cpu'):
        self.phi = mfg_solver.phi
        self.alpha0 = mfg_solver.alpha[0].item()
        self.TT = env.TT
        self.dim = env.dim

        class _MFGGenWrapper:
            def __init__(self, phi_fn, alpha0, dim, TT):
                self.phi = phi_fn
                self.alpha0 = alpha0
                self.dim = dim
                self.TT = TT
            def __call__(self, t, z):
                with torch.no_grad():
                    XT = torch.cat([z, t], dim=1)
                    grad = self.phi.get_grad_potential(XT)
                    return z + t * (-1.0 / self.alpha0) * grad[:, :self.dim]

        super().__init__(
            generator=_MFGGenWrapper(self.phi, self.alpha0, env.dim, env.TT),
            env=env, lambda_L=lambda_L, lambda_E=lambda_E, lambda_P=lambda_P, device=device,
        )

    def generate_trajectories(self, N, n_steps, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        z = self.env.sample_rho0(N).to(self.device)
        h = self.TT / n_steps
        traj, x = [], z.clone()
        with torch.no_grad():
            for k in range(n_steps):
                t = k * h
                XT = torch.cat([x, torch.full((N, 1), t, device=self.device)], dim=1)
                grad_phi = self.phi.get_grad_potential(XT, create_graph=False)
                x = x + h * (-1.0 / self.alpha0) * grad_phi[:, :self.dim]
                traj.append(x.detach().cpu().numpy())
        return np.stack(traj, axis=1)

    def _get_velocity(self, x_k, k, traj_t, dt_):
        """v = -(1/alpha_0) * grad_x Phi(x_k, t_k)."""
        with torch.no_grad():
            t_k = k * dt_
            batch = x_k.shape[0]
            t_tensor = torch.full((batch, 1), t_k, device=self.device, dtype=x_k.dtype)
            XT = torch.cat([x_k, t_tensor], dim=1)
            grad = self.phi.get_grad_potential(XT, create_graph=False)
            return -(1.0 / self.alpha0) * grad[:, :self.dim]


@torch.no_grad()
def _plot_training(ax, scenario, mfg, x0, device):
    """Swarm plot: rainbow scatter snapshots over time."""
    for center, radius in getattr(scenario, 'obstacles', getattr(scenario, '_obstacles', [])):
        cx, cy = center.cpu().numpy() if torch.is_tensor(center) else center
        ax.add_patch(Circle((cx, cy), radius, facecolor='darkgray',
                             edgecolor='black', alpha=0.4, zorder=1))

    n_part, n_steps = 200, 100
    traj_np = mfg.get_trajectories(x0[:n_part].to(device), n_steps=n_steps).cpu().numpy()

    n_snap = 10
    snap_idx = np.linspace(0, n_steps, n_snap).astype(int)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_snap))
    for si, c in zip(snap_idx, colors):
        ax.scatter(traj_np[si, :, 0], traj_np[si, :, 1], s=8, color=c, alpha=0.6, zorder=2)

    xlim = getattr(scenario, 'plot_xlim', (-6, 6))
    ylim = getattr(scenario, 'plot_ylim', (-6, 6))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)


def train_mfgnet(scenario, d=2, m=64, nTime=4, nt=4, n_iters=200,
                 batch_size=256, lr=1e-3, device='cpu', alpha_train=None,
                 stepper='rk4', plot_dir=None, model_dir=None,
                 plot_freq=50, print_freq=20):
    if alpha_train is None:
        # alpha: [costL, costF_obs, costF_ent, costKL, costHJB, cost_phiT]
        alpha_train = [scenario.lambda_L, 1.0, 1.0, 1.0, 0.1, 0.1]

    mfg = MeanFieldGame(d, m, nTime, alpha_train, T=scenario.TT, nt=nt, stepper=stepper).to(device)
    optimizer = torch.optim.Adam(mfg.phi.parameters(), lr=lr)
    warmup_iters = 1000
    loss_history = []
    plot_seed_x0 = scenario.sample_rho0(200).to(device)

    # Print weight configuration once
    aL, aObs, aEnt, aKL, aHJB, aPhiT = alpha_train
    print(f"Weights: lambda_L(costL)={aL}, alpha_obs(costF_obs, lambda_P={scenario.lambda_P})={aObs},"
          f" alpha_ent(costF_ent, lambda_E={scenario.lambda_E})={aEnt},"
          f" alpha_KL(costKL)={aKL}, alpha_HJB={aHJB}, alpha_phiT(Phi~KL)={aPhiT}")

    def _save_final_model(lh):
        if model_dir is None:
            return
        torch.save({
            'mfg_state_dict': mfg.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': lh,
            'architecture': {
                'd': d, 'm': m, 'nTime': nTime,
                'nt': nt, 'stepper': stepper,
                'alpha': alpha_train, 'TT': scenario.TT,
                'env_name': scenario.name,
            },
        }, os.path.join(model_dir, 'mfg_final.pt'))

    print(f"Training MFGNet on {device}: d={d}, m={m}, nTime={nTime}, nt={nt}, n_iters={n_iters}")
    train_start = time.time()
    for itr in range(n_iters + 1):
        x0 = scenario.sample_rho0(batch_size).to(device)
        w = torch.full((batch_size,), 1.0 / batch_size, device=device)
        rho0x = scenario.rho0_pdf(x0)

        def obstacle_q(x_with_t):
            return scenario.obstacle_potential(x_with_t[:, :scenario.dim])
        F_obs = Fp(Q_func=obstacle_q, lam=scenario.lambda_P)
        F_ent = Fe(rho0=None, rho0x=rho0x, lam=scenario.lambda_E)
        # Combined delta for HJB residual (sum of variational derivatives)
        F_combined = Fcomb([F_obs, F_ent])
        G_func = Gpref(Pref=scenario.terminal_cost, mu=1.0)

        optimizer.zero_grad()
        total_cost = mfg(x0, w, rho0x, F_obs, F_ent, F_combined.get_delta, G_func, G_func.get_delta)
        total_cost.backward()
        torch.nn.utils.clip_grad_norm_(mfg.phi.parameters(), max_norm=10.0)
        optimizer.step()

        if itr < warmup_iters:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (itr + 1) / warmup_iters

        costs = mfg.get_cost_breakdown()
        loss_history.append((costs['total'], costs['costL'],
                              costs['costF_obs'] + costs['costF_ent'],
                              costs['costKL']))

        if itr % print_freq == 0:
            print(f"  itr {itr:4d}: total={costs['total']:.4e},"
                  f" costL={costs['costL']:.4e},"
                  f" costF_obs={costs['costF_obs']:.4e},"
                  f" costF_ent={costs['costF_ent']:.4e},"
                  f" costKL={costs['costKL']:.4e},"
                  f" costHJB={costs['costHJB']:.4e},"
                  f" cost_phiT={costs['cost_phiT']:.4e},"
                  f" elapsed={(time.time() - train_start) / 60.0:.1f} min")

        if plot_dir is not None and itr % plot_freq == 0 and itr > 0:
            fig, ax = plt.subplots(figsize=(6, 5.5))
            _plot_training(ax, scenario, mfg, plot_seed_x0, device)
            ax.set_title(f'{itr} iterations', fontsize=16, y=1.02)
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, f'{scenario.name}_d{d}_plot-itr-{itr}.pdf'), dpi=150, bbox_inches='tight')
            plt.close(fig)

    total_h = (time.time() - train_start) / 3600.0
    print(f"Training complete --- total time: {total_h:.2f} hours")

    _save_final_model(loss_history)
    print("Training complete. Final model saved.")

    # Plot loss curves (total + terminal cost)
    if plot_dir is not None and len(loss_history) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        iters = list(range(len(loss_history)))
        totals = [lh[0] for lh in loss_history]
        gs = [lh[3] for lh in loss_history]
        ax.plot(iters, totals, color='#2196F3', linewidth=0.8, alpha=0.7, label='total')
        ax.plot(iters, gs, color='#FF5722', linewidth=0.8, alpha=0.7, label='costKL (terminal KL)')
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('Loss', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f'MFG-Net for {scenario.name} (d={d})', fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('symlog')
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'{scenario.name}_d{d}_loss_curves.pdf'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    return mfg, loss_history


def main():
    import warnings
    warnings.filterwarnings("ignore", message="ragged nested sequences")
    parser = argparse.ArgumentParser(description='MFG-Net: train and evaluate')
    parser.add_argument('--env', default='Gauss2Gauss_Obs3', choices=list(env_dict.keys()))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-iters', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--plot-freq', type=int, default=50)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--nTime', type=int, default=10)
    parser.add_argument('--nt', type=int, default=20)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--num-repeats', type=int, default=5)
    parser.add_argument('--stepper', default='rk4', choices=['rk4', 'rk1'])
    parser.add_argument('--lambda-p', type=float, default=None,
                        help='Override obstacle cost weight (default: env default)')

    args = parser.parse_args()
    device = torch.device(args.device)

    scenario_class = env_dict[args.env]
    d = max(2, args.dim)
    scenario = scenario_class(device=device, d=d)
    if args.lambda_p is not None:
        scenario.lambda_P = args.lambda_p
    env_name = scenario.name

    exp_dir = _create_experiment_dir(env_name, d)
    print(f"Experiment directory: {exp_dir}")

    log_path = os.path.join(exp_dir, f'{env_name}_d{d}.log')
    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, data):
            for f in self.files: f.write(data)
        def flush(self):
            for f in self.files: f.flush()
    sys.stdout = _Tee(sys.stdout, open(log_path, 'w', buffering=1))

    _save_config(exp_dir, env_name, vars(args), scenario.info_dict)

    plots_dir = os.path.join(exp_dir, 'plots')
    model_dir = os.path.join(exp_dir, 'models')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    mfg, loss_history = train_mfgnet(scenario, d=d, m=args.m, nTime=args.nTime, nt=args.nt,
                                      n_iters=args.max_iters, batch_size=args.batch_size,
                                      lr=args.lr, device=device, stepper=args.stepper,
                                      plot_dir=plots_dir, model_dir=model_dir,
                                      plot_freq=args.plot_freq, print_freq=args.print_freq)

    evaluator = MFGNetEvaluator(mfg, scenario, scenario.lambda_L, scenario.lambda_E, 10.0, device)
    _, cum_summary = evaluator.evaluate(os.path.join(exp_dir, 'mfg_eval'),
                                          num_repeats=args.num_repeats, n_particles=500)


if __name__ == '__main__':
    main()
