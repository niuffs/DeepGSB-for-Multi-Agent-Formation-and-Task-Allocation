"""MFG-Net cost evaluator.

RAW per-step costs: run (velocity field), cong (KDE log-density), obs (indicator).
Weighted cumulative: C_total = lambda_L * integral(run)*dt + lambda_E * integral(cong)*dt/d + lambda_P * integral(obs)*dt + C_term_KL.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# -- KDE utilities --

def _silverman_bandwidth(samples, n_geom=2):
    """Silverman bandwidth on the first n_geom geometric dims only."""
    x = samples[:, :n_geom] if samples.shape[1] > n_geom else samples
    N = x.shape[0]
    d = x.shape[1]
    sigma_hat = x.std(dim=0).mean().item()
    h = ((4 / (d + 2)) ** (1 / (d + 4))) * (N ** (-1 / (d + 4))) * sigma_hat
    return max(min(h, 0.5), 1e-3)


def _gaussian_kde_logpdf(x, samples, h, n_geom=2):
    """KDE log-density on first n_geom dims + analytical log-density for noise dims."""
    d = x.shape[1]
    ng = min(n_geom, d)
    xg = x[:, :ng]
    sg = samples[:, :ng]

    # KDE on geometric dims
    diff = xg.unsqueeze(1) - sg.unsqueeze(0)
    sq_dists = torch.sum(diff ** 2, dim=-1)
    log_kernel = -sq_dists / (2 * h ** 2)
    log_const = -0.5 * ng * np.log(2 * np.pi) - ng * np.log(h)
    log_rho = torch.logsumexp(log_kernel + log_const, dim=1) - np.log(samples.shape[0])

    if d <= ng:
        return log_rho

    # Analytical log-density for extra (noise) dimensions
    extra = x[:, ng:]
    noise_std = samples[:, ng:].std().item()
    noise_std = max(noise_std, 1e-4)
    extra_lp = -0.5 * (extra / noise_std).pow(2).sum(-1) \
               - 0.5 * (d - ng) * np.log(2 * np.pi * noise_std ** 2)
    return log_rho + extra_lp


def _estimate_kl_via_kde(x, target_log_prob_fn, h, n_geom=2):
    """KL(rho_hat || p_target) approx mean(log rho_hat - log p_target), KDE on n_geom dims."""
    log_rho = _gaussian_kde_logpdf(x.detach(), x.detach(), h, n_geom=n_geom)
    log_p_target = target_log_prob_fn(x.detach())
    kl = (log_rho - log_p_target).mean()
    return kl


# -- Obstacle potential --

def _obstacle_potential(x, obstacles, device):
    """Q(x) = sum 1_{||x - c_j|| < r_j} on first 2 dims (zero-pad convention)."""
    Q = torch.zeros(x.shape[0], device=device)
    xx = x[:, :2] if x.shape[1] > 2 else x  # obstacles only constrain xy-plane
    for center, radius in obstacles:
        center = center.to(device)
        dist = torch.norm(xx - center, dim=1)
        Q = Q + (dist < radius).float()
    return Q



class MFGEvaluatorBase:
    """Compute run/cong/obs/term costs from generated trajectories."""

    def __init__(self, generator, env, lambda_L, lambda_E, lambda_P, device="cpu"):
        self.generator = generator
        self.env = env
        self.lambda_L = lambda_L
        self.lambda_E = lambda_E
        self.lambda_P = lambda_P
        self.device = device
        self.dim = getattr(env, 'dim', 2)

        # Number of geometric dims for KDE (noise dims use analytical log-density)
        env_name = getattr(env, 'name', '')
        self.n_geom = 3 if 'SphereShell' in env_name else min(self.dim, 2)

        # Terminal cost
        self.end_point = env.end_point.to(device) if hasattr(env, 'end_point') else None
        self.obstacles = getattr(env, '_obstacles', getattr(env, 'obstacles', []))

    # -- Velocity field (override in subclass for direct gradient) --

    def _get_velocity(self, x_k, k, traj_t, dt_):
        """Default: finite-difference velocity approximation."""
        if k < traj_t.shape[1] - 1:
            v_k = (traj_t[:, k + 1, :] - traj_t[:, k, :]) / dt_
        else:
            v_k = (traj_t[:, k, :] - traj_t[:, k - 1, :]) / dt_
        return v_k

    # -- Target log-probability --

    def _target_log_prob(self, x):
        """log p_target(x) via env's target_dist."""
        return -self.env.terminal_cost(x)

    # -- Cost computation --

    def compute_costs(self, seed=None, n_particles=500):
        """Run one evaluation: compute RAW (unweighted) per-step costs.

        Args:
            seed: random seed for reproducibility.
            n_particles: number of particles for trajectory computation.
        """
        N = n_particles
        dt_ = 0.01
        T = self.env.TT
        n_steps = int(T / dt_)

        traj_np = self.generate_trajectories(N, n_steps, seed=seed)
        traj_t = torch.tensor(traj_np, dtype=torch.float32, device=self.device)

        run_cost = torch.zeros(n_steps, device=self.device)
        cong_cost = torch.zeros(n_steps, device=self.device)
        obs_cost = torch.zeros(n_steps, device=self.device)
        term_cost = torch.zeros(n_steps, device=self.device)

        for k in range(n_steps):
            x_k = traj_t[:, k, :]

            # Running cost (RAW): 0.5 * ||v||^2 from velocity field
            v_k = self._get_velocity(x_k, k, traj_t, dt_)
            run_k = 0.5 * (v_k ** 2).sum(dim=1)
            run_cost[k] = run_k.mean()

            # Obstacle cost (RAW): Q(x)
            Q = _obstacle_potential(x_k, self.obstacles, self.device)
            obs_cost[k] = Q.mean()

            # Congestion cost (RAW): log(rho_hat) via KDE on n_geom dims + analytical noise
            h = _silverman_bandwidth(x_k.detach(), n_geom=self.n_geom)
            log_rho = _gaussian_kde_logpdf(x_k.detach(), x_k.detach(), h, n_geom=self.n_geom)
            cong_cost[k] = log_rho.mean()

            # Terminal cost: only meaningful at final step (KL divergence)
            if k == n_steps - 1:
                h_term = _silverman_bandwidth(x_k.detach(), n_geom=self.n_geom)
                kl = _estimate_kl_via_kde(x_k, self._target_log_prob, h_term, n_geom=self.n_geom)
                term_cost[k] = kl.item()
            else:
                term_cost[k] = 0.0

        total_step = run_cost + cong_cost + obs_cost

        return {
            "run": run_cost.detach().cpu().numpy(),
            "cong": cong_cost.detach().cpu().numpy(),
            "obs": obs_cost.detach().cpu().numpy(),
            "term": term_cost.detach().cpu().numpy(),
            "total": total_step.detach().cpu().numpy(),
        }

    def evaluate(self, save_path, num_repeats=10, n_particles=500):
        print("=" * 60)
        print("MFG-Net Evaluation Results (weighted)")
        print("=" * 60)

        T = self.env.TT
        dt_ = 0.01
        n_steps = int(T / dt_)

        all_costs = {
            "run": np.zeros((num_repeats, n_steps)),
            "cong": np.zeros((num_repeats, n_steps)),
            "obs": np.zeros((num_repeats, n_steps)),
            "term": np.zeros((num_repeats, n_steps)),
            "total": np.zeros((num_repeats, n_steps)),
        }

        for r in range(num_repeats):
            seed = 42 + r
            costs = self.compute_costs(seed=seed, n_particles=n_particles)
            for key in all_costs:
                all_costs[key][r] = costs[key]
            print(f"  Repeat {r + 1}/{num_repeats} complete")

        # ---- Weighted cumulative totals ----
        lam = self.lambda_L
        le = self.lambda_E
        lp = self.lambda_P
        dim_norm = 1.0 / max(self.dim, 1)  # log-density quantities scale with dimension
        cum_run = dt_ * lam * all_costs["run"].sum(axis=1)
        cum_cong = dt_ * le * all_costs["cong"].sum(axis=1) * dim_norm
        cum_obs = dt_ * lp * all_costs["obs"].sum(axis=1)
        cum_term = all_costs["term"][:, -1] * dim_norm
        cum_total = cum_run + cum_cong + cum_obs + cum_term

        cum_summary = {}
        for label, arr in [("C_run", cum_run), ("C_cong", cum_cong),
                           ("C_obs", cum_obs), ("C_term_KL", cum_term),
                           ("C_total", cum_total)]:
            m, s = arr.mean(), arr.std()
            cum_summary[label] = (m, s)

        np.savez(os.path.join(save_path, f"{self.env.name}_d{self.dim}_cost_raw_data.npz"), **all_costs)

        step_summary = {}
        for key in all_costs:
            step_summary[key] = (all_costs[key].mean(axis=0), all_costs[key].std(axis=0))

        self._save_csv(save_path, step_summary, cum_summary, n_steps, dt_)
        self._plot_curves(save_path, all_costs, n_steps, dt_)
        self._plot_trajectories(save_path)

        print()
        print(f"  lambda_L (running)     = {lam}")
        print(f"  lambda_E (congestion)  = {le}")
        print(f"  lambda_P (obstacle)    = {lp}")
        print(f"  Dimension (d)          = {self.dim}")
        print(f"  Congestion normalized  = lambda_E * log(rho) / d")
        print(f"  Terminal cost          = KL(rho_T || p_target)")
        print("-" * 55)
        for key in ["C_run", "C_cong", "C_obs", "C_term_KL", "C_total"]:
            m, s = cum_summary[key]
            print(f"  {key:15s} = {m:.6f}  +/-  {s:.6f}")
        print("=" * 55)

        return all_costs, cum_summary

    def _save_csv(self, save_path, step_summary, cum_summary, n_steps, dt_):
        rows = []
        rows.append("# MFG-Net Cost Evaluation (raw per-step, weighted cumulatives)")
        rows.append(f"# lambda_L,{self.lambda_L}")
        rows.append(f"# lambda_E,{self.lambda_E}")
        rows.append(f"# lambda_P,{self.lambda_P}")
        rows.append(f"# dim,{self.dim}")
        rows.append(f"# dt,{dt_}")
        rows.append("")
        rows.append("# Weighted cumulative totals (mean, std)")
        rows.append(f"# C_run,{cum_summary['C_run'][0]:.8e},{cum_summary['C_run'][1]:.8e}")
        rows.append(f"# C_cong,{cum_summary['C_cong'][0]:.8e},{cum_summary['C_cong'][1]:.8e}")
        rows.append(f"# C_obs,{cum_summary['C_obs'][0]:.8e},{cum_summary['C_obs'][1]:.8e}")
        rows.append(f"# C_term_KL,{cum_summary['C_term_KL'][0]:.8e},{cum_summary['C_term_KL'][1]:.8e}")
        rows.append(f"# C_total,{cum_summary['C_total'][0]:.8e},{cum_summary['C_total'][1]:.8e}")
        rows.append("")
        rows.append("# Per-step raw (unweighted) values")
        header = "step,t,run_mean,run_std,cong_mean,cong_std,obs_mean,obs_std,term_kl,term_std,total_mean,total_std"
        rows.append(header)
        for k in range(n_steps):
            t = k * dt_
            row = (
                f"{k},{t:.4f},"
                f"{step_summary['run'][0][k]:.6e},{step_summary['run'][1][k]:.6e},"
                f"{step_summary['cong'][0][k]:.6e},{step_summary['cong'][1][k]:.6e},"
                f"{step_summary['obs'][0][k]:.6e},{step_summary['obs'][1][k]:.6e},"
                f"{step_summary['term'][0][k]:.6e},{step_summary['term'][1][k]:.6e},"
                f"{step_summary['total'][0][k]:.6e},{step_summary['total'][1][k]:.6e}"
            )
            rows.append(row)

        csv_path = os.path.join(save_path, f"{self.env.name}_d{self.dim}_cost_summary.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows))
        print(f"  Saved cost summary to: {csv_path}")

    def _plot_curves(self, save_path, all_costs, n_steps, dt_):
        """Plot cost curves.

        cost_curves.pdf           -- RAW (unweighted) per-step values, 1x3 subplots.
        weighted_cost_curves.pdf  -- WEIGHTED (lambda-applied) per-step values, combined.
        """
        t_axis = np.arange(n_steps) * dt_
        lam = self.lambda_L
        le = self.lambda_E
        lp = self.lambda_P
        dim_norm = 1.0 / max(self.dim, 1)
        problem_name = self.env.name

        color_map = {"run": "#2196F3", "cong": "#FF9800", "obs": "#F44336"}
        cost_types = ["run", "cong", "obs"]

        # ============================================================
        # cost_curves.pdf -- RAW (unweighted) values
        # ============================================================
        titles = {
            "run": r"Running cost $0.5\|v\|^2$",
            "cong": r"Congestion cost $\log\hat{\rho}/d$",
            "obs": r"Obstacle cost $Q(x)$",
        }
        legend_labels = {
            "run": r"$0.5\|v\|^2$",
            "cong": r"$\log\hat{\rho}/d$",
            "obs": r"$Q(x)$",
        }
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes = axes.flatten()

        for idx, key in enumerate(cost_types):
            ax = axes[idx]
            scale = dim_norm if key == "cong" else 1.0
            data = scale * all_costs[key]
            mean = data.mean(axis=0)
            std = data.std(axis=0)

            for r in range(data.shape[0]):
                ax.plot(t_axis, data[r], color=color_map[key], alpha=0.15, linewidth=0.5)

            ax.plot(t_axis, mean, color=color_map[key], linewidth=2, label=legend_labels[key])
            ax.fill_between(t_axis, mean - std, mean + std, color=color_map[key], alpha=0.2)
            ax.set_xlabel("t", fontsize=18)
            if idx == 0:
                ax.set_ylabel("Cost", fontsize=18)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_title(titles[key], fontsize=20)
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{problem_name}_d{self.dim}_cost_curves.pdf"), dpi=150)
        plt.close()

        # ============================================================
        # weighted_cost_curves.pdf -- WEIGHTED values (lambda applied)
        # ============================================================
        legend_w_labels = {
            "run": r"$\lambda_L\cdot0.5\|v\|^2$",
            "cong": rf"$\lambda_E\cdot\log\hat{{\rho}}$",
            "obs": r"$\lambda_P\cdot Q(x)$",
        }
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        for key in cost_types:
            w = {"run": lam, "cong": le * dim_norm, "obs": lp}[key]
            mean = w * all_costs[key].mean(axis=0)
            std = w * all_costs[key].std(axis=0)
            ax2.plot(t_axis, mean, color=color_map[key], linewidth=2.5, label=legend_w_labels[key])
            ax2.fill_between(t_axis, mean - std, mean + std, color=color_map[key], alpha=0.15)
        ax2.set_xlabel("t", fontsize=20)
        ax2.set_ylabel("Weighted per-step costs", fontsize=20)
        ax2.tick_params(axis='both', labelsize=18)
        ax2.set_title(
            f"MFG-Net for {problem_name} (d={self.dim})\n"
            rf"$\lambda_L={lam},\; \lambda_E={le},\; \lambda_P={lp}$",
            fontsize=22,
        )
        ax2.legend(fontsize=18)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{problem_name}_d{self.dim}_weighted_cost_curves.pdf"), dpi=150)
        plt.close()

        print(f"  Saved cost plots to: {save_path}")

    def _plot_trajectories(self, save_path):
        """Trajectory plots: swarm + particles with colour-coded time snapshots.
        Saves to parent of save_path (experiment root), not inside eval subdir.
        """
        from matplotlib.lines import Line2D
        traj_dir = os.path.dirname(os.path.normpath(save_path))

        dt_ = 0.01
        T = self.env.TT
        n_steps = int(T / dt_)
        env_name = self.env.name

        def _obs_patches():
            patches = []
            for center, radius in self.obstacles:
                cx, cy = center.cpu().numpy() if torch.is_tensor(center) else center
                patches.append(plt.Circle((cx, cy), radius, facecolor='darkgray',
                                          edgecolor='black', alpha=0.4, zorder=1))
            return patches

        xlim = getattr(self.env, 'plot_xlim', (-6, 6))
        ylim = getattr(self.env, 'plot_ylim', (-6, 6))

        # ============================================================
        # (1) Swarm -- dense scatter snapshots, no lines, no legend
        # ============================================================
        N_swarm = 100
        n_swarm_snap = 20
        traj_swarm = self.generate_trajectories(N_swarm, n_steps, seed=9999)
        sw_idx = np.linspace(0, n_steps - 1, n_swarm_snap).astype(int)
        sw_colors = plt.cm.rainbow(np.linspace(0, 1, n_swarm_snap))

        fig, ax = plt.subplots(figsize=(7, 6))
        for p in _obs_patches():
            ax.add_patch(p)
        for sidx, c in zip(sw_idx, sw_colors):
            ax.scatter(traj_swarm[:, sidx, 0], traj_swarm[:, sidx, 1],
                       s=6, color=c, alpha=0.6, zorder=2)
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('$x_1$', fontsize=16)
        ax.set_ylabel('$x_2$', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f'MFG-Net for {env_name} (d={self.dim})', fontsize=18)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        swarm_path = os.path.join(traj_dir, f"{env_name}_d{self.dim}_trajectories_swarm.pdf")
        plt.savefig(swarm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {swarm_path}")

        # ============================================================
        # (2) 10 individual particles with colour-coded snapshots
        # ============================================================
        N10 = 10
        n_snap = 5
        traj10 = self.generate_trajectories(N10, n_steps, seed=8888)

        # Fixed snapshot labels
        snap_idx = np.linspace(0, n_steps - 1, n_snap).astype(int)
        snap_labels = [r'$t=0$', r'$t=0.25T$', r'$t=0.50T$', r'$t=0.75T$', r'$t=T$']
        snap_colors = plt.cm.rainbow(np.linspace(0, 1, n_snap))

        fig, ax = plt.subplots(figsize=(9, 8))
        for p in _obs_patches():
            ax.add_patch(p)

        # Black trajectory lines (MFG-Net traj shape: (N, n_steps, dim))
        for i in range(N10):
            ax.plot(traj10[i, :, 0], traj10[i, :, 1],
                    'k-', alpha=0.15, linewidth=0.5, zorder=1)

        # Colour-coded scatter at each snapshot
        for si, c in zip(snap_idx, snap_colors):
            ax.scatter(traj10[:, si, 0], traj10[:, si, 1],
                       color=c, s=18, alpha=0.85, zorder=2)

        # Legend: colour-coded time snapshots only
        leg_handles = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=snap_colors[i], markersize=8,
                              label=snap_labels[i]) for i in range(n_snap)]
        ax.legend(handles=leg_handles, fontsize=12, loc='upper right')

        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('$x_1$', fontsize=16)
        ax.set_ylabel('$x_2$', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f'MFG-Net for {env_name} (d={self.dim})', fontsize=18)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        particles_path = os.path.join(traj_dir, f"{env_name}_d{self.dim}_trajectories_particles.pdf")
        plt.savefig(particles_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {particles_path}")
