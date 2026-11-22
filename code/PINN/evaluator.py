"""
PINN cost evaluator.

Cost definitions (per agent per time step, NO lambda weights):
  - Running cost:        0.5 * ||v||^2
  - Obstacle cost:       Q(x) = sum max(0, 1.2*r - ||x-c||)
  - Congestion cost:     log(rho_hat)   (KDE-based swarm entropy)
  - Terminal distance:   Huber(||x - B||)  (L2 far, squared near target)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# ================================================================
#  KDE helpers
# ================================================================

def _silverman_bandwidth(samples, n_geom=2):
    """Silverman bandwidth on the first n_geom geometric dims only.
    Caps at 0.5 to prevent curse-of-dimensionality degeneracy."""
    x = samples[:, :n_geom] if samples.shape[1] > n_geom else samples
    N = x.shape[0]
    d = x.shape[1]
    sigma_hat = x.std(dim=0).mean().item()
    h = ((4 / (d + 2)) ** (1 / (d + 4))) * (N ** (-1 / (d + 4))) * sigma_hat
    return max(min(h, 0.5), 1e-3)


def _gaussian_kde_logpdf(x, samples, h, n_geom=2):
    """Log KDE density on the first n_geom geometric dims, plus analytical
    log-density of the remaining (noise) dimensions.
    """
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


# ================================================================
#  Evaluator
# ================================================================

class PINNEvaluator:
    """Evaluates all cost metrics for a trained PINN solver.

    Cost curves omit lambda weights for hyperparameter exploration.

    Args:
        solver: Trained PINNSolver instance
        scenario: Obs3_Case1_2d instance
        device: Device
    """

    def __init__(self, solver, scenario, device="cpu"):
        self.solver = solver
        self.scenario = scenario
        self.lambda_L = scenario.lambda_L
        self.lambda_E = scenario.lambda_E
        self.lambda_P = scenario.lambda_P
        self.dim = getattr(scenario, 'dim', 2)
        self.device = device

        self.end_point = scenario.end_point.to(device)
        self.obstacle_thresholds = []
        for center, radius in scenario.obstacles:
            center = center.to(device)
            self.obstacle_thresholds.append({
                "center": center, "radius": radius, "threshold": 1.2 * radius,
            })

    # ------------------------------------------------------------------
    #  Obstacle potential (unweighted)
    # ------------------------------------------------------------------

    def obstacle_potential(self, x):
        """Q(x) = sum 1_{||xy-c|| < r} on first 2 dims (indicator, same as DeepGSB/MFG-Net)."""
        total_Q = torch.zeros(x.shape[0], device=self.device)
        x_2d = x[:, :2] if x.shape[1] > 2 else x
        for od in self.obstacle_thresholds:
            dist = torch.norm(x_2d - od["center"][:2], dim=1)
            total_Q += (dist < od["radius"]).float()
        return total_Q

    # ------------------------------------------------------------------
    #  One evaluation run
    # ------------------------------------------------------------------

    def compute_trajectory_costs(self, seed=None):
        """Run one evaluation: compute all per-step costs (no lambda weights).

        Returns:
            costs: dict with keys 'run', 'cong', 'obs', 'term'
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        N = 500
        dt = 0.01
        T = self.scenario.TT
        n_steps = int(T / dt)
        d = self.scenario.dim

        # Sample N particles from N(A, 0.1*I)
        p0 = torch.distributions.MultivariateNormal(
            self.scenario.start_point.to(self.device),
            covariance_matrix=self.scenario.init_cov.to(self.device),
        )
        x = p0.sample((N,))

        run_step = torch.zeros(n_steps, device=self.device)
        cong_step = torch.zeros(n_steps, device=self.device)
        obs_step = torch.zeros(n_steps, device=self.device)
        term_step = torch.zeros(n_steps, device=self.device)

        t_vals = torch.linspace(0, T - dt, n_steps, device=self.device)

        for k in range(n_steps):
            t_k = t_vals[k].expand(N, 1)

            # Control from PINN solver
            u = self.solver.get_control(x, t_k)

            # Running cost: 0.5 * ||v||^2  (no lambda_L)
            run_step[k] = (0.5 * (u ** 2).sum(dim=1)).mean()

            # Obstacle cost: Q(x)  (no lambda_P)
            Q = self.obstacle_potential(x)
            obs_step[k] = Q.mean()

            # Congestion cost via KDE swarm entropy: log(rho_hat) on geometric dims only
            h = _silverman_bandwidth(x.detach(), n_geom=2)
            log_rho = _gaussian_kde_logpdf(x.detach(), x.detach(), h, n_geom=2)
            cong_step[k] = log_rho.mean()

            # Terminal cost at current time (Huber: L2 far, squared near target)
            term_step[k] = self.scenario.terminal_cost(x, delta=1.0).mean()

            # Update position (deterministic ODE: no noise)
            x = x + u * dt

        return {
            "run": run_step.detach().cpu().numpy(),
            "cong": cong_step.detach().cpu().numpy(),
            "obs": obs_step.detach().cpu().numpy(),
            "term": term_step.detach().cpu().numpy(),
        }

    # ------------------------------------------------------------------
    #  Full evaluation
    # ------------------------------------------------------------------

    def evaluate(self, save_path, num_repeats=10):
        """Run repeated evaluations, compute cumulative totals, save results.

        Args:
            save_path: Directory to save CSV, plots, and raw data
            num_repeats: Number of repeated experiments

        Returns:
            all_costs: dict of (num_repeats, n_steps) arrays
            summary: dict of cumulative cost stats
        """
        print("=" * 60)
        print("Running PINN cost evaluation...")
        print("=" * 60)

        T = self.scenario.TT
        dt = 0.01
        n_steps = int(T / dt)

        all_costs = {
            "run": np.zeros((num_repeats, n_steps)),
            "cong": np.zeros((num_repeats, n_steps)),
            "obs": np.zeros((num_repeats, n_steps)),
            "term": np.zeros((num_repeats, n_steps)),
        }

        for r in range(num_repeats):
            base_seed = 42 + r + 3000
            costs = self.compute_trajectory_costs(seed=base_seed)
            for key in all_costs:
                all_costs[key][r] = costs[key]
            print(f"  Repeat {r + 1}/{num_repeats} complete")

        # Cumulative totals (lambda-weighted)
        cum_run = dt * self.lambda_L * all_costs["run"].sum(axis=1)
        cum_cong = dt * self.lambda_E * all_costs["cong"].sum(axis=1)
        cum_obs = dt * self.lambda_P * all_costs["obs"].sum(axis=1)
        cum_term = all_costs["term"][:, -1]
        cum_total = cum_run + cum_cong + cum_obs + cum_term

        summary = {}
        for label, arr in [("C_run", cum_run), ("C_cong", cum_cong),
                           ("C_obs", cum_obs), ("C_term", cum_term),
                           ("C_total", cum_total)]:
            mean_val, std_val = arr.mean(), arr.std()
            summary[label] = (mean_val, std_val)
            print(f"  {label} = {mean_val:.4f} +/- {std_val:.4f}")

        np.savez(os.path.join(save_path, "cost_raw_data.npz"), **all_costs)

        step_summary = {}
        for key in all_costs:
            step_summary[key] = (all_costs[key].mean(axis=0), all_costs[key].std(axis=0))

        self._save_summary_csv(save_path, step_summary, summary, n_steps, dt)
        self._plot_cost_curves(save_path, all_costs, n_steps, dt)
        self._plot_cost_analysis(save_path, all_costs, n_steps, dt)
        self._plot_trajectories(save_path, n_steps, dt)

        print("\n" + "-" * 50)
        print("FINAL Cost Summary (PINN, weighted)")
        print("-" * 50)
        print(f"Hyperparameters:  lambda_L={self.lambda_L},  lambda_E={self.lambda_E},  lambda_P={self.lambda_P}")
        print(f"  C_run  (running)       = {summary['C_run'][0]:.6f}")
        print(f"  C_cong (congestion)    = {summary['C_cong'][0]:.6f}")
        print(f"  C_obs  (obstacle)      = {summary['C_obs'][0]:.6f}")
        print(f"  C_term (terminal)      = {summary['C_term'][0]:.6f}")
        print(f"  -------------------------------------------------")
        print(f"  C_total                = {summary['C_total'][0]:.6f}  +/- {summary['C_total'][1]:.6f}")

        return all_costs, summary

    # ------------------------------------------------------------------
    #  CSV output
    # ------------------------------------------------------------------

    def _save_summary_csv(self, save_path, step_summary, cum_summary, n_steps, dt):
        rows = []
        rows.append(f"# lambda_L,{self.lambda_L}")
        rows.append(f"# lambda_E,{self.lambda_E}")
        rows.append(f"# lambda_P,{self.lambda_P}")
        rows.append(f"# dt,{dt}")
        rows.append("")
        rows.append(f"# C_run,{cum_summary['C_run'][0]:.8e},{cum_summary['C_run'][1]:.8e}")
        rows.append(f"# C_cong,{cum_summary['C_cong'][0]:.8e},{cum_summary['C_cong'][1]:.8e}")
        rows.append(f"# C_obs,{cum_summary['C_obs'][0]:.8e},{cum_summary['C_obs'][1]:.8e}")
        rows.append(f"# C_term,{cum_summary['C_term'][0]:.8e},{cum_summary['C_term'][1]:.8e}")
        rows.append(f"# C_total,{cum_summary['C_total'][0]:.8e},{cum_summary['C_total'][1]:.8e}")
        rows.append("")

        header = ("step,time,run_mean,run_std,cong_mean,cong_std,"
                  "obs_mean,obs_std,term_mean,term_std")
        rows.append(header)
        for k in range(n_steps):
            t = k * dt
            row = (
                f"{k},{t:.4f},"
                f"{step_summary['run'][0][k]:.6e},{step_summary['run'][1][k]:.6e},"
                f"{step_summary['cong'][0][k]:.6e},{step_summary['cong'][1][k]:.6e},"
                f"{step_summary['obs'][0][k]:.6e},{step_summary['obs'][1][k]:.6e},"
                f"{step_summary['term'][0][k]:.6e},{step_summary['term'][1][k]:.6e}"
            )
            rows.append(row)

        csv_path = os.path.join(save_path, "cost_summary.csv")
        with open(csv_path, "w") as f:
            f.write("\n".join(rows))
        print(f"  Saved cost summary to: {csv_path}")

    # ------------------------------------------------------------------
    #  Plot: per-step cost detail curves (run, cong, obs, term)
    # ------------------------------------------------------------------

    def _plot_cost_curves(self, save_path, all_costs, n_steps, dt):
        """2x2 subplots: unweighted run, cong, obs, term + weighted combined overview."""
        t_axis = np.arange(n_steps) * dt

        colors = {
            "run": "#2196F3", "cong": "#FF9800",
            "obs": "#F44336", "term": "#9C27B0",
        }
        labels = {
            "run": "$0.5\\|v\\|^2$",
            "cong": "$\\log\\hat{\\rho}/d$",
            "obs": "$Q(x)$",
            "term": "$\\mathrm{Huber}(\\|x-B\\|)$",
        }

        cost_types = ["run", "cong", "obs", "term"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, key in enumerate(cost_types):
            ax = axes[idx]
            mean = all_costs[key].mean(axis=0)
            std = all_costs[key].std(axis=0)

            for r in range(all_costs[key].shape[0]):
                ax.plot(t_axis, all_costs[key][r], color=colors[key], alpha=0.15, linewidth=0.5)

            ax.plot(t_axis, mean, color=colors[key], linewidth=2, label=labels[key])
            ax.fill_between(t_axis, mean - std, mean + std, color=colors[key], alpha=0.2)
            if idx in (1, 3):
                # Second column: no ylabel
                pass
            else:
                ax.set_ylabel("Cost", fontsize=18)
            title_names = {
                "run": "Running cost",
                "cong": "Congestion cost",
                "obs": "Obstacle cost",
                "term": "Terminal cost",
            }
            ax.set_title(f"{title_names[key]} {labels[key]}", fontsize=20)
            ax.set_xlabel('$t$', fontsize=18)
            ax.tick_params(axis='both', labelsize=16)
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "unweighted_cost_curves_detail.pdf"), dpi=150)
        plt.close()

        # Weighted combined overview
        weights = {"run": self.lambda_L, "cong": self.lambda_E, "obs": self.lambda_P, "term": 1.0}
        weighted_labels = {
            "run": f"$\\lambda_L\\cdot 0.5\\|v\\|^2$",
            "cong": f"$\\lambda_E\\cdot\\log\\hat{{\\rho}}/d$",
            "obs": f"$\\lambda_P\\cdot Q(x)$",
            "term": "$\\mathrm{Huber}(\\|x-B\\|)$",
        }

        fig2, ax2 = plt.subplots(figsize=(14, 8))
        for key in cost_types:
            wmean = weights[key] * all_costs[key].mean(axis=0)
            wstd = weights[key] * all_costs[key].std(axis=0)
            ax2.plot(t_axis, wmean, color=colors[key], linewidth=2.5, label=weighted_labels[key])
            ax2.fill_between(t_axis, wmean - wstd, wmean + wstd, color=colors[key], alpha=0.15)
        ax2.set_xlabel("t", fontsize=20)
        ax2.set_ylabel("Weighted per-step costs", fontsize=20)
        ax2.tick_params(axis='both', labelsize=18)
        ax2.set_title(
            f"PINN for {self.scenario.name} (d={self.dim})\n"
            f"$\\lambda_L={self.lambda_L},\\; \\lambda_E={self.lambda_E},\\; \\lambda_P={self.lambda_P}$",
            fontsize=22,
        )
        ax2.legend(fontsize=18)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "weighted_cost_curves.pdf"), dpi=150)
        plt.close()

        print(f"  Saved unweighted_detail + weighted_overview to: {save_path}")

    # ------------------------------------------------------------------
    #  Plot: cost analysis — incurred vs remaining
    # ------------------------------------------------------------------

    def _plot_cost_analysis(self, save_path, all_costs, n_steps, dt):
        """Plot already-incurred cost and two remaining-cost scenarios (weighted).

        Definitions at each time step k (t = k*dt), WITH lambda weights:
          - per_step_weighted = λ_L·run + λ_E·cong + λ_P·obs
          - Already incurred:  Σ_{0..k-1} per_step_weighted * dt
          - Remaining 1 (continue): Σ_{k..end} per_step_weighted * dt + term[-1]
          - Remaining 2 (stop):     (n_steps-k) * (λ_E·cong[k] + λ_P·obs[k]) * dt + term[k]

        incurred + remain₁ = constant (= total cost) at all k.
        """
        t_axis = np.arange(n_steps) * dt
        num_repeats = all_costs["run"].shape[0]

        incurred = np.zeros((num_repeats, n_steps))
        remain1 = np.zeros((num_repeats, n_steps))
        remain2 = np.zeros((num_repeats, n_steps))

        for r in range(num_repeats):
            run = all_costs["run"][r]
            cong = all_costs["cong"][r]
            obs = all_costs["obs"][r]
            term = all_costs["term"][r]
            per_step_weighted = self.lambda_L * run + self.lambda_E * cong + self.lambda_P * obs

            for k in range(n_steps):
                incurred[r, k] = np.sum(per_step_weighted[:k]) * dt
                remain1[r, k] = np.sum(per_step_weighted[k:]) * dt + term[-1]
                remain2[r, k] = (n_steps - k) * (self.lambda_E * cong[k] + self.lambda_P * obs[k]) * dt + term[k]

        incurred_m = incurred.mean(axis=0)
        remain1_m = remain1.mean(axis=0)
        remain2_m = remain2.mean(axis=0)
        incurred_s = incurred.std(axis=0)
        remain1_s = remain1.std(axis=0)
        remain2_s = remain2.std(axis=0)
        total = incurred_m + remain1_m

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(t_axis, incurred_m, "b-", linewidth=2.5, label="Incurred")
        ax.fill_between(t_axis, incurred_m - incurred_s, incurred_m + incurred_s, alpha=0.2)

        ax.plot(t_axis, remain1_m, color="orange", linewidth=2.5, label="Continue")
        ax.fill_between(t_axis, remain1_m - remain1_s, remain1_m + remain1_s, alpha=0.2)

        ax.plot(t_axis, remain2_m, "g-", linewidth=2.5, label="Stop")
        ax.fill_between(t_axis, remain2_m - remain2_s, remain2_m + remain2_s, alpha=0.2)

        ax.axhline(y=total[0], color="gray", linestyle="--", alpha=0.6,
                   label=f"Total cost = {total[0]:.4f}")

        ax.set_xlabel("t", fontsize=20)
        ax.set_ylabel("Cost", fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(
            f"PINN Cost Analysis: {self.scenario.name} (d={self.dim})\n"
            f"$\\lambda_L={self.lambda_L},\\; \\lambda_E={self.lambda_E},\\; \\lambda_P={self.lambda_P}$",
            fontsize=22,
        )
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "cost_analysis.pdf"), dpi=150)
        plt.close()
        print(f"  Saved cost analysis plot to: {save_path}")

    # ------------------------------------------------------------------
    #  Plot: swarm trajectories with congestion coloring
    # ------------------------------------------------------------------

    def _plot_trajectories(self, save_path, n_steps, dt):
        """Swarm + cross-section + 3D trajectory plots from a single particle set."""
        N = 100
        n_snap = 20
        T = self.scenario.TT
        prefix = f"{getattr(self.scenario, 'name', 'Unknown')}_d{self.dim}"
        plots_dir = os.path.join(os.path.dirname(save_path), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        p0 = torch.distributions.MultivariateNormal(
            self.scenario.start_point.to(self.device),
            covariance_matrix=self.scenario.init_cov.to(self.device),
        )
        x = p0.sample((N,))

        # Single trajectory integration (full d-dim)
        traj_full = np.zeros((n_steps + 1, N, self.dim))
        traj_full[0] = x.detach().cpu().numpy()
        t_vals = torch.linspace(0, T - dt, n_steps, device=self.device)
        for k in range(n_steps):
            t_k = t_vals[k].expand(N, 1)
            u = self.solver.get_control(x, t_k)
            x = x + u * dt
            traj_full[k + 1] = x.detach().cpu().numpy()

        traj_2d = traj_full[:, :, :2]

        # ================================================================
        # (1) Swarm: dense rainbow snapshots, no lines
        # ================================================================
        sw_idx = np.linspace(0, n_steps, n_snap).astype(int)
        sw_colors = plt.cm.rainbow(np.linspace(0, 1, n_snap))

        fig, ax = plt.subplots(figsize=(7, 6))
        for obs in self.obstacle_thresholds:
            center = obs["center"].cpu().numpy()
            circle = plt.Circle(center, obs["radius"],
                                facecolor='darkgray', edgecolor='black', alpha=0.4, zorder=1)
            ax.add_patch(circle)
        for sidx, c in zip(sw_idx, sw_colors):
            ax.scatter(traj_2d[sidx, :, 0], traj_2d[sidx, :, 1],
                       s=6, color=c, alpha=0.6, zorder=2)
        ax.set_aspect('equal')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('$x_1$', fontsize=16)
        ax.set_ylabel('$x_2$', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f'PINN for {self.scenario.name} (d={self.dim})', fontsize=18)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{prefix}_trajectories_swarm.pdf"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {prefix}_trajectories_swarm.pdf")

        # ================================================================
        # (2) Cross-section trajectory for d>2 (first 10 particles)
        # ================================================================
        if self.dim > 2:
            from matplotlib.lines import Line2D
            N10 = 10
            n_snap10 = 5
            traj10 = traj_full[:, :N10, :2]

            snap_idx = np.linspace(0, n_steps, n_snap10).astype(int)
            snap_labels = [r'$t=0$', r'$t=0.25T$', r'$t=0.50T$', r'$t=0.75T$', r'$t=T$']
            snap_colors = plt.cm.rainbow(np.linspace(0, 1, n_snap10))

            fig2, ax2 = plt.subplots(figsize=(9, 8))
            for obs in self.obstacle_thresholds:
                center = obs["center"].cpu().numpy()
                circle = plt.Circle(center, obs["radius"],
                                    facecolor='darkgray', edgecolor='black', alpha=0.4, zorder=1)
                ax2.add_patch(circle)
            for i in range(N10):
                ax2.plot(traj10[:, i, 0], traj10[:, i, 1],
                         'k-', alpha=0.15, lw=0.5, zorder=1)
            for si, c in zip(snap_idx, snap_colors):
                ax2.scatter(traj10[si, :, 0], traj10[si, :, 1],
                            color=c, s=18, alpha=0.85, zorder=2)
            leg = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=snap_colors[i], markersize=8,
                          label=snap_labels[i]) for i in range(n_snap10)]
            ax2.legend(handles=leg, fontsize=12, loc='upper right')
            ax2.set_aspect('equal')
            ax2.set_xlim(-6, 6)
            ax2.set_ylim(-6, 6)
            ax2.set_xlabel('$x_1$', fontsize=16)
            ax2.set_ylabel('$x_2$', fontsize=16)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.set_title(f'PINN for {self.scenario.name} (d={self.dim})', fontsize=18)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{prefix}_trajectory.pdf"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {prefix}_trajectory.pdf")

        # ================================================================
        # (3) 3D trajectory for d>=3 (same particles, first 10 for clarity)
        # ================================================================
        if self.dim >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            N3d = min(10, N)
            fig3 = plt.figure(figsize=(10, 8))
            ax3 = fig3.add_subplot(111, projection='3d')
            for obs in self.obstacle_thresholds:
                c = obs["center"].cpu().numpy()[:2]
                z_vals = np.linspace(-4, 4, 20)
                theta = np.linspace(0, 2 * np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z_vals)
                x_grid = c[0] + obs['radius'] * np.cos(theta_grid)
                y_grid = c[1] + obs['radius'] * np.sin(theta_grid)
                ax3.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.20, edgecolor='none')
            for i in range(N3d):
                ax3.plot(traj_full[:, i, 0], traj_full[:, i, 1], traj_full[:, i, 2],
                         'k-', alpha=0.15, lw=0.5, zorder=1)
            ax3.scatter(traj_full[0, :N3d, 0], traj_full[0, :N3d, 1], traj_full[0, :N3d, 2],
                        color='blue', s=30, alpha=0.8, zorder=3, label=r'$t=0$')
            ax3.scatter(traj_full[-1, :N3d, 0], traj_full[-1, :N3d, 1], traj_full[-1, :N3d, 2],
                        color='red', s=30, alpha=0.8, zorder=3, label=r'$t=T$')
            ax3.set_xlabel('$x_1$', fontsize=16)
            ax3.set_ylabel('$x_2$', fontsize=16)
            ax3.set_zlabel('$x_3$', fontsize=16)
            ax3.tick_params(axis='both', labelsize=14)
            ax3.set_title(f'PINN for {self.scenario.name} (d={self.dim})', fontsize=18)
            ax3.legend(fontsize=14, loc='upper right')
            ax3.set_xlim(-6, 6)
            ax3.set_ylim(-6, 6)
            ax3.view_init(elev=25, azim=-45)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{prefix}_trajectory_3d.pdf"), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {prefix}_trajectory_3d.pdf")

        print(f"  Saved trajectory plots to: {plots_dir}")
