"""
DeepGSB Evaluator
==================
Evaluate trained Schrodinger Bridge policies:
  - Forward control process cost: running (lambda_L), congestion (lambda_E), obstacle (lambda_P)
  - Terminal cost: KL divergence between empirical terminal distribution and target
  - KDE-based density estimation for congestion and terminal KL
"""

import os
import logging
import numpy as np
import torch
import torch.distributions as td
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d import Axes3D

from options import Options
from mfg import MFG
from deepgsb import DeepGSB

log = logging.getLogger(__file__)


# ================================================================
#  KDE utilities
# ================================================================

def silverman_bandwidth(samples, n_geom=2):
    """Silverman's rule-of-thumb bandwidth for multivariate KDE.

    For high-D experiments the extra dimensions are independent N(0, 0.1^2)
    noise -- KDE is only meaningful on the first ``n_geom`` geometric dims.
    The bandwidth is computed on those dims and capped at 0.5 to prevent
    degeneracy under the curse of dimensionality.
    """
    x = samples[:, :n_geom] if samples.shape[1] > n_geom else samples
    n = x.shape[0]
    d = x.shape[1]
    sigma_hat = x.std(dim=0).mean().item()
    h = ((4 / (d + 2)) ** (1 / (d + 4))) * (n ** (-1 / (d + 4))) * sigma_hat
    return min(max(h, 1e-3), 0.5)


def gaussian_kde_logpdf(x, samples, h, n_geom=2, noise_std=None):
    """Compute log KDE density on the first ``n_geom`` geometric dimensions,
    plus the analytical log-density of the remaining noise dimensions.

    Noise dims are modelled as i.i.d. Normal(0, noise_std^2). When
    ``noise_std`` is None, the standard deviation is estimated from the
    samples themselves, which is correct for both PaddedDistribution
    (extra_std=0.1) and native high-D environments.
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
    log_rho_geom = torch.logsumexp(log_kernel + log_const, dim=1) - np.log(samples.shape[0])

    if d <= ng:
        return log_rho_geom

    # Estimate noise std from samples if not provided
    extra = x[:, ng:]
    if noise_std is None:
        noise_std = samples[:, ng:].std().item()
        noise_std = max(noise_std, 1e-4)  # prevent degenerate log(0)
    noise_var = noise_std ** 2
    extra_lp = -0.5 * (extra / noise_std).pow(2).sum(-1) - 0.5 * (d - ng) * np.log(2 * np.pi * noise_var)
    return log_rho_geom + extra_lp


def estimate_kl_via_kde(samples_from_p, dist_q, h, n_geom=2):
    """KL(p || q) estimated via KDE on geometric dims + analytical noise dims."""
    log_p_hat = gaussian_kde_logpdf(samples_from_p, samples_from_p, h, n_geom=n_geom)
    log_q = dist_q.log_prob(samples_from_p)
    kl = (log_p_hat - log_q).mean()
    return kl


# ================================================================
#  Model loading
# ================================================================

def load_deepgsb(model_dir: str, device="cpu") -> DeepGSB:
    """Load a trained DeepGSB model from model directory.

    Args:
        model_dir: Directory containing config.txt and model files.
        device: Device to load the model onto.
    """
    # Reconstruct options from saved config
    config_path = os.path.join(model_dir, "config.txt")

    # Read problem_name from config if available
    problem_name = "Gauss2Gauss_Obs3"
    x_dim = 2
    lambda_L = 0.05
    lambda_E = 1.0
    lambda_P = 500.0
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                if line.startswith("problem_name:"):
                    problem_name = line.split(":", 1)[1].strip()
                elif line.startswith("x_dim:"):
                    x_dim = int(line.split(":", 1)[1].strip())
                elif line.startswith("lambda_L:"):
                    lambda_L = float(line.split(":", 1)[1].strip())
                elif line.startswith("lambda_E:"):
                    lambda_E = float(line.split(":", 1)[1].strip())
                elif line.startswith("lambda_P:"):
                    lambda_P = float(line.split(":", 1)[1].strip())

    opt = Options(
        problem_name=problem_name, seed=0, gpu=0, load=None,
        dir="", group="0", name="", log_fn=None, log_tb=False,
        cpu=(device == "cpu"), t0=0.0, T=1.0, interval=100,
        policy_net="toy", diffusion_std=1.0, train_bs_x=128,
        num_stage=10, num_itr=250, samp_bs=5000,
        samp_method="gauss", rb_bs_x=128,
        lr=5e-4, lr_y=1e-3, lr_gamma=0.999, lr_step=1000,
        l2_norm=0.0, optimizer="AdamW", noise_type="gaussian",
        ema=0.99, snapshot_freq=1, ckpt_freq=0,
        sb_param="actor-critic", use_rb_loss=False,
        multistep_td=True, buffer_size=20000, weighted_loss=False,
        use_env_config=True, x_dim=x_dim, device=device,
        run_dir=model_dir, eval_path=os.path.join(model_dir, "plots"),
        model_path=os.path.join(model_dir, "models"),
        log_dir=model_dir, lambda_L=lambda_L, lambda_E=lambda_E, lambda_P=lambda_P,
    )

    mfg = MFG(opt)
    deepgsb = DeepGSB(opt, mfg)

    # Load the trained model weights
    fwd_path = os.path.join(model_dir, "models", "forward_policy.pt")
    bwd_path = os.path.join(model_dir, "models", "backward_policy.pt")

    if os.path.exists(fwd_path):
        deepgsb.z_f.load_state_dict(torch.load(fwd_path, map_location=device))
    if os.path.exists(bwd_path):
        deepgsb.z_b.load_state_dict(torch.load(bwd_path, map_location=device))

    deepgsb.z_f.freeze()
    deepgsb.z_b.freeze()
    return deepgsb


# ================================================================
#  Cost Evaluator
# ================================================================

class DeepGSBEvaluator:
    """Evaluate a trained DeepGSB model.

    Cost components (all unweighted internally):
      - Running cost:     0.5 * ||v||^2                  (process loss lambda_L)
      - Obstacle cost:    indicator(||xy-c|| < r)        (obstacle loss lambda_P)
      - Congestion cost:  log rho_hat(x) (KDE)           (congestion loss lambda_E)
      - Terminal cost:    KL(rho_T || p_target)          (terminal KL divergence)

    Weighted sum:  C_total = lambda_L*C_run + lambda_E*C_cong + lambda_P*C_obs + C_term_KL
    """

    def __init__(self, deepgsb: DeepGSB, scenario, device="cpu"):
        self.deepgsb = deepgsb
        self.policy_f = deepgsb.z_f
        self.policy_b = deepgsb.z_b
        self.g = deepgsb.z_f.g
        self.mfg = deepgsb.mfg
        self.device = device

        self.scenario = scenario
        self.lambda_L = getattr(scenario, 'lambda_L', 0.01)
        self.lambda_E = getattr(scenario, 'lambda_E', 1.0)
        self.lambda_P = getattr(scenario, 'lambda_P', 500.0)
        self.dim = getattr(scenario, 'dim', 2)
        # Number of geometric (structurally meaningful) dimensions.
        # Extra dims are independent N(0, 0.1^2) noise -- process via analytical formula.
        name = getattr(scenario, 'name', '')
        self.n_geom = 3 if 'SphereShell' in name else min(self.dim, 2)

        # Obstacle geometry
        self.obstacles = []
        obs_list = getattr(scenario, 'obstacles', [])
        for center, radius in obs_list:
            c = center.to(device) if isinstance(center, torch.Tensor) else torch.tensor(center, device=device)
            self.obstacles.append({"center": c, "radius": radius})

    def get_control(self, x, t):
        z = self.policy_f(x, t)
        return self.g * z

    def get_pf_ode_drift(self, x, t):
        """Probability Flow ODE drift: 0.5(z_f - z_b)*g^2 (forward direction)."""
        z_f = self.policy_f(x, t)
        z_b = self.policy_b(x, t)
        return 0.5 * (z_f - z_b) * self.g

    def obstacle_potential(self, x):
        Q = torch.zeros(x.shape[0], device=self.device)
        for ob in self.obstacles:
            xy = x[..., :2]
            dist = torch.norm(xy - ob["center"][:2], dim=1)
            Q += (dist < ob["radius"]).float()
        return Q

    @torch.no_grad()
    def evaluate(self, save_path, n_train_samples=200, n_viz_samples=200, num_repeats=10):
        log = logging.getLogger(__name__)
        log.info("=" * 60)
        log.info("DeepGSB Evaluation Results (weighted)")
        log.info("=" * 60)
        os.makedirs(save_path, exist_ok=True)

        T = self.scenario.TT
        dt = 0.01
        n_steps = int(T / dt)

        all_costs = {
            "run": np.zeros((num_repeats, n_steps)),
            "cong": np.zeros((num_repeats, n_steps)),
            "obs": np.zeros((num_repeats, n_steps)),
            "term_kl": np.zeros((num_repeats,)),
        }

        for r in range(num_repeats):
            seed = 42 + r
            torch.manual_seed(seed)
            np.random.seed(seed)

            # --- Dense trajectory for cost computation ---
            x = self.mfg.p0.sample(batch=n_train_samples)
            traj_full = torch.zeros(n_steps + 1, n_train_samples, self.dim, device=self.device)
            traj_full[0] = x

            run_step = torch.zeros(n_steps, device=self.device)
            cong_step = torch.zeros(n_steps, device=self.device)
            obs_step = torch.zeros(n_steps, device=self.device)
            t_vals = torch.linspace(0, T - dt, n_steps, device=self.device)

            for k in range(n_steps):
                t_k = t_vals[k].expand(x.shape[0], 1)
                u = self.get_pf_ode_drift(x, t_k)

                run_step[k] = (0.5 * (u ** 2).sum(dim=1)).mean()

                Q = self.obstacle_potential(x)
                obs_step[k] = Q.mean()

                # KDE congestion cost (on geometric dims only, analytical noise)
                h = silverman_bandwidth(x.detach(), n_geom=self.n_geom)
                log_rho = gaussian_kde_logpdf(x.detach(), x.detach(), h, n_geom=self.n_geom)
                cong_step[k] = log_rho.mean()

                x = x + u * dt
                traj_full[k + 1] = x

            all_costs["run"][r] = run_step.detach().cpu().numpy()
            all_costs["cong"][r] = cong_step.detach().cpu().numpy()
            all_costs["obs"][r] = obs_step.detach().cpu().numpy()

            # --- Terminal KL divergence (KDE vs target) ---
            x_term = x.detach()
            h_term = silverman_bandwidth(x_term, n_geom=self.n_geom)
            kl_term = estimate_kl_via_kde(x_term, self.mfg.pT.distribution, h_term, n_geom=self.n_geom)
            all_costs["term_kl"][r] = kl_term.detach().cpu().item()

            # --- Sparse trajectory for visualization ---
            viz_idx = torch.randperm(n_train_samples)[:n_viz_samples]
            traj_viz = traj_full[:, viz_idx, :].detach().cpu().numpy()

            if r == 0:
                # Save trajectory for first repeat for plotting
                self._traj_viz = traj_viz
                self._t_vals = t_vals.detach().cpu().numpy()

            log.info(f"  Repeat {r + 1}/{num_repeats} complete")

        # --- Summary statistics ---
        # Log-density quantities (congestion, terminal KL) scale linearly with dim d.
        # Normalize by 1/d so HD experiments remain comparable to 2D.
        dim_norm = 1.0 / max(self.dim, 1)
        cum_run = dt * self.lambda_L * all_costs["run"].sum(axis=1)
        cum_cong = dt * self.lambda_E * all_costs["cong"].sum(axis=1) * dim_norm
        cum_obs = dt * self.lambda_P * all_costs["obs"].sum(axis=1)
        cum_term_kl = all_costs["term_kl"] * dim_norm
        cum_total = cum_run + cum_cong + cum_obs + cum_term_kl

        summary = {}
        for label, arr in [("C_run", cum_run), ("C_cong", cum_cong),
                           ("C_obs", cum_obs), ("C_term_KL", cum_term_kl),
                           ("C_total", cum_total)]:
            mean_val, std_val = arr.mean(), arr.std()
            summary[label] = (mean_val, std_val)

        # --- Save results ---
        prefix = f"{getattr(self.scenario, 'name', 'Unknown')}_d{self.dim}"
        np.savez(os.path.join(save_path, f"{prefix}_cost_raw_data.npz"), **all_costs)

        self._print_summary(summary)
        self._save_summary_csv(save_path, all_costs, summary, n_steps, dt)
        self._plot_all(save_path, all_costs, n_steps, dt)

        return all_costs, summary

    def _print_summary(self, summary):
        log = logging.getLogger(__name__)
        log.info(f"  lambda_L (running)     = {self.lambda_L}")
        log.info(f"  lambda_E (congestion)  = {self.lambda_E}")
        log.info(f"  lambda_P (obstacle)    = {self.lambda_P}")
        log.info(f"  Dimension (d)          = {self.dim}")
        log.info(f"  Congestion normalized  = lambda_E * log(rho) / d")
        log.info(f"  Terminal cost          = KL(rho_T || p_target)")
        log.info("-" * 55)
        for key in ["C_run", "C_cong", "C_obs", "C_term_KL", "C_total"]:
            m, s = summary[key]
            log.info(f"  {key:15s} = {m:.6f}  +/-  {s:.6f}")
        log.info("=" * 55)

    def _save_summary_csv(self, save_path, all_costs, summary, n_steps, dt):
        rows = [
            f"# lambda_L,{self.lambda_L}",
            f"# lambda_E,{self.lambda_E}",
            f"# lambda_P,{self.lambda_P}",
            f"# dim,{self.dim}",
            f"# n_geom,{self.n_geom}",
            f"# congestion_normalized,lambda_E*log(rho)/d",
            f"# KDE_on_first_n_geom_dims_plus_analytical_noise,true",
            f"# dt,{dt}",
            f"# terminal_cost_type,KL(rho_T || p_target)",
            "",
        ]
        rows.append("# Weighted cumulative totals (mean, std)")
        for key in ["C_run", "C_cong", "C_obs", "C_term_KL", "C_total"]:
            m, s = summary[key]
            rows.append(f"# {key},{m:.8e},{s:.8e}")
        rows.append("")

        header = "step,time,run_mean,run_std,cong_mean,cong_std,obs_mean,obs_std"
        rows.append(header)
        for k in range(n_steps):
            t = k * dt
            row = (f"{k},{t:.4f},"
                   f"{all_costs['run'].mean(axis=0)[k]:.6e},{all_costs['run'].std(axis=0)[k]:.6e},"
                   f"{all_costs['cong'].mean(axis=0)[k]:.6e},{all_costs['cong'].std(axis=0)[k]:.6e},"
                   f"{all_costs['obs'].mean(axis=0)[k]:.6e},{all_costs['obs'].std(axis=0)[k]:.6e}")
            rows.append(row)

        prefix = f"{getattr(self.scenario, 'name', 'Unknown')}_d{self.dim}"
        with open(os.path.join(save_path, f"{prefix}_cost_summary.csv"), "w") as f:
            f.write("\n".join(rows))

    def _plot_all(self, save_path, all_costs, n_steps, dt):
        self._plot_cost_curves(save_path, all_costs, n_steps, dt)

    def _plot_cost_curves(self, save_path, all_costs, n_steps, dt):
        t_axis = np.arange(n_steps) * dt
        colors = {"run": "#2196F3", "cong": "#FF9800", "obs": "#F44336"}
        cost_types = ["run", "cong", "obs"]

        # ============================================================
        # cost_curves.pdf -- UNWEIGHTED per-step values, 1x3 subplots
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
        dim_norm = 1.0 / max(self.dim, 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, key in enumerate(cost_types):
            ax = axes[idx]
            scale = dim_norm if key == "cong" else 1.0
            data = scale * all_costs[key]
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            for r in range(data.shape[0]):
                ax.plot(t_axis, data[r], color=colors[key], alpha=0.1, linewidth=0.5)
            ax.plot(t_axis, mean, color=colors[key], linewidth=2, label=legend_labels[key])
            ax.fill_between(t_axis, mean - std, mean + std, color=colors[key], alpha=0.2)
            ax.set_xlabel("$t$", fontsize=18)
            if idx == 0:
                ax.set_ylabel("Cost", fontsize=18)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_title(titles[key], fontsize=20)
            ax.legend(fontsize=16)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        prefix = f"{getattr(self.scenario, 'name', 'Unknown')}_d{self.dim}"
        plt.savefig(os.path.join(save_path, f"{prefix}_cost_curves.pdf"), dpi=150)
        plt.close()

        # ============================================================
        # weighted_cost_curves.pdf -- WEIGHTED (lambda-applied) values
        # ============================================================
        dim_norm = 1.0 / max(self.dim, 1)
        weights = {"run": self.lambda_L, "cong": self.lambda_E * dim_norm, "obs": self.lambda_P}
        legend_w_labels = {
            "run": r"$\lambda_L\cdot0.5\|v\|^2$",
            "cong": rf"$\lambda_E\cdot\log\hat{{\rho}}$",
            "obs": r"$\lambda_P\cdot Q(x)$",
        }
        problem_name = getattr(self.scenario, 'name', 'Unknown')
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for key in cost_types:
            wm = weights[key] * all_costs[key].mean(axis=0)
            ws = weights[key] * all_costs[key].std(axis=0)
            ax2.plot(t_axis, wm, color=colors[key], linewidth=2.5, label=legend_w_labels[key])
            ax2.fill_between(t_axis, wm - ws, wm + ws, color=colors[key], alpha=0.15)
        ax2.set_xlabel("$t$", fontsize=20)
        ax2.set_ylabel("Weighted per-step costs", fontsize=20)
        ax2.tick_params(axis='both', labelsize=18)
        ax2.set_title(
            f"DeepGSB for {problem_name} (d={self.dim})" + "\n"
            rf"$\lambda_L={self.lambda_L},\; \lambda_E={self.lambda_E},\; \lambda_P={self.lambda_P}$",
            fontsize=22,
        )
        ax2.legend(fontsize=18)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{prefix}_weighted_cost_curves.pdf"), dpi=150)
        plt.close()



# ================================================================
#  CLI entry point
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--n-train-samples", type=int, default=200)
    parser.add_argument("--n-viz-samples", type=int, default=200)
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.model_dir, "deepgsb_eval")
    deepgsb = load_deepgsb(args.model_dir, device=args.device)

    from envs import get_env_class
    env_class = get_env_class(deepgsb.mfg.problem_name)
    scenario = env_class(device=args.device)

    evaluator = DeepGSBEvaluator(deepgsb, scenario, device=args.device)
    evaluator.evaluate(save_dir, n_train_samples=args.n_train_samples,
                       n_viz_samples=args.n_viz_samples,
                       num_repeats=args.num_repeats)
