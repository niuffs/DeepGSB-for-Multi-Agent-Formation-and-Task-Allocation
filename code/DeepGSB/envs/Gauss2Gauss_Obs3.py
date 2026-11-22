"""Baseline 2D scenario: Gaussian start -> three circular obstacles -> Gaussian target."""
import math
import torch
import torch.distributions as td
from easydict import EasyDict as edict


class Gauss2Gauss_Obs3:
    """Gaussian start -> three circular obstacles -> Gaussian target."""

    def __init__(self, device='cpu', d=2, init_cov=0.1, diffusion_std=1.0):
        self.dim = d
        self.TT = 1.0
        self.lambda_L = 0.05
        self.lambda_E = 0.05
        self.lambda_P = 10.0
        self.device = device
        self.name = "Gauss2Gauss_Obs3"
        self.diffusion_std = diffusion_std

        # Obstacles (2D -> dD with zero padding)
        obstacles_2d = [
            ((2., 2.), 1.0),
            ((-3., 1.), 0.5),
            ((0., -2.), 1.5),
        ]
        self.obstacles = []
        for (cx, cy), radius in obstacles_2d:
            center = torch.zeros(d, device=device)
            center[0] = cx
            center[1] = cy
            self.obstacles.append((center, radius))

        # Start and target
        self.start_point = torch.zeros(d, device=device)
        self.start_point[:2] = torch.tensor((-3., -2.), device=device)
        self.init_cov = init_cov * torch.eye(d, device=device)

        self.end_point = torch.zeros(d, device=device)
        self.end_point[:2] = torch.tensor((4., 4.), device=device)

        self.end_cov = 0.05
        self.target_dist = td.MultivariateNormal(
            self.end_point, self.end_cov * torch.eye(d, device=device)
        )

    def sample_rho0(self, num_samples, var_scale=0.1):
        mu = self.start_point.unsqueeze(0)
        return math.sqrt(var_scale) * torch.randn(num_samples, self.dim, device=self.device) + mu

    def terminal_cost(self, x):
        """KL-based terminal cost: -log p_T(x)."""
        return -self.target_dist.log_prob(x)

    @property
    def info_dict(self):
        return {
            'env_name': self.name, 'dim': self.dim,
            'TT': self.TT, 'lambda_L': self.lambda_L,
            'lambda_E': self.lambda_E, 'lambda_P': self.lambda_P,
        }

    # 3D visualization (env-specific, for d > 2)

    @torch.no_grad()
    def render_setup(self, opt, save_path=None):
        """Scenario setup: 2D (always) + 3D (when d > 2)."""
        import os, numpy as np, matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from mfg.constraint import build_constraint

        constraints = build_constraint(self.name, 2000, opt.device, x_dim=opt.x_dim)
        p0_samp = constraints.p0.sample(2000).detach().cpu().numpy()
        pT_samp = constraints.pT.sample(2000).detach().cpu().numpy()

        out_path = save_path or getattr(opt, 'eval_path', '.')
        os.makedirs(out_path, exist_ok=True)

        # 2D view (xy-plane)
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_aspect('equal')
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        ax.set_xticks(range(-6, 7, 2)); ax.set_yticks(range(-6, 7, 2))
        ax.set_xlabel('$x_1$', fontsize=18); ax.set_ylabel('$x_2$', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f"DeepGSB for {self.name}", fontsize=18)

        for center, radius in self.obstacles:
            circle = Circle((center[0].item(), center[1].item()), radius,
                            facecolor='darkgray', edgecolor='black', alpha=0.4)
            ax.add_patch(circle)

        ax.scatter(p0_samp[:, 0], p0_samp[:, 1], s=3, color='blue', alpha=0.4, label=r'$\rho_0$')
        ax.scatter(pT_samp[:, 0], pT_samp[:, 1], s=3, color='red', alpha=0.4, label=r'$\rho_T$')
        ax.legend(fontsize=14, loc='upper right')
        plt.savefig(os.path.join(out_path, f"{self.name}_d{self.dim}_setup.pdf"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 3D view (when d > 2)
        if self.dim > 2:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for center, radius in self.obstacles:
                cx, cy = center[0].item(), center[1].item()
                z_vals = np.linspace(-4, 4, 20)
                theta = np.linspace(0, 2 * np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z_vals)
                x_grid = cx + radius * np.cos(theta_grid)
                y_grid = cy + radius * np.sin(theta_grid)
                ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.20, edgecolor='none')

            ax.scatter(p0_samp[:, 0], p0_samp[:, 1], p0_samp[:, 2] if self.dim > 2 else np.zeros(p0_samp.shape[0]),
                       s=3, color='blue', alpha=0.4, label=r'$\rho_0$')
            ax.scatter(pT_samp[:, 0], pT_samp[:, 1], pT_samp[:, 2] if self.dim > 2 else np.zeros(pT_samp.shape[0]),
                       s=3, color='red', alpha=0.4, label=r'$\rho_T$')

            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(-4, 4)
            ax.set_xlabel('x', fontsize=16); ax.set_ylabel('y', fontsize=16); ax.set_zlabel('z', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_title(f"{self.name} (d={self.dim})", fontsize=18)
            ax.legend(fontsize=14, loc='upper right')
            ax.view_init(elev=25, azim=-45)
            plt.savefig(os.path.join(out_path, f"{self.name}_d{self.dim}_setup_3d.pdf"), dpi=150, bbox_inches='tight')
            plt.close(fig)

    @staticmethod
    def build_constraint(batch_size, device, x_dim=None):
        from mfg.constraint import Sampler, ProblemDists
        d = x_dim or 2
        mu0 = torch.zeros(d, device=device)
        mu0[0] = -3.0
        mu0[1] = -2.0
        cov0 = 0.1 * torch.eye(d, device=device)
        p0 = Sampler(td.MultivariateNormal(mu0, cov0), batch_size, device)
        muT = torch.zeros(d, device=device)
        muT[0] = 4.0
        muT[1] = 4.0
        covT = 0.05 * torch.eye(d, device=device)
        pT = Sampler(td.MultivariateNormal(muT, covT), batch_size, device)
        return ProblemDists(p0, pT)

    @staticmethod
    def get_terminal_positions(n_particles: int, device: str):
        """10 points uniformly sampled from [3.5, 4.5] x [3.5, 4.5]."""
        pts = torch.rand(n_particles, 2, device=device) * 1.0 + 3.5
        return pts

    @staticmethod
    def get_default_config(sb_param='actor-critic'):
        config = edict()
        config.problem_name = 'Gauss2Gauss_Obs3'
        config.x_dim = 2
        config.seed = 42
        config.t0 = 0.0
        config.T = 1.0
        config.interval = 100
        config.diffusion_std = 1.0
        config.num_itr = 250
        config.train_bs_x = 128
        config.rb_bs_x = 128
        config.samp_bs = 5000
        config.snapshot_freq = 1
        config.optimizer = 'AdamW'
        config.l2_norm = 0.0
        config.lr_step = 1000
        config.noise_type = 'gaussian'
        config.ema = 0.99
        config.ckpt_freq = 0
        config.buffer_size = 20000
        config.use_rb_loss = False
        config.weighted_loss = False
        config.lambda_L = 0.05
        config.lambda_E = 0.05
        config.lambda_P = 10.0

        if sb_param == 'actor-critic':
            config.sb_param = 'actor-critic'
            config.policy_net = 'toy'
            config.lr = 5e-4
            config.lr_y = 1e-3
            config.lr_gamma = 0.999
            config.num_stage = 10
            config.multistep_td = True
            config.samp_method = 'gauss'
        elif sb_param == 'critic':
            config.sb_param = 'critic'
            config.policy_net = 'toy'
            config.lr = 5e-4
            config.lr_gamma = 0.999
            config.num_stage = 10
            config.multistep_td = True
            config.samp_method = 'gauss'
        else:
            raise ValueError(f"Unknown sb_param: {sb_param}")

        return config
