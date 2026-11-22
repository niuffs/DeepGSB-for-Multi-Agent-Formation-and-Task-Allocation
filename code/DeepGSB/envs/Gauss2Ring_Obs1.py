"""
Scenario: Gaussian -> ring distribution around obstacle at (2, 0).
  - Start: Gaussian at left (-4, 0), cov=0.1
  - Obstacle: Circular obstacle at (2, 0) (radius 2.0)
  - Target (T=1): Ring distribution centered at origin (R=2.0, alpha=1.5)
    clearing the obstacle.
  - No intermediate constraints.

For d > 2, all distributions are zero-padded in dims 3+, with small
independent Gaussian noise.
"""
import math
import torch
import torch.distributions as td
from easydict import EasyDict as edict


class PaddedDistribution(td.Distribution):
    """Extend a k-dim distribution to full_dim by appending
    independent Normal(0, extra_std^2) dimensions."""
    arg_constraints = {}

    def __init__(self, base_dist, full_dim, extra_std=0.1,
                 device='cpu', validate_args=None):
        super().__init__(
            batch_shape=base_dist.batch_shape,
            event_shape=torch.Size([full_dim]),
            validate_args=validate_args,
        )
        self.base = base_dist
        self.full_dim = full_dim
        k = base_dist.event_shape[0] if base_dist.event_shape else 1
        self.base_dim = k
        self.extra_std = extra_std
        self.device = device
        self._extra_logZ = 0.5 * (full_dim - k) * math.log(2 * math.pi * extra_std ** 2)

    def sample(self, sample_shape=torch.Size()):
        base = self.base.sample(sample_shape)
        n_extra = self.full_dim - self.base_dim
        extra = torch.randn(*sample_shape, n_extra, device=self.device) * self.extra_std
        return torch.cat([base, extra], dim=-1)

    def log_prob(self, value):
        base_lp = self.base.log_prob(value[..., :self.base_dim])
        extra = value[..., self.base_dim:]
        extra_lp = -0.5 * (extra / self.extra_std).pow(2).sum(-1) - self._extra_logZ
        return base_lp + extra_lp


class RingDistribution(td.Distribution):
    """Smooth 2D ring distribution: p(x) ~ exp(-alpha (||x-c||^2 - R^2)^2)."""

    arg_constraints = {}
    event_shape = torch.Size([2])

    def __init__(self, R: float, alpha: float,
                 center=(0.0, 0.0), device='cpu',
                 validate_args=None):
        super().__init__(validate_args=validate_args)
        self.R = float(R)
        self.alpha = float(alpha)
        self.cx = float(center[0])
        self.cy = float(center[1])
        self.device = device
        self.logZ = math.log(math.pi) + 0.5 * (math.log(math.pi) - math.log(alpha))

    def log_prob(self, value):
        dx = value[..., 0] - self.cx
        dy = value[..., 1] - self.cy
        r2 = dx ** 2 + dy ** 2
        return -self.alpha * (r2 - self.R ** 2) ** 2 - self.logZ

    def sample(self, sample_shape=torch.Size()):
        n = sample_shape[0] if len(sample_shape) > 0 else 1
        theta = torch.rand(n, device=self.device) * (2 * math.pi)
        sigma_r = 1.0 / (2.0 * self.R * math.sqrt(2.0 * self.alpha))
        r = self.R + torch.randn(n, device=self.device) * sigma_r
        r = r.clamp(min=0)
        x = self.cx + r * torch.cos(theta)
        y = self.cy + r * torch.sin(theta)
        return torch.stack([x, y], dim=1)

    @property
    def support(self):
        return td.constraints.real


class Gauss2Ring_Obs1:
    """2D: Gaussian start -> ring distribution around obstacle at (2, 0)."""

    def __init__(self, device='cpu', d=2, init_cov=0.1, diffusion_std=1.0):
        self.dim = d
        self.TT = 1.0
        self.lambda_L = 0.05
        self.lambda_E = 0.05
        self.lambda_P = 10.0
        self.device = device
        self.name = "Gauss2Ring_Obs1"
        self.diffusion_std = diffusion_std

        # Obstacle at (2, 0), radius 1.5 (zero-padded for d > 2)
        obs_center = torch.zeros(d, device=device)
        obs_center[0] = 2.0
        obs_center[1] = 0.0
        self.obstacles = [(obs_center, 1.5)]

        # Start: Gaussian at left (zero-padded for d > 2)
        self.start_point = torch.zeros(d, device=device)
        self.start_point[0] = -4.0
        self.init_cov = init_cov

        # Target: ring distribution around (2, 0) (R=2.5, alpha=1.5)
        self._build_target_dist()

    def _build_target_dist(self):
        ring = RingDistribution(R=2.5, alpha=1.5, center=(2.0, 0.0), device=self.device)
        if self.dim == 2:
            self.target_dist = ring
        else:
            self.target_dist = PaddedDistribution(
                ring, full_dim=self.dim, extra_std=0.1, device=self.device
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
        ax.legend(fontsize=14, loc="upper right")
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
            ax.legend(fontsize=14, loc="upper right")
            ax.view_init(elev=25, azim=-45)
            plt.savefig(os.path.join(out_path, f"{self.name}_d{self.dim}_setup_3d.pdf"), dpi=150, bbox_inches='tight')
            plt.close(fig)

    @staticmethod
    def build_constraint(batch_size, device, x_dim=None):
        import torch.distributions as td
        from mfg.constraint import Sampler, ProblemDists

        d = x_dim or 2

        # p0: Gaussian at left (zero-padded for d > 2)
        mu0 = torch.zeros(d, device=device)
        mu0[0] = -4.0
        cov0 = 0.1 * torch.eye(d, device=device)
        p0 = Sampler(td.MultivariateNormal(mu0, cov0), batch_size, device)

        # pT: ring around (2,0), R=2.5, zero-padded for d > 2
        ring = RingDistribution(R=2.5, alpha=1.5, center=(2.0, 0.0), device=device)
        if d == 2:
            pT_dist = ring
        else:
            pT_dist = PaddedDistribution(ring, full_dim=d, extra_std=0.1, device=device)
        pT = Sampler(pT_dist, batch_size, device)

        return ProblemDists(p0, pT)

    @staticmethod
    def get_default_config(sb_param='actor-critic'):
        config = edict()
        config.problem_name = 'Gauss2Ring_Obs1'
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
