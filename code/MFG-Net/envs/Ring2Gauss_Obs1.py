"""
MFG-Net environment: Ring2Gauss_Obs1.

Swapped version of Gauss2Ring_Obs1: ring start (previously target) -> Gaussian target (previously start).
Obstacles and visualization unchanged.

For d>2, extra dimensions are zero-padded with independent Gaussian noise.
"""
import math
import numpy as np
import torch
import torch.distributions as td


# ------------------------------------------------------------
#  PaddedDistribution  --  extend a k-dim distribution to d-dim
# ------------------------------------------------------------

class PaddedDistribution(td.Distribution):
    """Extend a ``k``-dim distribution to ``full_dim`` by appending
    independent ``Normal(0, extra_std^2)`` dimensions."""
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


# ------------------------------------------------------------
#  RingDistribution -- smooth ring with Gaussian radial profile
# ------------------------------------------------------------

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


# ------------------------------------------------------------
#  Environment
# ------------------------------------------------------------

class Ring2Gauss_Obs1:
    """Ring start -> obstacle -> Gaussian target (swapped from Gauss2Ring_Obs1).
    For d>2, extra dims are zero-padded with independent Gaussian noise."""

    def __init__(self, device='cpu', d=2):
        self.dim = d
        self.nu = 0.1
        self.TT = 1.0
        self.lambda_L = 0.05
        self.lambda_E = 0.05
        self.lambda_P = 500.0
        self.device = device
        self.name = "Ring2Gauss_Obs1"

        # Obstacle at (2, 0), radius 1.5 (same as Gauss2Ring_Obs1)
        self.obstacles = [
            (torch.tensor((2.0, 0.0), device=device), 1.5),
        ]

        # Start: ring distribution around (2, 0)
        self.ring = RingDistribution(R=2.5, alpha=1.5, center=(2.0, 0.0), device=device)
        if self.dim == 2:
            self.rho0_dist = self.ring
        else:
            self.rho0_dist = PaddedDistribution(
                self.ring, full_dim=self.dim, extra_std=0.1, device=self.device
            )

        # End point (reference for backward compat)
        self.end_point = torch.zeros(d, device=device)
        self.end_point[0] = -4.0

        # Target: Gaussian at (-4, 0), cov=0.1*I (formerly start_point of Gauss2Ring_Obs1)
        self.target_point = torch.zeros(d, device=device)
        self.target_point[0] = -4.0
        self.target_cov = 0.1
        self.target_dist = td.MultivariateNormal(
            self.target_point, self.target_cov * torch.eye(d, device=device)
        )

    def sample_rho0(self, num_samples, var_scale=0.1):
        """Sample from ring distribution (previously target)."""
        return self.rho0_dist.sample([num_samples])

    def rho0_log_pdf(self, x):
        """Log-pdf of ring distribution (+ noise dims)."""
        return self.rho0_dist.log_prob(x)

    def rho0_pdf(self, x):
        return torch.exp(self.rho0_log_pdf(x))

    def obstacle_potential(self, x):
        """Linear barrier on first 2 dims: max(0, r - ||x-c||)."""
        out = torch.zeros(x.shape[0], device=self.device)
        x_2d = x[:, :2] if x.shape[1] > 2 else x
        for center, radius in self.obstacles:
            dist = torch.norm(x_2d - center, dim=1)
            out = out + torch.clamp(radius - dist, min=0)
        return out

    def terminal_cost(self, x):
        """KL-based terminal cost: -log p_T(x)."""
        return -self.target_dist.log_prob(x)

    @property
    def info_dict(self):
        return {
            'env_name': self.name, 'dim': self.dim, 'nu': self.nu,
            'TT': self.TT, 'lambda_L': self.lambda_L,
            'lambda_E': self.lambda_E, 'lambda_P': self.lambda_P,
        }
