"""
MFG-Net environment: Gauss2Rings_Obs1.

Terminal cost uses KL divergence (negative log-probability of two-ring target).

Scenario: Gaussian start at (-4,0) -> obstacle at origin -> two ring targets (R=0.7, alpha=8).
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
#  MixtureRings -- mixture of multiple RingDistribution objects
# ------------------------------------------------------------

class MixtureRings(td.Distribution):
    """Mixture of multiple ``RingDistribution`` objects."""

    arg_constraints = {}
    event_shape = torch.Size([2])

    def __init__(self, rings, weights=None, device='cpu', validate_args=None):
        super().__init__(validate_args=validate_args)
        self.rings = rings
        n = len(rings)
        if weights is None:
            log_w = torch.full((n,), -math.log(n), device=device)
        else:
            w = torch.tensor(weights, device=device)
            log_w = w.log()
        self._log_weights = log_w - torch.logsumexp(log_w, dim=0)
        self.device = device

    def sample(self, sample_shape=torch.Size()):
        n = sample_shape[0] if len(sample_shape) > 0 else 1
        idx = torch.multinomial(self._log_weights.exp(), n, replacement=True)
        samples = torch.zeros(n, 2, device=self.device)
        for i, ring in enumerate(self.rings):
            mask = idx == i
            if mask.any():
                samples[mask] = ring.sample([mask.sum().item()])
        return samples

    def log_prob(self, value):
        log_probs = torch.stack([ring.log_prob(value) for ring in self.rings], dim=-1)
        return torch.logsumexp(log_probs + self._log_weights, dim=-1)


# ------------------------------------------------------------
#  Environment
# ------------------------------------------------------------

class Gauss2Rings_Obs1:
    """2D: left Gaussian start -> center obstacle -> two ring targets at right.
    For d>2, extra dims are zero-padded with independent Gaussian noise."""

    def __init__(self, device='cpu', d=2):
        self.dim = d
        self.nu = 0.1
        self.TT = 1.0
        self.lambda_L = 0.05
        self.lambda_E = 0.05
        self.lambda_P = 500.0
        self.device = device
        self.name = "Gauss2Rings_Obs1"

        # Central circular obstacle at origin (radius 1.5)
        self.obstacles = [
            (torch.tensor((0.0, 0.0), device=device), 1.5),
        ]

        # Start: Gaussian at left (-4, 0), zero-padded for d>2
        self.start_point = torch.zeros(d, device=device)
        self.start_point[0] = -4.0
        self.init_cov = 0.1 * torch.eye(d, device=device)

        # End point (reference)
        self.end_point = torch.zeros(d, device=device)
        self.end_point[0] = 3.0

        # Ring target centers
        self.ring_centers = [
            torch.tensor((3.0, 2.5), device=device),
            torch.tensor((3.0, -2.5), device=device),
        ]

        self._build_target_dist()

    def _build_target_dist(self):
        rings = [
            RingDistribution(R=0.7, alpha=8.0, center=c.tolist(), device=self.device)
            for c in self.ring_centers
        ]
        mix = MixtureRings(rings, device=self.device)
        if self.dim == 2:
            self.target_dist = mix
        else:
            self.target_dist = PaddedDistribution(
                mix, full_dim=self.dim, extra_std=0.1, device=self.device
            )

    def sample_rho0(self, num_samples, var_scale=0.1):
        mu = self.start_point.unsqueeze(0)
        return math.sqrt(var_scale) * torch.randn(num_samples, self.dim, device=self.device) + mu

    def rho0_log_pdf(self, x):
        diff = x - self.start_point
        return -0.5 * torch.sum(diff ** 2 / self.init_cov.diag(), dim=1) \
               - 0.5 * self.dim * math.log(2 * math.pi) \
               - 0.5 * torch.log(self.init_cov.diag()).sum()

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
