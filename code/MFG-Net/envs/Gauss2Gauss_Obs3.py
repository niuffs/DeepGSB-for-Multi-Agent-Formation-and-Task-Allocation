"""
MFG-Net environment: Gauss2Gauss_Obs3.

Terminal cost uses KL divergence (negative log-probability of target distribution).

Scenario: Gaussian start at (-3,-2) -> three circular obstacles -> Gaussian target at (4,4).
Supports arbitrary dimension d >= 2 via zero-padding convention.
"""
import math
import numpy as np
import torch
import torch.distributions as td


class Gauss2Gauss_Obs3:
    """Gaussian start -> three circular obstacles -> Gaussian target.

    Args:
        device: torch device
        d: spatial dimension (>= 2); extra dims are zero-padded
    """

    def __init__(self, device='cpu', d=2):
        self.dim = d
        self.nu = 0.1
        self.TT = 1.0
        self.lambda_L = 0.05      # Running cost weight
        self.lambda_E = 0.05      # Congestion weight
        self.lambda_P = 500.0     # Obstacle weight
        self.device = device
        self.name = "Gauss2Gauss_Obs3"

        # Obstacles (2D circles -- only constrain first 2 dims for d>2)
        self.obstacles = [
            (torch.tensor((2., 2.), device=device), 1.0),
            (torch.tensor((-3., 1.), device=device), 0.5),
            (torch.tensor((0., -2.), device=device), 1.5),
        ]

        # Start: zero-padded to d-dim
        self.start_point = torch.zeros(d, device=device)
        self.start_point[0] = -3.0
        self.start_point[1] = -2.0
        self.init_cov = 0.1 * torch.eye(d, device=device)

        # End / Target distribution (d-dim Gaussian, zero-padded)
        self.end_point = torch.zeros(d, device=device)
        self.end_point[0] = 4.0
        self.end_point[1] = 4.0
        self.target_cov = 0.05
        self.target_dist = td.MultivariateNormal(
            self.end_point, self.target_cov * torch.eye(d, device=device)
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
