"""Distributional constraints (p0, pT) for each scenario.

Each environment class provides a build_constraint static method
that returns (p0, pT) Sampler objects.
"""

import logging
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.distributions as td

from envs import get_env_class

log = logging.getLogger(__file__)


class Sampler:
    """Wrapper around a torch Distribution for batched sampling and log-prob."""

    def __init__(self, distribution: td.Distribution, batch_size: int, device: str):
        self.distribution = distribution
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)

    def sample(self, batch: Optional[int] = None) -> torch.Tensor:
        if batch is None:
            batch = self.batch_size
        return self.distribution.sample([batch]).to(self.device)


class ProblemDists(NamedTuple):
    p0: Sampler
    pT: Sampler
    constraints: tuple = ()  # tuple of (t, sampler), sorted by t


def build_constraint(problem_name: str, batch_size: int, device: str, x_dim: int = None) -> ProblemDists:
    """Build (p0, pT) constraints by delegating to the environment class."""
    env_class = get_env_class(problem_name)

    if hasattr(env_class, 'build_constraint') and callable(env_class.build_constraint):
        try:
            return env_class.build_constraint(batch_size, device, x_dim=x_dim or 20)
        except TypeError:
            return env_class.build_constraint(batch_size, device)

    # Fallback: generic Gaussian builder
    log.warning("No custom build_constraint for %s, using defaults.", problem_name)
    return _generic_gaussian_constraint(problem_name, batch_size, device)


def _generic_gaussian_constraint(problem_name: str, batch_size: int, device: str):
    """Generic Gaussian p0 and pT as safe fallback."""
    d = 2
    mu0 = torch.zeros(d, device=device)
    cov0 = torch.eye(d, device=device)
    dist0 = td.MultivariateNormal(mu0, cov0)
    p0 = Sampler(dist0, batch_size, device)

    muT = torch.ones(d, device=device)
    covT = torch.eye(d, device=device)
    distT = td.MultivariateNormal(muT, covT)
    pT = Sampler(distT, batch_size, device)

    return ProblemDists(p0, pT)
