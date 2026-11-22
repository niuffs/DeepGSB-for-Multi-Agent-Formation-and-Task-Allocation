import torch
import math


# ============================================================
#  Probability Distributions
# ============================================================

class Gaussian:
    """Multivariate Gaussian with diagonal covariance."""

    def __init__(self, dim, sigma=None, mu=None, alpha=1.0):
        self.dim = dim
        self.sigma = torch.as_tensor(sigma or torch.ones(dim), dtype=torch.float)
        self.mu = torch.as_tensor(mu or torch.zeros(dim), dtype=torch.float)
        self.alpha = alpha

    def __call__(self, X):
        t1 = self.alpha / (math.sqrt((2 * math.pi) ** self.dim * self.sigma.prod()))
        t2 = torch.exp(-0.5 * torch.sum(((X - self.mu) / self.sigma) ** 2, dim=1))
        return t1 * t2

    def sample(self, n):
        return self.sigma * torch.randn(n, self.dim) + self.mu

    def log_pdf(self, X):
        const = -0.5 * self.dim * math.log(2 * math.pi)
        const -= 0.5 * torch.log(self.sigma).sum()
        const += math.log(self.alpha)
        return const - 0.5 * torch.sum(((X - self.mu) / self.sigma) ** 2, dim=1)

    def score(self, X):
        return -(X - self.mu) / self.sigma ** 2


class GaussianMixture:
    """Mixture of Gaussians."""

    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.dim = gaussians[0].dim

    def __call__(self, X):
        p = self.gaussians[0](X)
        for k in range(1, len(self.gaussians)):
            p = p + self.gaussians[k](X)
        return p

    def sample(self, n):
        total_mass = sum(g.alpha for g in self.gaussians)
        X_list = []
        for k in range(len(self.gaussians) - 1):
            n_k = max(1, int(round(n * self.gaussians[k].alpha / total_mass)))
            X_list.append(self.gaussians[k].sample(n_k))
        remaining = n - sum(x.shape[0] for x in X_list)
        X_list.append(self.gaussians[-1].sample(max(1, remaining)))
        return torch.cat(X_list, dim=0)[:n]


# ============================================================
#  Running Cost F
# ============================================================

class Fp:
    """Potential-based running cost: F(U,t) = lambda * Q(x)."""

    def __init__(self, Q_func, lam=1.0):
        self.Q = Q_func
        self.lam = lam

    def __call__(self, U, t):
        d = U.shape[1] - 5
        t_tensor = torch.full((U.shape[0], 1), float(t), device=U.device, dtype=U.dtype)
        return self.lam * self.Q(torch.cat([U[:, :d], t_tensor], dim=1)).unsqueeze(1)

    def get_delta(self, U, t):
        return self.__call__(U, t)


class Fe:
    """Entropy running cost: F(U,t) = lambda * (log(rho_0) - dv)."""

    def __init__(self, rho0, rho0x, lam=1.0):
        self.rho0 = rho0
        self.rho0x = rho0x
        self.lam = lam

    def __call__(self, U, t):
        # U layout: [x(d), ldet(1), dv(1), df_obs(1), df_ent(1), hj(1)] -> total d+5 columns
        # ldet is at index d = U.shape[1] - 5
        ldet = U[:, U.shape[1] - 5]
        return self.lam * (torch.log(self.rho0x.clamp(min=1e-20)) - ldet).unsqueeze(1)

    def get_delta(self, U, t):
        # variational derivative of F(rho) = integral rho log rho: deltaF/delta rho = 1 + log rho = 1 + log rho_0 - ldet
        ldet = U[:, U.shape[1] - 5]
        log_rho0x = torch.log(self.rho0x.clamp(min=1e-20))
        return self.lam * (log_rho0x - ldet + 1).unsqueeze(1)


class F0:
    """Zero running cost."""

    def __call__(self, U, t):
        return torch.zeros(U.shape[0], 1, device=U.device)

    def get_delta(self, U, t):
        return torch.zeros(U.shape[0], 1, device=U.device)


class Fcomb:
    """Sum of multiple running costs."""

    def __init__(self, Fs):
        self.Fs = Fs

    def __call__(self, U, t):
        return sum(f(U, t) for f in self.Fs)

    def get_delta(self, U, t):
        return sum(f.get_delta(U, t) for f in self.Fs)


# ============================================================
#  Terminal Cost G
# ============================================================

class Gls:
    """Least-squares terminal cost."""

    def __init__(self, rho0, rho1, rho0x, rho1x, mu=1.0):
        self.rho0 = rho0
        self.rho1 = rho1
        self.rho0x = rho0x
        self.rho1x = rho1x
        self.mu = mu

    def __call__(self, U):
        d = U.shape[1] - 5
        detDy = torch.exp(U[:, d])
        val = self.mu * 0.5 * (self.rho0x / detDy - self.rho1(U[:, :d])) ** 2 * (detDy / self.rho0x)
        return val

    def get_delta(self, U):
        d = U.shape[1] - 5
        detDy = torch.exp(U[:, d])
        return self.mu * (self.rho0x / detDy - self.rho1(U[:, :d]))


class Gkl:
    """KL divergence terminal cost."""

    def __init__(self, rho0, rho1, rho0x, rho1x, mu=1.0):
        self.rho0 = rho0
        self.rho1 = rho1
        self.rho0x = rho0x
        self.rho1x = rho1x
        self.mu = mu

    def __call__(self, U):
        d = U.shape[1] - 5
        ldet = U[:, d]
        log_rho0x = torch.log(self.rho0x.clamp(min=1e-20))
        log_rho1 = torch.log(self.rho1(U[:, :d]).clamp(min=1e-20))
        return self.mu * (log_rho0x - ldet - log_rho1)

    def get_delta(self, U):
        d = U.shape[1] - 5
        ldet = U[:, d]
        log_rho0x = torch.log(self.rho0x.clamp(min=1e-20))
        log_rho1 = torch.log(self.rho1(U[:, :d]).clamp(min=1e-20))
        return self.mu * (1.0 + log_rho0x - ldet - log_rho1)


class Gpref:
    """Preference terminal cost: G(U) = mu * Pref(x)."""

    def __init__(self, Pref, mu=1.0):
        self.Pref = Pref
        self.mu = mu

    def __call__(self, U):
        return self.mu * self.Pref(U[:, :U.shape[1] - 5])

    def get_delta(self, U):
        return self.mu * self.Pref(U[:, :U.shape[1] - 5])


class Gcomb:
    """Sum of multiple terminal costs."""

    def __init__(self, Gs):
        self.Gs = Gs

    def __call__(self, U):
        return sum(g(U) for g in self.Gs)

    def get_delta(self, U):
        return sum(g.get_delta(U) for g in self.Gs)
