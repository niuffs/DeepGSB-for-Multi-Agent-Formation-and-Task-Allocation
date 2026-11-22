"""ODE integration and Mean Field Game solver."""
import torch
import torch.nn as nn
from .modules import PotentialNN


# ============================================================
#  Trace computation (CUDA-optimized)
# ============================================================

def _compute_trace_vmap(grad_phi_x, XT, d, batch, device):
    """Compute tr(grad^2 phi) = sum_i d^2 phi / d x_i^2 using XT (requires_grad)."""
    # -- PyTorch 2.0+: vmap-batched per-dimension Hessian diagonal --
    try:
        from torch.func import vmap

        def _diag(i):
            return torch.autograd.grad(
                grad_phi_x[:, i].sum(), XT[:, i],
                create_graph=True, retain_graph=True,
            )[0]

        trace = vmap(_diag)(torch.arange(d, device=device))
        return trace.sum(dim=0)
    except (ImportError, AttributeError, RuntimeError):
        pass

    # -- Fallback: sequential loop using XT --
    trace = torch.zeros(batch, device=device)
    for i in range(d):
        hess_i = torch.autograd.grad(
            grad_phi_x[:, i].sum(), XT,
            retain_graph=True, create_graph=True,
        )[0]
        trace = trace + hess_i[:, i]
    return trace


# ============================================================
#  ODE integration
# ============================================================

def odefun(phi, U, t, alpha, F_func, getDeltaF):
    """RHS of ODE: U = [x; ldet; dv; df_obs; df_ent; hj] -> dU/dt."""
    d = U.shape[1] - 5
    x = U[:, :d]
    dev = U.device
    t_tensor = torch.full((U.shape[0], 1), t, device=dev)
    XT = torch.cat([x, t_tensor], dim=1).detach().requires_grad_(True)

    phi_val = phi(XT)
    grad_phi = torch.autograd.grad(phi_val.sum(), XT, create_graph=True)[0]

    dx = -(1.0 / alpha[0]) * grad_phi[:, :d]

    # Compute trace tr(grad^2 phi) = sum_i d^2 phi / d x_i^2
    # Use vmap (PyTorch 2.0+) if available for batched per-dimension Hessian
    trace = _compute_trace_vmap(grad_phi[:, :d], XT, d, XT.shape[0], dev)
    dl = -(1.0 / alpha[0]) * trace.unsqueeze(1)

    dv = 0.5 * (dx ** 2).sum(dim=1, keepdim=True)
    df = F_func(U, t)  # shape (batch, 2): [obs, ent]

    dphi_dt = grad_phi[:, -1]
    deltaF = getDeltaF(U, t).squeeze(-1)
    hj = torch.abs(-dphi_dt - alpha[1] * deltaF + alpha[0] * dv.squeeze(-1)).unsqueeze(1)

    return torch.cat([dx, dl, dv, df, hj], dim=1)


def rk4_step(phi, U, t, h, alpha, F_func, getDeltaF):
    """Classical RK4 step."""
    K1 = odefun(phi, U, t, alpha, F_func, getDeltaF)
    K2 = odefun(phi, U + 0.5 * h * K1, t + 0.5 * h, alpha, F_func, getDeltaF)
    K3 = odefun(phi, U + 0.5 * h * K2, t + 0.5 * h, alpha, F_func, getDeltaF)
    K4 = odefun(phi, U + h * K3, t + h, alpha, F_func, getDeltaF)
    return U + (h / 6.0) * (K1 + 2.0 * K2 + 2.0 * K3 + K4)


def rk1_step(phi, U, t, h, alpha, F_func, getDeltaF):
    """Forward Euler step."""
    return U + h * odefun(phi, U, t, alpha, F_func, getDeltaF)


def integrate(phi, U0, tspan, nt, alpha, F_func, getDeltaF, stepper='rk4'):
    """Integrate ODE from tspan[0] to tspan[1], return final state."""
    h = (tspan[1] - tspan[0]) / nt
    t = tspan[0]
    U = U0
    step = rk4_step if stepper == 'rk4' else rk1_step
    for _ in range(nt):
        U = step(phi, U, t, h, alpha, F_func, getDeltaF)
        t = t + h
    return U


def integrate_with_path(phi, U0, tspan, nt, alpha, F_func, getDeltaF, stepper='rk4'):
    """Integrate ODE and return all intermediate states (nt+1, batch, d+4)."""
    h = (tspan[1] - tspan[0]) / nt
    t = tspan[0]
    U = U0
    U_array = [U0.clone()]
    step = rk4_step if stepper == 'rk4' else rk1_step
    for _ in range(nt):
        U = step(phi, U, t, h, alpha, F_func, getDeltaF)
        t = t + h
        U_array.append(U.clone())
    return torch.stack(U_array, dim=0)


# ============================================================
#  Mean Field Game Solver
# ============================================================

class MeanFieldGame(nn.Module):
    """Mean Field Game solver.

    Learns the potential Phi(x,t;theta) to solve:
        min_theta  int int alpha_1 * L(x,-grad Phi) + alpha_2 * F(rho) dx dt
        + alpha_3 * G(rho(.,T)) dx
        + alpha_4 * int HJB_residual dx dt + alpha_5 * int HJB_final dx
    """

    def __init__(self, dim, hidden_dim, nTime, alpha, T=1.0, nt=2, stepper='rk4'):
        super().__init__()
        self.phi = PotentialNN(dim, hidden_dim, nTime, T)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float), requires_grad=False)
        self.d = dim
        self.TT = T
        self.nt = nt
        self.tspan = (0.0, T)
        self.stepper = stepper

    def forward(self, x0, w, rho0x, F_obs, F_ent, getDeltaF, G_func, getDeltaG):
        batch = x0.shape[0]
        dev = x0.device

        # Build wrapper that returns 2 columns for F_func: [dF_obs, dF_ent]
        def F_wrapper(U, t):
            return torch.cat([F_obs(U, t), F_ent(U, t)], dim=1)

        U0 = torch.zeros(batch, self.d + 5, device=dev)
        U0[:, :self.d] = x0
        U_final = integrate(self.phi, U0, self.tspan, self.nt, self.alpha,
                            F_wrapper, getDeltaF, self.stepper)

        costL = torch.dot(U_final[:, self.d + 1], w)
        costF_obs = torch.dot(U_final[:, self.d + 2], w)
        costF_ent = torch.dot(U_final[:, self.d + 3], w)
        g_vals = G_func(U_final)                           # = Pref(x_T) = -log p_target
        log_rho0x = torch.log(rho0x.clamp(min=1e-20))
        ldet = U_final[:, self.d]
        log_rho_T = log_rho0x - ldet                      # log rho_T(x_T)
        kl_T = log_rho_T + g_vals                         # KL per particle: log rho_T - log p_target
        costKL = torch.dot(kl_T, w)                       # full KL terminal cost
        costHJB = torch.dot(U_final[:, self.d + 4], w)

        # phi_T: delta G / delta rho = 1 + log rho_T - log p_target = 1 + kl_T
        XT_final = torch.cat([
            U_final[:, :self.d],
            torch.full((batch, 1), self.TT, device=dev)
        ], dim=1)
        phi_T = self.phi(XT_final).squeeze(-1)
        target_phiT = 1.0 + kl_T
        cost_phiT = torch.dot(torch.abs(phi_T - target_phiT), w)

        cs = torch.stack([costL, costF_obs, costF_ent, costKL, costHJB, cost_phiT])
        total = torch.dot(self.alpha, cs)

        self._costs = {
            'costL': costL.item(), 'costF_obs': costF_obs.item(),
            'costF_ent': costF_ent.item(), 'costKL': costKL.item(),
            'costHJB': costHJB.item(), 'cost_phiT': cost_phiT.item(), 'total': total.item(),
        }
        self._U_final = U_final
        return total

    def get_trajectories(self, x0, n_steps=None):
        """Generate trajectories by integrating dx/dt = -(1/alpha_1) grad_x Phi."""
        batch = x0.shape[0]
        dev = x0.device
        d = self.d
        if n_steps is None:
            n_steps = self.nt * 4
        h = self.TT / n_steps
        t = 0.0
        x = x0.detach()
        traj = [x.cpu()]
        for _ in range(n_steps):
            with torch.enable_grad():
                XT = torch.cat([x.detach(), torch.full((batch, 1), t, device=dev)], dim=1).requires_grad_(True)
                grad_phi = self.phi.get_grad_potential(XT, create_graph=False)
            x = (x + h * (-1.0 / self.alpha[0]) * grad_phi[:, :d]).detach()
            t = t + h
            traj.append(x.cpu())
        return torch.stack(traj, dim=0)

    def get_cost_breakdown(self):
        return dict(self._costs)
