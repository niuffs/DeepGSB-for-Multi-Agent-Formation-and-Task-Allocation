import torch
import torch.nn as nn
import math
from .activation import msigma, linInter1D


class SingleLayer(nn.Module):
    """Single neural network layer: sigma(Kx + b)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.K = nn.Parameter(0.01 * torch.randn(out_features, in_features))
        self.b = nn.Parameter(0.1 * torch.randn(out_features))

    def forward(self, x):
        return msigma(torch.mm(x, self.K.T) + self.b)


class ResNN(nn.Module):
    """Residual Neural Network with time-interpolated weights.

    For each time sub-interval:
        x <- x + h_k * sigma(K(t_k) * x + b(t_k))

    where K(t), b(t) are linearly interpolated from a time grid.
    """

    def __init__(self, in_features, hidden_features, nTime, T=1.0):
        super().__init__()
        self.K = nn.Parameter(0.01 * torch.randn(nTime, hidden_features, in_features))
        self.b = nn.Parameter(0.1 * torch.randn(nTime, hidden_features))
        self.TT = T
        self.nTime = nTime
        self.register_buffer('ts', torch.linspace(0, T, nTime))

    @property
    def nLayers(self):
        return self.nTime - 1

    def forward(self, x):
        for k in range(self.nLayers):
            hk = self.ts[k + 1] - self.ts[k]
            K_k = linInter1D(self.ts[k], self.TT, self.K)
            b_k = linInter1D(self.ts[k], self.TT, self.b)
            x = x + hk * msigma(torch.mm(x, K_k.T) + b_k)
        return x


class NN(nn.Module):
    """Sequential stack of layers (SingleLayer and/or ResNN)."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PotentialNN(nn.Module):
    """Potential function approximator for mean field games.

    Phi(x,t) = w' * NN([x; t]) + 0.5 * [x;t]' * A * [x;t] + c' * [x;t] + z

    The NN consists of:
        SingleLayer(d+1 -> m)
        ResNN(m -> m with nTh-1 layers)
    """

    def __init__(self, dim, hidden_dim, nTime, T=1.0):
        super().__init__()
        input_dim = dim + 1  # space + time

        # NN layers
        self.K1 = nn.Parameter(0.01 * torch.randn(hidden_dim, input_dim))
        self.b1 = nn.Parameter(0.1 * torch.randn(hidden_dim))
        self.K2 = nn.Parameter(0.01 * torch.randn(nTime, hidden_dim, hidden_dim))
        self.b2 = nn.Parameter(0.1 * torch.randn(nTime, hidden_dim))

        # Output weights (quadratic part)
        self.w = nn.Parameter(torch.ones(hidden_dim))
        self.A = nn.Parameter(torch.zeros(input_dim, input_dim))
        self.c = nn.Parameter(torch.zeros(input_dim))
        self.z = nn.Parameter(torch.zeros(1))

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.nTime = nTime
        self.TT = T
        self.register_buffer('ts', torch.linspace(0, T, nTime))

    def forward(self, XT):
        """Evaluate Phi at points (x, t).

        Args:
            XT: (batch, d+1), last column is time t

        Returns:
            (batch,) Phi values
        """
        # Layer 1: SingleLayer
        out = msigma(torch.mm(XT, self.K1.T) + self.b1)  # (batch, m)

        # Layer 2: ResNN
        for k in range(self.nTime - 1):
            hk = self.ts[k + 1] - self.ts[k]
            K_k = linInter1D(self.ts[k], self.TT, self.K2)
            b_k = linInter1D(self.ts[k], self.TT, self.b2)
            out = out + hk * msigma(torch.mm(out, K_k.T) + b_k)  # (batch, m)

        # Quadratic output: w' * NN(XT) + 0.5 * XT' * A * XT + c' * XT + z
        A_sym = 0.5 * (self.A + self.A.T)
        quad = 0.5 * torch.sum((XT @ A_sym.T) * XT, dim=1)  # (batch,)
        linear = torch.mv(XT, self.c)  # (batch,)

        return torch.mv(out, self.w) + quad + linear + self.z

    def _compute_grad(self, XT, create_graph=True):
        """Compute gradient of Phi w.r.t. inputs, handling no_grad context."""
        if not torch.is_grad_enabled():
            with torch.enable_grad():
                XT_in = XT.detach().requires_grad_(True)
                phi_val = self.forward(XT_in)
                return torch.autograd.grad(phi_val.sum(), XT_in, create_graph=False)[0]
        else:
            XT_in = XT.detach().requires_grad_(True)
            phi_val = self.forward(XT_in)
            return torch.autograd.grad(phi_val.sum(), XT_in, create_graph=create_graph)[0]

    def get_grad_potential(self, XT, create_graph=True):
        """Compute gradient of Phi w.r.t. inputs (x, t).

        Args:
            XT: (batch, d+1)
            create_graph: whether to retain graph (True for training, False for eval)

        Returns:
            grad: (batch, d+1)
        """
        return self._compute_grad(XT, create_graph=create_graph)

    def get_velocity(self, XT):
        """Extract velocity field v = -(1/alpha_1) * grad_x Phi and dPhi/dt from grad Phi.

        Args:
            XT: (batch, d+1)

        Returns:
            vel: (batch, d)
            dPhi_dt: (batch,)
        """
        grad = self._compute_grad(XT, create_graph=torch.is_grad_enabled())
        vel = grad[:, :self.dim]
        dPhi_dt = grad[:, self.dim]
        return vel, dPhi_dt

    def get_grad_and_hessian(self, XT):
        """Compute gradient and Hessian of Phi w.r.t. spatial x.

        Args:
            XT: (batch, d+1)

        Returns:
            grad: (batch, d+1) gradient w.r.t. all inputs
            hess: (batch, d+1, d+1) full Hessian
        """
        grad = self._compute_grad(XT, create_graph=True)

        hess = []
        for i in range(XT.shape[1]):
            g_i = torch.autograd.grad(grad[:, i].sum(), XT,
                                       retain_graph=True, create_graph=True)[0]
            hess.append(g_i.unsqueeze(1))
        hess = torch.cat(hess, dim=1)  # (batch, d+1, d+1)

        return grad, hess

    def get_trace_hessian(self, XT):
        """Compute trace of spatial Hessian tr(grad^2_xx Phi).

        Args:
            XT: (batch, d+1)

        Returns:
            trace: (batch,) tr(grad^2_xx Phi)
        """
        grad = self._compute_grad(XT, create_graph=True)
        trace = torch.zeros(XT.shape[0], device=XT.device)
        d = self.dim
        for i in range(d):
            hess_i = torch.autograd.grad(grad[:, i].sum(), XT,
                                          retain_graph=True, create_graph=True)[0]
            trace = trace + hess_i[:, i]
        return trace
