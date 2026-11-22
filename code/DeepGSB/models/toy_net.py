import torch
import torch.nn as nn

from options import Options
from .util import SiLU, timestep_embedding


class ToyY(torch.nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, time_embed_dim: int = 128):
        super(ToyY, self).__init__()
        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = nn.Sequential(
            nn.Linear(data_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.out_module = nn.Sequential(
            nn.Linear(hid + hid, hid),
            SiLU(),
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t[None]
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        xt_out = torch.cat([x_out, t_out], dim=1)
        out = self.out_module(xt_out)
        return out


class ToyZ(torch.nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, time_embed_dim: int = 128):
        super(ToyZ, self).__init__()
        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = nn.Sequential(
            nn.Linear(data_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t[None]
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out = self.out_module(x_out + t_out)
        return out


def build_toy_net_policy(opt: Options, YorZ: str) -> torch.nn.Module:
    """Build ToyY or ToyZ network with dimension from opt."""
    data_dim = getattr(opt, 'x_dim', 2)
    if YorZ == "Y":
        return ToyY(data_dim=data_dim)
    elif YorZ == "Z":
        return ToyZ(data_dim=data_dim)
    raise ValueError(f"Unknown YorZ: {YorZ}")
