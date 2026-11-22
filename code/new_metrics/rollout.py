"""Rollout builders for DeepGSB / MFG-Net / PINN (Gauss2Gauss_Obs3).

All three share the same initial particles and the same deterministic ODE
integration (dt=0.01, 100 steps), so differences come only from the policy.

    python rollout.py --algo deepgsb --ns 20 500      # generate trajectories
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # rebuttal root
DT, STEPS = 0.01, 100

# experiment dirs shipped with the rebuttal package
RUNS = {
    ("deepgsb", 20): "DeepGSB/Experiments/Gauss2Gauss_Obs3_d20_2026-06-20__01-37-47",
    ("deepgsb", 50): "DeepGSB/Experiments/Gauss2Gauss_Obs3_d50_2026-06-20__02-02-32",
    ("mfgnet", 20): "MFG-Net/Experiments/Gauss2Gauss_Obs3_d20_2026-06-20__00-46-03",
    ("mfgnet", 50): "MFG-Net/Experiments/Gauss2Gauss_Obs3_d50_2026-06-20__01-46-36",
    ("pinn", 20): "PINN/Experiments/Gauss2Gauss_Obs3_d20_2026-06-20__00-32-21",
    ("pinn", 50): "PINN/Experiments/Gauss2Gauss_Obs3_d50_2026-06-20__00-56-18",
}


# the three projects export same-named top-level packages (models, envs, ...),
# so switching project must evict them or the first import wins forever
_PKGS = {"models", "envs", "evaluator", "mfg", "deepgsb"}


def _use(project):
    """Put <root>/<project> first on sys.path and drop cached same-named packages."""
    sys.path.insert(0, os.path.join(ROOT, project))
    for name in list(sys.modules):
        if name.split(".")[0] in _PKGS:
            del sys.modules[name]


def init_x(dim, n, seed):
    """Shared start: N((-3,-2,0..), 0.1*I). Same for every algorithm."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim, dtype=np.float32)
    mu[:2] = (-3.0, -2.0)
    return (mu + np.sqrt(0.1) * rng.standard_normal((n, dim))).astype(np.float32)


def target(dim):
    t = np.zeros(dim, dtype=np.float32)
    t[:2] = (4.0, 4.0)
    return t


# ---- per-algorithm rollouts: f(x0) -> (traj (N,T+1,d), u (N,T,d)) ----------

def _deepgsb(dim, run_dir):
    _use("DeepGSB")
    from evaluator import load_deepgsb, DeepGSBEvaluator
    from envs.Gauss2Gauss_Obs3 import Gauss2Gauss_Obs3

    dg = load_deepgsb(run_dir, device="cpu")
    ev = DeepGSBEvaluator(dg, Gauss2Gauss_Obs3(device=torch.device("cpu"), d=dim), device="cpu")

    @torch.no_grad()
    def run(x0, n_steps):
        x = torch.tensor(x0)
        tr, us = [x.numpy().copy()], []
        for k in range(n_steps):
            u = ev.get_pf_ode_drift(x, torch.full((x.shape[0], 1), k * DT))
            us.append(u.numpy().copy())
            x = x + u * DT
            tr.append(x.numpy().copy())
        return np.stack(tr, 1), np.stack(us, 1)
    return run


def _mfgnet(dim, run_dir):
    _use("MFG-Net")
    from models.mfg import MeanFieldGame

    ck = torch.load(os.path.join(run_dir, "models", "mfg_final.pt"), map_location="cpu")
    a = ck["architecture"]
    mfg = MeanFieldGame(dim=a["d"], hidden_dim=a["m"], nTime=a["nTime"],
                        alpha=a["alpha"], T=a["TT"], nt=a["nt"], stepper=a["stepper"])
    mfg.load_state_dict(ck["mfg_state_dict"])
    mfg.eval()
    a0, d_ = mfg.alpha[0].item(), a["d"]
    h = a["TT"] / STEPS

    @torch.no_grad()
    def run(x0, n_steps):
        x = torch.tensor(x0)
        tr, us = [x.numpy().copy()], []
        for k in range(n_steps):
            g = mfg.phi.get_grad_potential(
                torch.cat([x, torch.full((x.shape[0], 1), k * h)], 1), create_graph=False)
            u = -(1.0 / a0) * g[:, :d_]
            us.append(u.numpy().copy())
            x = x + h * u
            tr.append(x.numpy().copy())
        return np.stack(tr, 1), np.stack(us, 1)
    return run


def _pinn(dim, run_dir):
    _use("PINN")
    from models.mlp import MLP
    from models.pinn_solver import PINNSolver
    from envs.Gauss2Gauss_Obs3 import Gauss2Gauss_Obs3

    with open(os.path.join(run_dir, "models", "pinn_net_metadata.json")) as f:
        meta = json.load(f)
    sc = Gauss2Gauss_Obs3(device=torch.device("cpu"), d=dim)

    def fG(inp, out):                    # terminal-condition transform
        return out * (1 - inp[:, -1:]) + sc.terminal_cost(inp[:, :-1], delta=1.0).unsqueeze(1) * inp[:, -1:]

    net = MLP(layer_sizes=meta["layer_sizes"], activation=torch.tanh, device="cpu")
    net.add_output_transform(fG)
    net.load_state_dict(torch.load(os.path.join(run_dir, "models", "pinn_net.pth"), map_location="cpu"))
    net.eval()
    solver = PINNSolver(lambda_L=sc.lambda_L, obstacles=sc.obstacle_dicts, endpoint_B=sc.end_point,
                        batch_size=1, model=net, sigma=meta["sigma"], T=sc.TT,
                        device="cpu", lambda_P=sc.lambda_P)

    def run(x0, n_steps):                # get_control needs grad enabled
        x = torch.tensor(x0)
        tr, us = [x.numpy().copy()], []
        for k in range(n_steps):
            u = solver.get_control(x, torch.full((x.shape[0], 1), k * DT))
            us.append(u.detach().numpy().copy())
            x = (x + u.detach() * DT).detach()
            tr.append(x.numpy().copy())
        return np.stack(tr, 1), np.stack(us, 1)
    return run


BUILD = {"deepgsb": _deepgsb, "mfgnet": _mfgnet, "pinn": _pinn}


def build(algo, dim, run_dir=None):
    return BUILD[algo](dim, run_dir or os.path.join(ROOT, RUNS[(algo, dim)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=list(BUILD))
    ap.add_argument("--dims", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--ns", type=int, nargs="+", default=[20, 50, 100, 200, 500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for dim in args.dims:
        print(f"[{args.algo} d{dim}] building ...", flush=True)
        run = build(args.algo, dim)
        for n in args.ns:
            for s in args.seeds:
                out = os.path.join(args.out_dir, f"traj_{args.algo}_d{dim}_N{n}_s{s}.npz")
                if os.path.exists(out):
                    continue
                t0 = time.time()
                tr, u = run(init_x(dim, n, s), STEPS)
                np.savez_compressed(out, traj=tr.astype(np.float32), u=u.astype(np.float32),
                                    algo=args.algo, dim=dim, n=n, seed=s)
                print(f"  {os.path.basename(out)} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
