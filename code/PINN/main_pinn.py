"""
PINN main entry point: train and evaluate PINN on a given scenario.
Single-agent optimal control via physics-informed neural networks.

Supports d=2,3,10 with zero-padding for extra dimensions.
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch

from envs.Gauss2Gauss_Obs3 import Gauss2Gauss_Obs3
from models.mlp import MLP
from models.pinn_solver import PINNSolver
from evaluator import PINNEvaluator


# ================================================================
#  Default hyperparameters
# ================================================================

_hp = {
    "seed": 42,
    "activation": torch.tanh,
    "iterations": 5000,                         # Adam iterations for PDE loss
    "batch_size": 500,                          # Collocation points per iteration
    "lr": 1e-3,                                 # Learning rate
    "sigma": 1.0,                              # Noise intensity for HJB PDE (matches DeepGSB diffusion_std)
    "eval_num_repeats": 5,                     # Evaluation repeats (matches DeepGSB/MFG-Net)
    "lambda_E": 0.1,                            # Congestion weight (reference only)
}


# ================================================================
#  Utility functions
# ================================================================

def _get_time_string():
    out = str(datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
    return out[:10] + '__' + out[11:]


def _create_experiment_dir(env_name, dim):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Experiments')
    run_name = f'{env_name}_d{dim}_{_get_time_string()}'
    exp_dir = os.path.join(base_dir, run_name)
    for sub in ['models', 'plots', 'pinn_eval']:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    return exp_dir


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  Random seed set to: {seed} (device: {device})")


# ================================================================
#  Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="PINN for optimal control")
    parser.add_argument("--dim", type=int, default=2,
                        help="State space dimension (default: 2). Extra dims are zero-padded.")
    parser.add_argument("--problem-name", type=str, default="Gauss2Gauss_Obs3",
                        help="Scenario name (default: Gauss2Gauss_Obs3)")
    parser.add_argument("--iterations", type=int, default=_hp["iterations"],
                        help=f"Training iterations (default: {_hp['iterations']})")
    parser.add_argument("--batch-size", type=int, default=_hp["batch_size"],
                        help=f"Collocation batch size (default: {_hp['batch_size']})")
    parser.add_argument("--lr", type=float, default=_hp["lr"],
                        help=f"Learning rate (default: {_hp['lr']})")
    parser.add_argument("--seed", type=int, default=_hp["seed"],
                        help=f"Random seed (default: {_hp['seed']})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect cuda:0 or cpu)")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Arguments: {args}")

    d = args.dim
    _hp["iterations"] = args.iterations
    _hp["batch_size"] = args.batch_size
    _hp["lr"] = args.lr
    _hp["seed"] = args.seed

    # Scenario
    if args.problem_name == "Gauss2Gauss_Obs3":
        scenario = Gauss2Gauss_Obs3(device=device, d=d)
    else:
        raise ValueError(f"Unknown problem: {args.problem_name}")
    print(f"Scenario: {scenario.info_dict}")

    # Experiment directory
    exp_dir = _create_experiment_dir(scenario.name, scenario.dim)
    exp_models = os.path.join(exp_dir, "models")
    exp_plots = os.path.join(exp_dir, "plots")
    exp_eval = os.path.join(exp_dir, "pinn_eval")
    print(f"Saving results to: {exp_dir}")

    # Log file
    log_path = os.path.join(exp_dir, f'{scenario.name}.log')
    class _Tee:
        def __init__(self, *files): self.files = files
        def write(self, data):
            for f in self.files: f.write(data)
        def flush(self):
            for f in self.files: f.flush()
    sys.stdout = _Tee(sys.stdout, open(log_path, 'w', buffering=1))

    # Reproducibility
    set_seed(_hp["seed"], device)

    # Build network with terminal condition transform
    def fG(inputs, outputs):
        """Enforce terminal condition: phi(x, T) = Huber(||x - B||)."""
        return outputs * (1 - inputs[:, -1:]) \
               + scenario.terminal_cost(inputs[:, :-1], delta=1.0).unsqueeze(1) * inputs[:, -1:]

    layer_sizes = [d + 1, 240, 240, 240, 1]  # Input: (x1,...,xd, t), Output: phi
    pinn_net = MLP(
        layer_sizes=layer_sizes,
        activation=_hp["activation"],
        device=device,
    )
    pinn_net.add_output_transform(fG)

    # Save single config.txt
    config_summary = {**_hp,
        "layer_sizes": layer_sizes,
        "dim": scenario.dim, "T": scenario.TT, "sigma": _hp["sigma"],
        "lambda_L": scenario.lambda_L, "lambda_E": scenario.lambda_E,
        "lambda_P": scenario.lambda_P,
        "start_point": scenario.start_point.tolist(),
        "end_point": scenario.end_point.tolist(),
    }
    config_summary["activation"] = config_summary["activation"].__name__ if callable(config_summary["activation"]) else str(config_summary["activation"])
    with open(os.path.join(exp_dir, "config.txt"), "w") as f:
        for key, val in sorted(config_summary.items()):
            f.write(f"{key}: {val}\n")
        f.write(f"env_info: {scenario.info_dict}\n")

    # Build PINN solver
    solver = PINNSolver(
        lambda_L=scenario.lambda_L,
        obstacles=scenario.obstacle_dicts,
        endpoint_B=scenario.end_point,
        batch_size=_hp["batch_size"],
        model=pinn_net,
        sigma=_hp["sigma"],
        T=scenario.TT,
        lr=_hp["lr"],
        device=device,
        save_path=exp_plots,
        lambda_P=scenario.lambda_P,
    )
    solver.name = scenario.name

    # Train
    solver.train(iterations=_hp["iterations"], log_freq=50)

    # Post-training plots
    print("\nPlotting training loss...")
    solver.plot_loss(save_path=exp_plots)

    print("Plotting value function heatmaps...")
    solver.plot_phi(save_path=exp_plots)

    print("Plotting sample trajectories...")
    for startpoint in [
        torch.tensor((2, 0), device=device),
        torch.tensor((0, 2), device=device),
        torch.tensor((0, 0), device=device),
        torch.tensor((-3, -2), device=device),
    ]:
        solver.plot_samples(startpoint, save_path=exp_plots)

    # Post-training: cost evaluation (includes 2D/3D trajectory plots)
    evaluator = PINNEvaluator(
        solver=solver, scenario=scenario, device=device,
    )
    # Use standard lambdas for weighted cost summary (fair comparison with DeepGSB/MFG-Net)
    evaluator.lambda_L = 0.05
    evaluator.lambda_E = 0.05
    evaluator.lambda_P = 10.0
    evaluator.evaluate(save_path=exp_eval, num_repeats=_hp["eval_num_repeats"])

    # Save trained model
    print("\nSaving trained model...")
    torch.save(pinn_net.state_dict(), os.path.join(exp_models, "pinn_net.pth"))
    meta_path = os.path.join(exp_models, "pinn_net_metadata.json")
    activation_name = _hp["activation"].__name__ if callable(_hp["activation"]) else str(_hp["activation"])
    with open(meta_path, "w") as f:
        json.dump({
            "name": "pinn_net", "type": "MLP",
            "layer_sizes": layer_sizes, "activation": activation_name,
            "sigma": _hp["sigma"], "lambda_L": scenario.lambda_L,
            "lambda_E": scenario.lambda_E,
            "lambda_P": scenario.lambda_P,
            "T": scenario.TT, "dim": scenario.dim, "lr": _hp["lr"],
            "iters": _hp["iterations"], "batch_size": _hp["batch_size"],
            "start_point": scenario.start_point.tolist(),
            "end_point": scenario.end_point.tolist(),
        }, f, indent=2, default=str)
    print(f"  Saved: pinn_net.pth + pinn_net_metadata.json")
    print(f"All models saved to: {exp_models}")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
