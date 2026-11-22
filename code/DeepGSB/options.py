"""Options parsing and experiment directory setup."""
import argparse
import os
import sys
import time
import torch
from envs import get_env_class


class Options:
    """Lightweight container populated from argparse + env defaults."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __repr__(self):
        items = sorted(f"{k}={v}" for k, v in self.__dict__.items() if k not in ('device', 'run_dir'))
        return f"Options({', '.join(items)})"
    def __getattr__(self, name):
        return None


def set(problem_name: str = None, seed: int = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-name", type=str, default='Gauss2Gauss_Obs3')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (default: auto-detect)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU (deprecated: use --device cpu)")
    parser.add_argument("--dir", type=str, default='run')
    parser.add_argument("--group", type=str, default='0')
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log-fn", type=str, default=None)
    parser.add_argument("--log-tb", action="store_true")
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--policy-net", type=str)
    parser.add_argument("--diffusion-std", type=float, default=1.0)
    parser.add_argument("--sb-param", type=str, default='actor-critic', choices=['actor-critic', 'critic'])
    parser.add_argument("--train-bs-x", type=int, default=128)
    parser.add_argument("--num-stage", type=int, default=5)
    parser.add_argument("--num-itr", type=int, default=100)
    parser.add_argument("--samp-bs", type=int, default=5000)
    parser.add_argument("--samp-method", type=str, default='jacobi', choices=['jacobi', 'gauss'])
    parser.add_argument("--rb-bs-x", type=int)
    parser.add_argument("--use-rb-loss", action="store_true")
    parser.add_argument("--multistep-td", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--weighted-loss", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-y", type=float, default=None)
    parser.add_argument("--lr-gamma", type=float, default=1.0)
    parser.add_argument("--lr-step", type=int, default=1000)
    parser.add_argument("--l2-norm", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--noise-type", type=str, default='gaussian', choices=['gaussian', 'rademacher'])
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--snapshot-freq", type=int, default=1)
    parser.add_argument("--ckpt-freq", type=int, default=0)
    parser.add_argument("--lambda-L", type=float, default=0.05)
    parser.add_argument("--lambda-E", type=float, default=1.0)
    parser.add_argument("--lambda-P", type=float, default=500.0)
    parser.add_argument("--obs-weight", type=float, default=1500.0)
    parser.add_argument("--x-dim-override", type=int, default=None,
                        help="Override spatial dimension for HD experiments")
    parser.add_argument("--no-env-config", action="store_true",
                        help="Skip env defaults, use CLI args only")

    if problem_name is not None:
        parser.set_defaults(problem_name=problem_name)
    if seed is not None:
        parser.set_defaults(seed=seed)

    parsed_args = parser.parse_args()
    use_env_config = not getattr(parsed_args, 'no_env_config', False)

    if use_env_config:
        env_class = get_env_class(parsed_args.problem_name)
        parser.set_defaults(**env_class.get_default_config(parsed_args.sb_param))
        if problem_name is not None:
            parser.set_defaults(problem_name=problem_name)
        if seed is not None:
            parser.set_defaults(seed=seed)
        parsed_args = parser.parse_args()

    opt = parsed_args

    if opt.x_dim_override is not None:
        opt.x_dim = opt.x_dim_override

    # Device selection: --device overrides --cpu and auto-detection
    if opt.device is not None:
        pass
    elif opt.cpu:
        opt.device = "cpu"
    elif torch.cuda.is_available():
        opt.device = "cuda:0"
    else:
        opt.device = "cpu"
    if opt.device.startswith("cuda"):
        import re
        dev_id = int(re.sub(r'[^0-9]', '', opt.device) or 0)
        torch.cuda.set_device(dev_id)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if opt.name is None:
        opt.name = opt.dir

    timestamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    x_dim = getattr(opt, 'x_dim', 2)
    main_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    opt.run_dir = os.path.join(main_dir, "Experiments", f"{opt.problem_name}_d{x_dim}_{timestamp}")
    os.makedirs(opt.run_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(opt.run_dir, "config.txt")
    with open(config_path, "w") as f:
        for key, val in sorted(vars(opt).items()):
            if key in ("device", "run_dir", "eval_path", "model_path", "log_dir", "load"):
                continue
            f.write(f"{key}: {val}\n")

    opt.model_path = os.path.join(opt.run_dir, "models")
    os.makedirs(opt.model_path, exist_ok=True)

    if opt.snapshot_freq:
        opt.eval_path = os.path.join(opt.run_dir, "plots")
        os.makedirs(opt.eval_path, exist_ok=True)

    opt.log_dir = opt.run_dir
    opt.load = None
    opt.use_env_config = use_env_config

    return Options(**vars(opt))
