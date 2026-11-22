"""DeepGSB: train a Schrodinger Bridge policy for swarm control.

Usage:
    python main_deepgsb.py [--problem-name Gauss2Gauss_Obs3] [--device cpu|cuda] [--x-dim-override 20]
"""
import logging
import pathlib
import sys

import torch

import options
from deepgsb import DeepGSB
from mfg import MFG
from options import Options

PROBLEM_NAME = "Gauss2Gauss_Obs3"
SEED = 42


def setup_logger(log_dir: pathlib.Path, problem_name: str, x_dim: int) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"{problem_name}_d{x_dim}.log"),
        ],
    )


def run(opt: Options):
    log = logging.getLogger(__name__)
    log.info("=" * 55)
    log.info("  Deep Generalized Schrodinger Bridge")
    log.info("=" * 55)
    log.info("Experiment: %s  |  Seed: %d", opt.problem_name, opt.seed)
    mfg = MFG(opt)
    deepgsb = DeepGSB(opt, mfg)
    deepgsb.train(opt)


def main():
    import warnings
    warnings.filterwarnings("ignore", message="ragged nested sequences")
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    opt = options.set(problem_name=PROBLEM_NAME, seed=SEED)

    run_dir = pathlib.Path(opt.run_dir)
    setup_logger(run_dir, opt.problem_name, opt.x_dim)

    run(opt)


if __name__ == "__main__":
    main()
