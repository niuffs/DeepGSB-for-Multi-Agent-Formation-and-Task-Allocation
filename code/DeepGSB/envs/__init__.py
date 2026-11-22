"""Environment registry for DeepGSB."""

from .Gauss2Gauss_Obs3 import Gauss2Gauss_Obs3
from .Gauss2Ring_Obs1 import Gauss2Ring_Obs1
from .Gauss2Rings_Obs1 import Gauss2Rings_Obs1
from .MixGauss2SphereShell_Obs3 import MixGauss2SphereShell_Obs3

ENV_REGISTRY = {
    "Gauss2Gauss_Obs3": Gauss2Gauss_Obs3,
    "Gauss2Ring_Obs1": Gauss2Ring_Obs1,
    "Gauss2Rings_Obs1": Gauss2Rings_Obs1,
    "MixGauss2SphereShell_Obs3": MixGauss2SphereShell_Obs3,
}


def get_env_class(problem_name: str):
    """Look up an env class by name (case-insensitive)."""
    name = problem_name.lower()
    for key, cls in ENV_REGISTRY.items():
        if key.lower() == name:
            return cls
    raise ValueError(f"Unknown problem: {problem_name}")
