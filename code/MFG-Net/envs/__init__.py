"""Environment registry for MFG-Net."""

from envs.Gauss2Gauss_Obs3 import Gauss2Gauss_Obs3
from envs.Gauss2Ring_Obs1 import Gauss2Ring_Obs1
from envs.Gauss2Rings_Obs1 import Gauss2Rings_Obs1
from envs.MixGauss2SphereShell_Obs3 import MixGauss2SphereShell_Obs3
from envs.Ring2Gauss_Obs1 import Ring2Gauss_Obs1
from envs.Rings2Gauss_Obs1 import Rings2Gauss_Obs1
from envs.SphereShell2MixGauss_Obs3 import SphereShell2MixGauss_Obs3

env_dict = {
    'Gauss2Gauss_Obs3': Gauss2Gauss_Obs3,
    'Gauss2Ring_Obs1': Gauss2Ring_Obs1,
    'Gauss2Rings_Obs1': Gauss2Rings_Obs1,
    'MixGauss2SphereShell_Obs3': MixGauss2SphereShell_Obs3,
    'Ring2Gauss_Obs1': Ring2Gauss_Obs1,
    'Rings2Gauss_Obs1': Rings2Gauss_Obs1,
    'SphereShell2MixGauss_Obs3': SphereShell2MixGauss_Obs3,
}
