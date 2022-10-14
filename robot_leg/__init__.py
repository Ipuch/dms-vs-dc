"""
robot_leg is a package for the Humanoid2D model.
-------------------------------------------------------------------------


# --- The main optimal control programs --- #
HumanoidOcp

HumanoidOcpMultiPhase

"""
VERSION = "0.1.0"
from .ocp.viz import add_custom_plots

from .ocp.leg_ocp import LegOCP
from .ocp.arm_ocp import ArmOCP
from .ocp.miller_ocp import MillerOcp as MillerOCP
from .ocp.upper_limb import UpperLimbOCP
from .ocp.miller_ocp_one_phase import MillerOcpOnePhase
from .ocp.humanoid_ocp_multiphase import HumanoidOcpMultiPhase as HumanoidOCP

from .models.enums import Models
