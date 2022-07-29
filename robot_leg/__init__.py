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
from .bioptim_plugin.integration_function import Integration
from .models.enums import Models