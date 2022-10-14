from enum import Enum
from leg_ocp import LegOCP
from arm_ocp import ArmOCP
from miller_ocp import MillerOcp as MillerOCP
from upper_limb import UpperLimbOCP
from humanoid_ocp_multiphase import HumanoidOcpMultiPhase as HumanoidOCP


class OCP(Enum):

    LEG = (LegOCP,)
    ARM = (ArmOCP,)
    ACROBAT = (MillerOCP,)
    UPPER_LIMP = (UpperLimbOCP,)
    HUMANOID_10DOF = (HumanoidOCP,)
