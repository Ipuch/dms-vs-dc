from enum import Enum
from leg_ocp import LegOCP
from arm_ocp import ArmOCP
from miller_ocp import MillerOcp as MillerOCP


class OCP(Enum):

    LEG = (LegOCP,)
    ARM = (ArmOCP,)
    ACROBAT = (MillerOCP,)
