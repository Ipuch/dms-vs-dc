from enum import Enum
from pathlib import Path


class Models(Enum):

    LEG = Path(__file__).parent.__str__() + "/hexapod_leg.bioMod"
    ARM = Path(__file__).parent.__str__() + "/robot_arm.bioMod"
