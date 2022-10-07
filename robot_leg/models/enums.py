from enum import Enum
from pathlib import Path


class Models(Enum):
    LEG = Path(__file__).parent.__str__() + "/hexapod_leg.bioMod"
    ARM = Path(__file__).parent.__str__() + "/robot_arm.bioMod"
    ACROBAT = Path(__file__).parent.__str__() + "/acrobat.bioMod"

    UPPER_LIMB_QUAT_TEMPLATE = (
        Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_quat_degroote.bioMod"
    )
    UPPER_LIMB_QUAT_VARIABLES = (
        Path(__file__).parent.__str__()
        + "/wu_converted_definitif_without_floating_base_template_quat_with_variables_degroote.bioMod"
    )
    UPPER_LIMB_XYZ_TEMPLATE = (
        Path(__file__).parent.__str__() + "/wu_converted_definitif_without_floating_base_template_xyz_offset.bioMod"
    )
    UPPER_LIMB_XYZ_VARIABLES = (
        Path(__file__).parent.__str__()
        + "/wu_converted_definitif_without_floating_base_template_xyz_offset_with_variables.bioMod"
    )
