from enum import Enum
from pathlib import Path


class ResultFolders(Enum):
    # LEG_100 = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "LEG_18_10_22_100"
    LEG_100 = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "LEG_2023_2"
    ARM_100 = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "ARM_18_10_22_100"
    ACROBAT_100 = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "ACROBAT_18_10_22_100"
    WALKING_100 = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "HUMANOID_10DOF_21-10-22_2"
    UPPER_LIMB_100 = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "UPPER_LIMB_XYZ_VARIABLES_21-10-22_22"

    ALL_LEG = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "ALL_LEG"
    ALL_ARM = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "ALL_ARM"
    ALL_ACROBAT = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "ALL_ACROBAT"

    LEG = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "LEG_30-09-22_2"
    ARM = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "ARM_30-09-22_2"
    ACROBAT = Path(__file__).parent.parent.parent.__str__() + "/dms-vs-dc-results/" + "ACROBAT_30-09-22_2"

    LEG_2022_08_12_TEST = "/home/puchaud/Projets_Python/dms-vs-dc-results/LEG_19-08-22_2"
    LEG_2022_07_19 = "/home/puchaud/Projets_Python/dms-vs-dc-results/raw_17-07-22"
    ARM_2022_07_29 = "/home/puchaud/Projets_Python/dms-vs-dc-results/arm_29-07-22"
    LEG_2022_08_01 = "/home/puchaud/Projets_Python/dms-vs-dc-results/LEG_01-08-22"
    ARM_2022_08_01_E6 = "/home/puchaud/Projets_Python/dms-vs-dc-results/ARM_01-08-22_2"
    ARM_2022_08_02_E10 = "/home/puchaud/Projets_Python/dms-vs-dc-results/ARM_02-08-22_2"
    MILLER = "/home/puchaud/Projets_Python/dms-vs-dc-results/miller_01-08-22"
    MILLER_2 = "/home/puchaud/Projets_Python/dms-vs-dc-results/ACROBAT_05-08-22"
    MILLER_TEST = "/home/puchaud/Projets_Python/dms-vs-dc-results/ACROBAT_16-08-22_2"


