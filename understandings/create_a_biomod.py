import biorbd_casadi as biorbd
import numpy as np


# Defining two identical segments for testing
name_first = "First_segment"
name_second = "Second_segment"
parent_first = "ROOT"
parent_second = "Pelvis"
translations = "xyz"
rotations = "xyz"
ranges = []
roto_trans = biorbd.RotoTrans(np.array([1, 2, 3]), np.array([4, 5, 6]), "xyz")
mass = 10
com = biorbd.Vector3d(1, 2, 3)
inertia = biorbd.Matrix3d(1, 2, 3, 4, 5, 6, 7, 8, 9)
char = biorbd.SegmentCharacteristics(mass, com, inertia)

# Create an empty model and add the two segments
lukas = biorbd.Model()
lukas.AddSegment(name_first, parent_first, translations, rotations, ranges, ranges, ranges, char, roto_trans)
lukas.AddSegment(name_second, parent_second, translations, rotations, ranges, ranges, ranges, char, roto_trans)

# Write the model and read back
biorbd.Writer().writeModel(lukas, "coucou.bioMod")
lukas_coucou = biorbd.Model("coucou.bioMod")
