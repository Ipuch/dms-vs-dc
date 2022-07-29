import biorbd
import numpy as np
from pyomeca import Rototrans, Angles


class Segment:
    def __init__(
        self,
        name: str,
        parent: str,
        rtinmatrix: bool,
        rt: np.array,
        xyz: np.array,
        translations: str,
        rotations: str,
        ranges: np.array,
        mass: float,
        com: np.array,
        inertia: np.ndarray,
        mesh: np.array,
    ):
        self.name = name
        if parent is None:
            self.parent = None
        else:
            self.parent = parent
        self.rtinmatrix = rtinmatrix
        self.rt = rt
        self.xyz = xyz
        self.roto_trans = biorbd.RotoTrans(rt, xyz, "xyz")
        self.rt = rt
        self.rotations = rotations
        if translations is None:
            self.translations = ""
        else:
            self.translations = translations

        if len(ranges.shape) == 1:
            ranges = ranges[:, np.newaxis].T
        self.ranges = ranges
        self.rangesQ = [biorbd.Range(ii[0], ii[1]) for ii in ranges]
        self.rangesQdot = [biorbd.Range(ii[0] * 10, ii[1] * 10) for ii in ranges]
        self.rangesQddot = [biorbd.Range(ii[0] * 100, ii[1] * 100) for ii in ranges]
        self.mass = mass
        self.com = biorbd.Vector3d(float(com[0]), float(com[1]), float(com[2]))
        self.inertia = biorbd.Matrix3d(
            float(inertia[0, 0]),
            float(inertia[0, 1]),
            float(inertia[0, 2]),
            float(inertia[1, 0]),
            float(inertia[1, 1]),
            float(inertia[1, 2]),
            float(inertia[2, 0]),
            float(inertia[2, 1]),
            float(inertia[2, 2]),
        )
        self.char = biorbd.SegmentCharacteristics(self.mass, self.com, self.inertia)

        self.mesh = mesh


class Contact:
    def __init__(
        self,
        name: str,
        parent: str,
        position: np.array,
        axis: str,
        acc: np.array = 0,
    ):
        self.name = name
        if parent is None:
            self.parent = None
        else:
            self.parent = parent
        self.position = position
        self.axis = axis
        self.acc = acc


class Marker:
    def __init__(
        self,
        name: str,
        parent: str,
        position: np.array,
        technical: bool = True,
        anatomical: bool = False,
        axis_to_remove: str = "",
    ):
        self.name = name
        if parent is None:
            self.parent = None
        else:
            self.parent = parent
        self.position = position
        self.technical = technical
        self.anatomical = anatomical
        self.axis_to_remove = axis_to_remove


def huygens(inertia, mass, com, new_com):
    d = new_com - com
    a = d[0]
    b = d[1]
    c = d[2]
    return inertia + mass * np.array(
        [[b**2 + c**2, -a * b, -a * c], [-a * b, a**2 + c**2, -b * c], [-a * c, -b * c, a**2 + b**2]]
    )


def merge_segment(Segment_master, Segment_child):
    # Compute the homogenous transform from child to master segment
    angles = Angles(Segment_child.rt[:, np.newaxis, np.newaxis])
    translations = Angles(Segment_child.xyz[:, np.newaxis, np.newaxis])
    roto_trans = Rototrans.from_euler_angles(angles=angles, angle_sequence="xyz", translations=translations)
    T = roto_trans.values[:, :, 0]

    # Compute the child's com in master's frame
    child_com = Segment_child.com.to_array()[:, np.newaxis]
    child_com_in_master = np.matmul(T, np.concatenate([child_com, np.ones((1, 1))], axis=0))
    child_com_in_master = np.delete(child_com_in_master, 3)

    # weighted mean to compute the new com
    new_com = np.average(
        np.concatenate((Segment_master.com.to_array()[:, np.newaxis], child_com_in_master[:, np.newaxis]), axis=1),
        weights=[Segment_master.mass, Segment_child.mass],
        axis=1,
    )

    # compute the new inertia matrix at the new com
    new_inertia = huygens(
        Segment_master.inertia.to_array(), Segment_master.mass, Segment_master.com.to_array(), new_com
    ) + huygens(Segment_child.inertia.to_array(), Segment_child.mass, child_com_in_master, new_com)

    # mesh in the master frame
    n_mesh = Segment_child.mesh.shape[0]
    mesh_in_master_frame = np.matmul(T, np.concatenate([Segment_child.mesh.T, np.ones((1, n_mesh))], axis=0))
    mesh_in_master_frame = np.delete(mesh_in_master_frame, 3, axis=0).T
    all_mesh = np.concatenate((Segment_master.mesh, mesh_in_master_frame))

    # create the merged segment
    merge_seg = Segment(
        name=Segment_master.name,
        parent=Segment_master.parent,
        rtinmatrix=False,
        rt=Segment_master.rt,
        xyz=Segment_master.xyz,
        translations=Segment_master.translations,
        rotations=Segment_master.rotations,
        ranges=Segment_master.ranges,
        mass=Segment_master.mass + Segment_child.mass,
        com=new_com,
        inertia=new_inertia,
        mesh=all_mesh,
    )
    return merge_seg


class Model:
    def __init__(self, name):
        self.model = biorbd.Model()
        self.name = name
        self.filename = name + ".bioMod"
        self.segments = []

    def add_segment(self, segment: Segment):
        self.segments.append(segment)
        self.model.AddSegment(
            segment.name,
            segment.parent,
            segment.translations,
            segment.rotations,
            segment.rangesQ,
            segment.rangesQdot,
            segment.rangesQddot,
            segment.char,
            segment.roto_trans,
        )
        idx_seg = self.model.nbSegment() - 1
        for m in segment.mesh:
            self.model.mesh(idx_seg).addPoint(m)

    def write_model(self):
        biorbd.Writer().writeModel(self.model, self.filename)

    def add_contact(self, contact: Contact):
        parent_int = self.model.GetBodyRbdlId(contact.parent)
        self.model.AddConstraint(parent_int, contact.position, contact.axis, contact.name, contact.acc)
        print("ello")

    def add_marker(self, marker: Marker):
        parent_int = self.model.GetBodyRbdlId(marker.parent)
        self.model.addMarker(
            marker.position,
            marker.name,
            marker.parent,
            marker.technical,
            marker.anatomical,
            marker.axis_to_remove,
            parent_int,
        )
        print("hello")


my_model = Model("humanoid_10_dof")
torso = Segment(
    name="Torso",
    parent="ROOT",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.zeros(3),
    translations="yz",
    rotations="x",
    ranges=np.array([[-1, 1], [0, 1.5], [-np.pi / 2, np.pi / 6]]),
    mass=52.8093248044798,
    com=np.array([0, 0, 0.24]),
    inertia=np.array([[5, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, 0.7]]),
)
head = Segment(
    name="Head",
    parent="Torso",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.array([0, 0, 0.56]),
    translations="",
    rotations="x",
    ranges=np.array([-np.pi / 6, np.pi / 6]),
    mass=5.41,
    com=np.array([0, 0, 0.12]),
    inertia=np.array([[0.12, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, 0.24]]),
)
r_thigh = Segment(
    name="RThigh",
    parent="Torso",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.zeros(3),
    translations="",
    rotations="x",
    ranges=np.array([-np.pi / 2, np.pi / 2]),
    mass=10.6751189590988,
    com=np.array([0, 0, -0.18]),
    inertia=np.array([[0.7, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, -0.41]]),
)
l_thigh = Segment(
    name="LThigh",
    parent="Torso",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.zeros(3),
    translations="",
    rotations="x",
    ranges=np.array([-np.pi / 2, np.pi / 2]),
    mass=10.6751189590988,
    com=np.array([0, 0, -0.18]),
    inertia=np.array([[0.7, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, -0.41]]),
)

r_arm = Segment(
    name="RArm",
    parent="Torso",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.array([0, 0, 0.56]),
    translations="",
    rotations="x",
    ranges=np.array([-np.pi / 2, np.pi / 2]),
    mass=1.5,
    com=np.array([0, 0, -0.12]),
    inertia=np.array([[0.2, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, -0.3]]),
)

l_arm = Segment(
    name="LArm",
    parent="Torso",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.array([0, 0, 0.56]),
    translations="",
    rotations="x",
    ranges=np.array([-np.pi / 2, np.pi / 2]),
    mass=1.5,
    com=np.array([0, 0, -0.12]),
    inertia=np.array([[0.2, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, -0.3]]),
)

r_forearm = Segment(
    name="RForearm",
    parent="RArm",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.array([0, 0, -0.3]),
    translations="",
    rotations="x",
    ranges=np.array([0, np.pi]),
    mass=1.5,
    com=np.array([0, 0, -0.10]),
    inertia=np.array([[0.2, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, -0.2]]),
)

l_forearm = Segment(
    name="LForearm",
    parent="LArm",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.array([0, 0, -0.3]),
    translations="",
    rotations="x",
    ranges=np.array([0, np.pi]),
    mass=1.5,
    com=np.array([0, 0, -0.10]),
    inertia=np.array([[0.2, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, -0.2]]),
)

r_shank = Segment(
    name="RShank",
    parent="RThigh",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.array([0, 0, -0.41]),
    translations="",
    rotations="x",
    ranges=np.array([-np.pi, 0]),
    mass=4.25505875898884,
    com=np.array([0, 0, -0.2]),
    inertia=np.array([[0.7, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, -0.46]]),
)

l_shank = Segment(
    name="LShank",
    parent="LThigh",
    rtinmatrix=False,
    rt=np.zeros(3),
    xyz=np.array([0, 0, -0.41]),
    translations="",
    rotations="x",
    ranges=np.array([-np.pi, 0]),
    mass=4.25505875898884,
    com=np.array([0, 0, -0.2]),
    inertia=np.array([[0.7, 0, 0], [0, 0, 0], [0, 0, 0]]),
    mesh=np.array([[0, 0, 0], [0, 0, -0.46]]),
)

r_foot_contact = Contact(
    name="RFoot",
    parent="RShank",
    position=np.array([0, 0, -0.46]),
    axis="yz",
)
r_foot_marker = Marker(
    name="RFoot",
    parent="RShank",
    position=np.array([0, 0, -0.46]),
)
l_foot_marker = Marker(
    name="LFoot",
    parent="LShank",
    position=np.array([0, 0, -0.46]),
)

my_model.add_segment(torso)
my_model.add_segment(head)
my_model.add_segment(r_thigh)
my_model.add_segment(l_thigh)
my_model.add_segment(r_arm)
my_model.add_segment(l_arm)
my_model.add_segment(r_forearm)
my_model.add_segment(l_forearm)
my_model.add_segment(r_shank)
my_model.add_segment(l_shank)

my_model.add_contact(r_foot_contact)
my_model.add_marker(r_foot_marker)
my_model.add_marker(l_foot_marker)

merge_segment(r_thigh, r_shank)

print("hello")

# Write the model and read back
# biorbd.Writer().writeModel(lukas, "coucou.bioMod")
# lukas_coucou = biorbd.Model("coucou.bioMod")
