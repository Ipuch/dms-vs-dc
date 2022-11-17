"""
This file is to display the human model into bioviz
"""
import os
import bioviz
from robot_leg import Models

# model_name = Models.LEG
# model_name = Models.ARM
# model_name = Models.ACROBAT
# model_name = Models.UPPER_LIMB_XYZ_VARIABLES
model_name = Models.HUMANOID_10DOF

export_model = False
background_color = (1, 1, 1) if export_model else (0.5, 0.5, 0.5)
show_gravity_vector = False if export_model else True
show_floor = False if export_model else True
show_local_ref_frame = False if export_model else True
show_global_ref_frame = False if export_model else True
show_markers = False if export_model else True
show_mass_center = False if export_model else True
show_global_center_of_mass = False if export_model else True
show_segments_center_of_mass = False if export_model else True


def print_all_camera_parameters(biorbd_viz: bioviz.Viz):
    print("Camera roll: ", biorbd_viz.get_camera_roll())
    print("Camera zoom: ", biorbd_viz.get_camera_zoom())
    print("Camera position: ", biorbd_viz.get_camera_position())
    print("Camera focus point: ", biorbd_viz.get_camera_focus_point())


if model_name == Models.LEG:

    biorbd_viz = bioviz.Viz(model_name.value,
        show_gravity_vector=False,
        show_floor=False,
        show_local_ref_frame=show_local_ref_frame,
        show_global_ref_frame=show_global_ref_frame,
        show_markers=show_markers,
        show_mass_center=show_mass_center,
        show_global_center_of_mass=False,
        show_segments_center_of_mass=True,
        mesh_opacity=1,
        background_color=(1, 1, 1),
                             )

    biorbd_viz.resize(1000, 1000)
    biorbd_viz.set_camera_roll(-82.89751054930615)
    biorbd_viz.set_camera_zoom(2.7649491449197656)
    biorbd_viz.set_camera_position(1.266097531449429, -0.6523601622496974, 0.24962580067391163)
    biorbd_viz.set_camera_focus_point(0.07447263939980919, 0.025078204682856153, -0.013568198245759833)


if model_name == Models.ARM:
    biorbd_viz = bioviz.Viz(
        model_name.value,
        show_gravity_vector=False,
        show_floor=False,
        show_local_ref_frame=show_local_ref_frame,
        show_global_ref_frame=show_global_ref_frame,
        show_markers=show_markers,
        show_mass_center=show_mass_center,
        show_global_center_of_mass=False,
        show_segments_center_of_mass=True,
        mesh_opacity=1,
        background_color=(1, 1, 1),
    )
    biorbd_viz.resize(1000, 1000)

    biorbd_viz.set_q([-0.15, 0.24, -0.41, 0.21, 0, 0])
    biorbd_viz.set_camera_roll(-84.5816885957667)
    biorbd_viz.set_camera_zoom(2.112003880097381)
    biorbd_viz.set_camera_position(1.9725681105744026, -1.3204979216430117, 0.35790018139336177)
    biorbd_viz.set_camera_focus_point(-0.3283876664932833, 0.5733643134562766, 0.018451815011995998)

if model_name == Models.ACROBAT:
    biorbd_viz = bioviz.Viz(
        model_name.value,
        show_gravity_vector=False,
        show_floor=False,
        show_local_ref_frame=False,
        show_global_ref_frame=False,
        show_markers=False,
        show_mass_center=False,
        show_global_center_of_mass=False,
        show_segments_center_of_mass=False,
        mesh_opacity=1,
        background_color=(1, 1, 1),
    )
    biorbd_viz.set_camera_position(-8.782458942185185, 0.486269131372712, 4.362010279585766)
    biorbd_viz.set_camera_roll(90)
    biorbd_viz.set_camera_zoom(0.308185240948253)
    biorbd_viz.set_camera_focus_point(1.624007185850899, 0.009961251074366406, 1.940316420941989)
    biorbd_viz.resize(600, 900)


if model_name == Models.UPPER_LIMB_XYZ_VARIABLES:
    biorbd_viz = bioviz.Viz(
        model_name.value,
        show_gravity_vector=False,
        show_floor=False,
        show_local_ref_frame=False,
        show_global_ref_frame=False,
        show_markers=False,
        show_mass_center=False,
        show_global_center_of_mass=False,
        show_segments_center_of_mass=False,
        mesh_opacity=1,
        background_color=(1, 1, 1),
    )
    biorbd_viz.resize(1000, 1000)

    # biorbd_viz.set_q([-0.15, 0.24, -0.41, 0.21, 0, 0])
    biorbd_viz.set_camera_roll(-100.90843467296737)
    biorbd_viz.set_camera_zoom(1.9919059008044755)
    biorbd_viz.set_camera_position(0.8330547810707182, 2.4792370867179256, 0.1727481994453778)
    biorbd_viz.set_camera_focus_point(-0.2584435804313228, 0.8474543937884143, 0.2124670559215174)
    # get current path file
    # file_name, extension = os.path.splitext(model_name)
    # biorbd_viz.snapshot(f"{file_name}/{Models.UPPER_LIMB_XYZ_VARIABLES.name}.png")

if model_name == Models.HUMANOID_10DOF:
    biorbd_viz = bioviz.Viz(
        model_name.value[0],
        show_gravity_vector=False,
        show_floor=False,
        show_local_ref_frame=False,
        show_global_ref_frame=False,
        show_markers=False,
        show_mass_center=False,
        show_global_center_of_mass=False,
        show_segments_center_of_mass=False,
        mesh_opacity=1,
        background_color=(1, 1, 1),
    )
    biorbd_viz.resize(1000, 1000)

    biorbd_viz.set_q([-0.20120228,  0.84597746, -0.12389997, -0.15, 0.41, -0.37, -0.86, 0.36, 0.39, 0.66, -0.58, 0])
    biorbd_viz.set_camera_roll(-91.44517177211645)
    biorbd_viz.set_camera_zoom(0.7961539827851234)
    biorbd_viz.set_camera_position(4.639962934524132, 0.4405891958030146, 0.577705598983718)
    biorbd_viz.set_camera_focus_point(-0.2828701273331326, -0.04065388066757992, 0.9759133347931428)


biorbd_viz.exec()
print_all_camera_parameters(biorbd_viz)

print("Done")
