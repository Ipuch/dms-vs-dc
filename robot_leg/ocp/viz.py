import numpy as np
import biorbd_casadi as biorbd
from bioptim import PlotType, BiorbdInterface


def plot_com(x, nlp):
    com_func = biorbd.to_casadi_func(
        "CoMPlot", nlp.model.CoM, nlp.states["q"].mx, expand=False
    )
    com_dot_func = biorbd.to_casadi_func(
        "Compute_CoM",
        nlp.model.CoMdot,
        nlp.states["q"].mx,
        nlp.states["qdot"].mx,
        expand=False,
    )
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.concatenate(
        (np.array(com_func(q)[1:, :]), np.array(com_dot_func(q, qdot)[1:, :]))
    )


def plot_qddot(x, u, nlp):
    return np.array(nlp.dynamics_func(x, u, []))[nlp.states["qdot"].index, :]


def plot_contact_acceleration(x, u, nlp):
    qddot = nlp.states["qddot"] if "qddot" in nlp.states else nlp.controls["qddot"]
    acc_x = biorbd.to_casadi_func(
        "acc_0",
        nlp.model.rigidContactAcceleration(
            nlp.states["q"].mx, nlp.states["qdot"].mx, qddot.mx, 0
        ).to_mx(),
        nlp.states["q"].mx,
        nlp.states["qdot"].mx,
        qddot.mx,
        expand=False,
    )

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])
    if "qddot" in nlp.states:
        qddot = nlp.states["qddot"].mapping.to_second.map(x[qddot.index, :])
    else:
        qddot = nlp.controls["qddot"].mapping.to_second.map(u[qddot.index, :])

    return np.array(acc_x(q, qdot, qddot)[list(nlp.model.rigidContactAxisIdx(0)), :])


def add_custom_plots(ocp):
    for i, nlp in enumerate(ocp.nlp):
        ocp.add_plot(
            "CoM",
            lambda t, x, u, p: plot_com(x, nlp),
            phase=i,
            legend=["CoMy", "Comz", "CoM_doty", "CoM_dotz"],
        )
    for i, nlp in enumerate(ocp.nlp):
        ocp.add_plot(
            "qddot",
            lambda t, x, u, p: plot_qddot(x, u, nlp),
            phase=i,
            legend=["qddot"],
            plot_type=PlotType.INTEGRATED,
        )

        # if "qddot" in ocp.nlp[0].states:
        # ocp.add_plot(
        #     "Contact_Acceleration", lambda t, x, u, p: plot_contact_acceleration(x, u, nlp), phase=i,
        #     legend=["Contact_acceleration_x",
        #             "Contact_acceleration_y"], plot_type=PlotType.INTEGRATED)
        # ocp.add_plot_penalty(CostType.ALL)
