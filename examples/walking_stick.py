import numpy as np
import matplotlib.pyplot as plt

import biorbd_casadi as biorbd
from bioptim import OdeSolver, CostType, Solver, PlotType, SolutionIntegrator, Shooting

from transcriptions import HumanoidOCP, Models


def main():
    n_shooting = 30
    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    # ode_solver = OdeSolver.RK4()
    # ode_solver = OdeSolver.IRK()
    ode_solver = OdeSolver.COLLOCATION()

    n_threads = 8
    # for human in Humanoid2D:
    human = Models.HUMANOID_10DOF

    # --- Solve the program --- #
    humanoid = HumanoidOCP(
        biorbd_model_path=human.value,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        n_threads=n_threads,
        nb_phases=1,
        seed=42,
    )

    add_custom_plots(humanoid.ocp)
    humanoid.ocp.add_plot_penalty(CostType.ALL)
    print("number of states: ", humanoid.ocp.v.n_all_x)
    print("number of controls: ", humanoid.ocp.v.n_all_u)
    humanoid.ocp.print(to_console=True, to_graph=False)

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = humanoid.ocp.solve(solv)

    # --- Show results --- #
    sol.graphs(show_bounds=True)
    sol.print_cost()

    out = sol.integrate(
        shooting_type=Shooting.SINGLE,
        keep_intermediate_points=False,
        merge_phases=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )

    plt.figure()
    # ocp in blue
    plt.plot(sol.time, sol.states["q"].T, label="ocp", marker=".", color="blue")
    # integration in red
    plt.plot(out.time, out.states["q"].T, label="integrated", marker="+", color="red")
    plt.legend()
    plt.show()

    sol.animate(n_frames=0)


def plot_com(x, nlp):
    com_func = biorbd.to_casadi_func("CoMPlot", nlp.model.model.CoM, nlp.states["q"].mx, expand=False)
    com_dot_func = biorbd.to_casadi_func(
        "Compute_CoM",
        nlp.model.model.CoMdot,
        nlp.states["q"].mx,
        nlp.states["qdot"].mx,
        expand=False,
    )
    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    qdot = nlp.states["qdot"].mapping.to_second.map(x[nlp.states["qdot"].index, :])

    return np.concatenate((np.array(com_func(q)[1:, :]), np.array(com_dot_func(q, qdot)[1:, :])))


def plot_qddot(x, u, nlp):
    return np.array(nlp.dynamics_func(x, u, []))[nlp.states["qdot"].index, :]


def plot_contact_acceleration(x, u, nlp):
    qddot = nlp.states["qddot"] if "qddot" in nlp.states else nlp.controls["qddot"]
    acc_x = biorbd.to_casadi_func(
        "acc_0",
        nlp.model.model.rigidContactAcceleration(nlp.states["q"].mx, nlp.states["qdot"].mx, qddot.mx, 0).to_mx(),
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
            "com",
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


if __name__ == "__main__":
    main()
