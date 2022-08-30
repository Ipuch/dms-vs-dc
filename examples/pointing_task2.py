from bioptim import OdeSolver, CostType, RigidBodyDynamics, Solver, DefectType
from robot_leg import ArmOCP

import numpy as np
from robot_leg import Integration


def torque_driven_dynamics(
    model: biorbd.Model,
    states: np.array,
    controls: np.array,
    params: np.array,
    fext: np.array,
) -> np.ndarray:
    q = states[: model.nbQ()]
    qdot = states[model.nbQ() :]
    tau = controls
    if fext is None:
        qddot = model.ForwardDynamics(q, qdot, tau).to_array()
    else:
        fext_vec = biorbd.VecBiorbdVector()
        fext_vec.append(fext)
        qddot = model.ForwardDynamics(
            q, qdot, tau, biorbd.VecBiorbdSpatialVector(), fext_vec
        ).to_array()
    return np.hstack((qdot, qddot))


def main():
    n_shooting = 50
    ode_solver = OdeSolver.RK4(n_integration_steps=5)
    # ode_solver = OdeSolver.IRK(polynomial_degree=4, defects_type=DefectType.IMPLICIT)
    # ode_solver = OdeSolver.COLLOCATION(polynomial_degree=4, defects_type=DefectType.IMPLICIT)
    # ode_solver = OdeSolver.RK4()
    # ode_solver = OdeSolver.COLLOCATION()
    time = 0.25
    n_threads = 8
    model_path = "../robot_leg/models/robot_arm.bioMod"

    # --- Solve the program --- #
    leg = ArmOCP(
        biorbd_model_path=model_path,
        phase_time=time,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        rigidbody_dynamics=RigidBodyDynamics.ODE,
        n_threads=n_threads,
        seed=0,
        # start_point=np.array([0.5, -0.02, 0.1]),
        end_point=np.array([0.63, 0, -0.1]),
    )

    leg.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(0)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = leg.ocp.solve(solv)

    # --- Show results --- #
    print(sol.status)
    sol.print_cost()

    # from humanoid_2d import Integration
    #
    integration = Integration(
        ocp=leg.ocp,
        solution=sol,
        state_keys=["q", "qdot"],
        control_keys=["tau"],
        parameters_keys=None,
        function=None,
    )
    #
    sol.animate(n_frames=0, show_floor=False, show_gravity=False)
    # sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
