from bioptim import OdeSolver, CostType, RigidBodyDynamics, Solver, DefectType
from robot_leg import LegOCP

import numpy as np
from robot_leg import Integration


def main():
    n_shooting = (20, 20)
    # n_shooting = 20
    ode_solver = OdeSolver.RK4(n_integration_steps=5)
    # ode_solver = OdeSolver.RK4()
    # ode_solver = OdeSolver.COLLOCATION()
    # time = 0.25
    time = 0.25, 0.25
    n_threads = 8
    # for human in Humanoid2D:
    model_path = "../robot_leg/models/hexapod_leg.bioMod"

    # --- Solve the program --- #
    leg = LegOCP(
        biorbd_model_path=model_path,
        phase_time=time,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        rigidbody_dynamics=RigidBodyDynamics.ODE,
        n_threads=n_threads,
        seed=20,
        start_point=np.array([0.22, 0.02, 0.03]),
        end_point=np.array([0.22, 0.021, -0.05]),
    )

    leg.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(10000)
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
        mode="constant_control"
    )

    from bioptim import Shooting, SolutionIntegrator

    out = integration.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        keep_intermediate_points=False,
        merge_phases=True,
        continuous=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,

    )

    #
    sol.animate(n_frames=0, show_floor=False, show_gravity=False)
    # sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
