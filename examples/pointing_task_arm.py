from bioptim import OdeSolver, CostType, RigidBodyDynamics, Solver, DefectType, Shooting, SolutionIntegrator
from robot_leg import ArmOCP
import numpy as np
import matplotlib.pyplot as plt


def main():
    n_shooting = 50
    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    # ode_solver = OdeSolver.IRK(polynomial_degree=4, defects_type=DefectType.IMPLICIT)
    # ode_solver = OdeSolver.COLLOCATION(polynomial_degree=4, defects_type=DefectType.IMPLICIT)
    ode_solver = OdeSolver.RK4()
    # ode_solver = OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT)
    time = 0.25
    n_threads = 8
    model_path = "../robot_leg/models/robot_arm.bioMod"

    # --- Solve the program --- #
    arm = ArmOCP(
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
    print("number of states: ", arm.ocp.v.n_all_x)
    print("number of controls: ", arm.ocp.v.n_all_u)

    arm.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = arm.ocp.solve(solv)

    # --- Show results --- #
    print(sol.status)
    sol.print_cost()

    out = sol.integrate(
        shooting_type=Shooting.SINGLE,
        keep_intermediate_points=False,
        merge_phases=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )

    # sol.animate(n_frames=0, show_floor=False, show_gravity=False)
    # sol.graphs(show_bounds=True)

    plt.figure()
    plt.plot(sol.time, sol.states["q"].T, label="ocp", marker=".")
    plt.plot(out.time, out.states["q"].T, label="integrated", marker="+")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
