from bioptim import OdeSolver, CostType, RigidBodyDynamics, Solver, DefectType, Shooting, SolutionIntegrator
from robot_leg import UpperLimbOCP, Models
import numpy as np
import matplotlib.pyplot as plt


def main():

    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    ode_solver = OdeSolver.COLLOCATION()

    n_threads = 8
    model_path = Models.UPPER_LIMB_XYZ_VARIABLES.value

    # --- Solve the program --- #
    myocp = UpperLimbOCP(
        biorbd_model_path=model_path,
        n_shooting=100,
        ode_solver=ode_solver,
        rigidbody_dynamics=RigidBodyDynamics.ODE,
        n_threads=n_threads,
        seed=42,
        # seed=None,
    )

    myocp.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(1000)
    solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = myocp.ocp.solve(solv)

    # --- Show results --- #
    sol.graphs(show_bounds=True)
    print(sol.status)
    sol.print_cost()

    out = sol.integrate(
        shooting_type=Shooting.SINGLE,
        keep_intermediate_points=False,
        merge_phases=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )

    plt.figure()
    plt.plot(sol.time, sol.states["q"].T, label="ocp", marker=".")
    plt.plot(out.time, out.states["q"].T, label="integrated", marker="+")
    plt.legend()
    plt.show()

    sol.animate(n_frames=0, show_floor=False, show_gravity=False)


if __name__ == "__main__":
    main()
