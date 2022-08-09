"""
This script runs the miller optimal control problem with a given set of parameters and save the results.
The main function is used in main_comparison.py and main_convergence.py. to run the different Miller optimal control problem.
"""
import os
import numpy as np
import pickle
from time import time

import biorbd
from bioptim import Solver, Shooting, RigidBodyDynamics, Shooting, SolutionIntegrator, BiorbdInterface, CostType
from robot_leg import ArmOCP, Integration


def torque_driven_dynamics(
    model: biorbd.Model, states: np.array, controls: np.array, params: np.array, fext: np.array
) -> np.ndarray:
    q = states[: model.nbQ()]
    qdot = states[model.nbQ() :]
    tau = controls
    if fext is None:
        qddot = model.ForwardDynamics(q, qdot, tau).to_array()
    else:
        fext_vec = biorbd.VecBiorbdVector()
        fext_vec.append(fext)
        qddot = model.ForwardDynamics(q, qdot, tau, biorbd.VecBiorbdSpatialVector(), fext_vec).to_array()
    return np.hstack((qdot, qddot))


def main(args: list = None):
    """
    Main function for the run_miller.py script.
    It runs the optimization and saves the results of a Miller Optimal Control Problem.

    Parameters
    ----------
    args : list
        List of arguments containing the following:
        args[0] : biorbd_model_path
            Path to the biorbd model.
        args[1] : i_rand
            Random seed.
        args[2] : n_shooting
            Number of shooting nodes.
        args[3] : dynamics_type (RigidBodyDynamics)
            Type of dynamics to use such as RigidBodyDynamics.ODE or RigidBodyDynamics.DAE_INVERSE_DYNAMICS, ...
        args[4] : ode_solver
            Type of ode solver to use such as OdeSolver.RK4, OdeSolver.RK2, ...
        args[5] : nstep
            Number of steps for the ode solver.
        args[6] : n_threads
            Number of threads to use.
        args[7] : out_path_raw
            Path to save the raw results.
    """
    if args:
        biorbd_model_path = args[0]
        ode_solver = args[1]
        n_shooting = args[2]
        n_threads = args[3]
        dynamics_type = args[4]
        out_path_raw = args[5]
        i_rand = args[6]
    else:
        biorbd_model_path = args[0]
        ode_solver = args[1]
        n_shooting = args[2]
        n_threads = args[3]
        dynamics_type = args[4]
        out_path_raw = args[5]
        i_rand = args[6]

    # --- Solve the program --- #
    my_ocp = ArmOCP(
        biorbd_model_path=biorbd_model_path,
        rigidbody_dynamics=dynamics_type,
        n_shooting=n_shooting,
        ode_solver=ode_solver,
        n_threads=n_threads,
        # seed=i_rand,
    )
    str_ode_solver = ode_solver.__str__().replace("\n", "_").replace(" ", "_")
    str_dynamics_type = dynamics_type.__str__().replace("RigidBodyDynamics.", "").replace("\n", "_").replace(" ", "_")
    filename = f"sol_irand{i_rand}_{n_shooting}_{str_ode_solver}_{ode_solver.defects_type.value}_{str_dynamics_type}"
    outpath = f"{out_path_raw}/" + filename

    # check if this file already exists if yes return
    print(outpath)
    if os.path.isfile(outpath + ".pckl"):
        print("dejà vu")
        return

    # --- Solve the program --- #
    show_online_optim = False
    print("Show online optimization", show_online_optim)
    solver = Solver.IPOPT(show_online_optim=show_online_optim, show_options=dict(show_bounds=True))

    solver.set_maximum_iterations(10000)
    solver.set_print_level(5)
    # solver.set_convergence_tolerance(1e-10)
    solver.set_linear_solver("ma57")

    my_ocp.ocp.add_plot_penalty(CostType.ALL)
    print(f"##########################################################")
    print(
        f"Solving ... \n"
        f"filename: {filename} \n"
        f"i_rand={i_rand},\n"
        f"dynamics_type={dynamics_type},\n"
        f"ode_solver={str_ode_solver},\n"
        f"n_shooting={n_shooting},\n"
        f"n_threads={n_threads}\n"
    )
    print(f"##########################################################")

    # --- time to solve --- #
    tic = time()
    sol = my_ocp.ocp.solve(solver)
    toc = time() - tic

    # states = sol.states["all"]
    # controls = sol.controls["all"]
    # parameters = sol.parameters["all"]

    sol.print_cost()

    print(f"#################################################### done ")
    print(
        f"Solved in {toc} sec \n"
        f"i_rand={i_rand},\n"
        f"dynamics_type={dynamics_type},\n"
        f"ode_solver={str_ode_solver},\n"
        f"n_shooting={n_shooting},\n"
        f"n_threads={n_threads}\n"
    )
    print(f"##################################################### done ")
    # sol.graphs(show_bounds=True)
    # --- Save the results --- #

    # integrer la dynamique direct

    integration = Integration(
        ocp=my_ocp.ocp,
        solution=sol,
        state_keys=["q", "qdot"],
        control_keys=["tau"],
        # fext_keys=["fext"] if dynamics_type == RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK else None,
        function=torque_driven_dynamics,
        mode="constant_control",
    )

    out = integration.integrate(
        shooting_type=Shooting.SINGLE_CONTINUOUS,
        keep_intermediate_points=False,
        merge_phases=False,
        continuous=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )

    f = open(f"{outpath}.pckl", "wb")
    data = {
        "model_path": biorbd_model_path,
        "phase_time": my_ocp.phase_time,
        "irand": i_rand,
        "computation_time": toc,
        "cost": sol.cost,
        "detailed_cost": sol.detailed_cost,
        "iterations": sol.iterations,
        "status": sol.status,
        "states": sol.states,
        "controls": sol.controls,
        "parameters": sol.parameters,
        "time": out.time_vector,
        "dynamics_type": dynamics_type,
        "ode_solver": ode_solver,
        "ode_solver_str": ode_solver.__str__().replace("\n", "_").replace(" ", "_"),
        "defects_type": ode_solver.defects_type,
        "q": sol.states_no_intermediate["q"],
        "qdot": sol.states_no_intermediate["qdot"],
        "q_integrated": out.states["q"],
        "qdot_integrated": out.states["qdot"],
        # "qddot_integrated": out.states["qdot"],
        "n_shooting": n_shooting,
        "n_theads": n_threads,
        # "q_integrated_linear": out_2.states["q"],
        # "qdot_integrated_linear": out_2.states["qdot"],
        # "time_linear": out_2.time_vector,
    }
    print("hello")
    pickle.dump(data, f)
    f.close()

    # leg_ocp.ocp.save(sol, f"{outpath}.bo")


if __name__ == "__main__":
    main()
