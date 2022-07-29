"""

"""
import os
from utils import generate_calls, run_pool, run_the_missing_ones
from robot_leg import Models
from multiprocessing import Pool, cpu_count
from datetime import date
from bioptim import OdeSolver, RigidBodyDynamics
from pathlib import Path
from bioptim import DefectType

from run_leg_ocp import main as main_leg_ocp
from run_arm_ocp import main as main_arm_ocp
from run_miller import main as main_miller_ocp


def main():
    # model = Models.ARM
    # model = Models.LEG
    model = Models.ACROBAT

    if model == Models.LEG:
        running_function = main_leg_ocp
    elif model == Models.ARM:
        running_function = main_arm_ocp
    elif model == Models.ACROBAT:
        running_function = main_miller_ocp
    else:
        raise ValueError("Unknown model")

    # --- Generate the output path --- #
    Date = date.today().strftime("%d-%m-%y")
    out_path = Path(Path(__file__).parent.__str__() + f"/../../dms-vs-dc-results/{model.name}_{Date}")
    try:
        os.mkdir(out_path)
    except:
        print(f"{out_path}" + Date + " is already created ")

   # --- Generate the parameters --- #
    n_thread = 2
    param = dict(
        model_str=[
            model.value,
        ],
        ode_solver=[
            OdeSolver.RK4(n_integration_steps=5),
            # OdeSolver.RK4(n_integration_steps=10),
            OdeSolver.RK8(n_integration_steps=5),
            # OdeSolver.RK8(n_integration_steps=10),
            # OdeSolver.CVODES(),
            OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
            OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
            OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
            OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
        ],
        # n_shooting=[20], # leg
        # n_shooting=[50], # arm
        n_shooting=[(125, 25)], # acrobat
        n_thread=[n_thread],
        dynamic_type=[
            RigidBodyDynamics.ODE,
            # RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
        ],
        out_path=[out_path.absolute().__str__()],
    )
    calls = int(10)

    my_calls = generate_calls(
        call_number=calls,
        parameters=param,
    )

    cpu_number = cpu_count()
    my_pool_number = int(cpu_number / n_thread)

    # running_function(my_calls[0])
    run_pool(
        running_function=running_function,
        calls=my_calls,
        pool_nb=4,
    )

    # run_the_missing_ones(
    #     out_path_raw, Date, n_shooting, ode_solver, nsteps, n_thread, model_str, my_pool_number
    # )


if __name__ == "__main__":
    main()
