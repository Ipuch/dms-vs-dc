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

from robot_leg import ArmOCP, LegOCP, MillerOCP
from run_ocp import RunOCP


def main(model: Models = None):
    # model = Models.ARM
    # model = Models.LEG
    model = Models.ACROBAT

    if model == Models.LEG:
        #n_shooting = [(20, 20)]
        n_shooting = [20]
        run_ocp = RunOCP(ocp_class=LegOCP, show_optim=False, iteration=0, print_level=5, ignore_already_run=False)
        running_function = run_ocp.main
    elif model == Models.ARM:
        n_shooting = [50]
        run_ocp = RunOCP(ocp_class=ArmOCP, show_optim=False, iteration=0, print_level=5, ignore_already_run=False)
        running_function = run_ocp.main
    elif model == Models.ACROBAT:
        n_shooting = [(125, 25)]
        run_ocp = RunOCP(ocp_class=MillerOCP, show_optim=False, iteration=0, print_level=5, ignore_already_run=False)
        running_function = run_ocp.main
    else:
        raise ValueError("Unknown model")

    # --- Generate the output path --- #
    # out_path = Path("/home/puchaud/Projets_Python/dms-vs-dc-results/ARM_01-08-22_2")
    #
    Date = date.today().strftime("%d-%m-%y")
    out_path = Path(Path(__file__).parent.__str__() + f"/../../dms-vs-dc-results/{model.name}_{Date}_2")
    try:
        os.mkdir(out_path)
    except:
        print(f"{out_path}" + Date + " is already created ")

   # --- Generate the parameters --- #
    n_thread = 8
    param = dict(
        model_str=[
            model.value,
        ],
        ode_solver=[
            OdeSolver.RK4(n_integration_steps=5),
            # OdeSolver.RK4(n_integration_steps=5),
            # # OdeSolver.RK4(n_integration_steps=10),
            # OdeSolver.RK8(n_integration_steps=1),
            # # OdeSolver.RK8(n_integration_steps=10),
            # # OdeSolver.CVODES(),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
            OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
        ],
        n_shooting=n_shooting,
        n_thread=[n_thread],
        dynamic_type=[
            RigidBodyDynamics.ODE,
            # RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
        ],
        out_path=[out_path.absolute().__str__()],
    )
    calls = int(1)

    my_calls = generate_calls(
        call_number=calls,
        parameters=param,
    )

    cpu_number = cpu_count()
    my_pool_number = int(cpu_number / n_thread)

    running_function(my_calls[0])
    # running_function(my_calls[1])
    # run_pool(
    #     running_function=running_function,
    #     calls=my_calls,
    #     pool_nb=4,
    # )


if __name__ == "__main__":
    main()
    main(model=Models.ARM)
