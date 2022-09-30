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

from robot_leg import ArmOCP, LegOCP, MillerOcpOnePhase
from run_ocp import RunOCP


def main(model: Models = None, iterations=10000, print_level=5, ignore_already_run=False, show_optim=False):

    if model == Models.LEG:
        # n_shooting = [(20, 20)]
        n_shooting = [20]
        run_ocp = RunOCP(
            ocp_class=LegOCP,
            show_optim=show_optim,
            iteration=iterations,
            print_level=print_level,
            ignore_already_run=ignore_already_run,
        )
        running_function = run_ocp.main
    elif model == Models.ARM:
        n_shooting = [50]
        run_ocp = RunOCP(
            ocp_class=ArmOCP,
            show_optim=show_optim,
            iteration=iterations,
            print_level=print_level,
            ignore_already_run=ignore_already_run,
        )
        running_function = run_ocp.main
    elif model == Models.ACROBAT:
        n_shooting = [125]
        run_ocp = RunOCP(
            ocp_class=MillerOcpOnePhase,
            show_optim=show_optim,
            iteration=iterations,
            print_level=print_level,
            ignore_already_run=ignore_already_run,
        )
        running_function = run_ocp.main
    else:
        raise ValueError("Unknown model")

    # --- Generate the output path --- #
    Date = date.today().strftime("%d-%m-%y")
    out_path = Path(
        Path(__file__).parent.__str__()
        + f"/../../dms-vs-dc-results/{model.name}_{Date}_2"
    )
    try:
        os.mkdir(out_path)
    except:
        print(f"{out_path}" + Date + " is already created ")

    # --- Generate the parameters --- #
    n_thread = 8
    # n_thread = 4
    #  n_thread = 32
    param = dict(
        model_str=[
            model.value,
        ],
        ode_solver=[
            OdeSolver.RK4(n_integration_steps=5),
            OdeSolver.RK8(n_integration_steps=2),
            # OdeSolver.CVODES(),
            OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
            OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
            OdeSolver.COLLOCATION(
                defects_type=DefectType.IMPLICIT, polynomial_degree=4
            ),
            OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
        ],
        n_shooting=n_shooting,
        n_thread=[n_thread],
        dynamic_type=[
            RigidBodyDynamics.ODE,
        ],
        out_path=[out_path.absolute().__str__()],
    )
    calls = int(2)

    my_calls = generate_calls(
        call_number=calls,
        parameters=param,
    )

    cpu_number = cpu_count()
    my_pool_number = int(cpu_number / n_thread)

    # running_function(my_calls[0])
    # running_function(my_calls[1])
    run_pool(
         running_function=running_function,
         calls=my_calls,
         pool_nb=my_pool_number,
    )


if __name__ == "__main__":
    # main(model=Models.LEG, iterations=0, print_level=5, ignore_already_run=False, show_optim=True)
    # main(model=Models.ARM, iterations=0, print_level=5, ignore_already_run=False, show_optim=False)
    main(model=Models.ACROBAT, iterations=2500, print_level=5, ignore_already_run=False, show_optim=False)

