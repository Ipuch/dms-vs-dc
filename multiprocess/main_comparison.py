"""

"""
import os


from multiprocessing import cpu_count
from datetime import date
from pathlib import Path
import pandas as pd

from bioptim import OdeSolver, RigidBodyDynamics, DefectType

from utils import generate_calls, run_pool
from transcriptions import ArmOCP, LegOCP, MillerOcpOnePhase, Models, UpperLimbOCP, HumanoidOCP
from run_ocp import RunOCP


def main(
    model: Models = None,
    iterations=10000,
    print_level=5,
    ignore_already_run=False,
    show_optim=False,
    seed_start=0,
    calls=1,
):

    if model == Models.LEG:
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
        # n_shooting = [75, 100, 150]
        run_ocp = RunOCP(
            ocp_class=MillerOcpOnePhase,
            show_optim=show_optim,
            iteration=iterations,
            print_level=print_level,
            ignore_already_run=ignore_already_run,
        )
        running_function = run_ocp.main
    elif model == Models.UPPER_LIMB_XYZ_VARIABLES:
        n_shooting = [100]
        run_ocp = RunOCP(
            ocp_class=UpperLimbOCP,
            show_optim=show_optim,
            iteration=iterations,
            print_level=print_level,
            ignore_already_run=ignore_already_run,
        )
        running_function = run_ocp.main

    elif model == Models.HUMANOID_10DOF:
        n_shooting = [30]
        run_ocp = RunOCP(
            ocp_class=HumanoidOCP,
            show_optim=show_optim,
            iteration=iterations,
            print_level=print_level,
            ignore_already_run=ignore_already_run,
        )
        running_function = run_ocp.main

    else:
        raise ValueError("Unknown model")

    ode_list = [
        OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
        OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
        OdeSolver.RK4(n_integration_steps=5),
        OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
    ]
    if model != Models.ACROBAT:
        if model == Models.HUMANOID_10DOF:  # no implicit
            ode_list = ode_list[1:]
        else:
            ode_list.append(OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=4))

    # --- Generate the output path --- #
    Date = date.today().strftime("%d-%m-%y")
    out_path = Path(
        Path(__file__).parent.__str__()
        + f"/../../dms-vs-dc-results/{model.name}_2023"
    )
    try:
        os.mkdir(out_path)
    except:
        print(f"{out_path}" + Date + " is already created ")

    # --- Generate the parameters --- #
    n_thread = 8
    #  n_thread = 32

    param = dict(
        model_str=[
            model.value,
        ],
        ode_solver=ode_list,
        n_shooting=n_shooting,
        n_thread=[n_thread],
        dynamic_type=[
            RigidBodyDynamics.ODE,
        ],
        out_path=[out_path.absolute().__str__()],
    )
    calls = int(calls)

    my_calls = generate_calls(
        call_number=calls,
        parameters=param,
        seed_start=seed_start,
    )

    cpu_number = cpu_count()
    my_pool_number = int(cpu_number / n_thread)

    # running_function(my_calls[0])
    # running_function(my_calls[1])
    columns = list(param.keys())
    columns.append("random")
    df = pd.DataFrame(my_calls, columns=columns)

    for ode_solver in ode_list:
        sub_df = df[df["ode_solver"] == ode_solver]
        my_calls = sub_df.to_numpy().tolist()
        run_pool(
            running_function=running_function,
            calls=my_calls,
            pool_nb=my_pool_number,
        )


if __name__ == "__main__":
    iteration = 3000
    main(
        model=Models.LEG,
        iterations=iteration,
        print_level=5,
        ignore_already_run=False,
        show_optim=False,
        seed_start=0,
        calls=100,
    )
    main(
        model=Models.ARM,
        iterations=iteration,
        print_level=5,
        ignore_already_run=True,
        show_optim=False,
        seed_start=0,
        calls=100,
    )
    main(
        model=Models.ACROBAT,
        iterations=iteration,
        print_level=5,
        ignore_already_run=True,
        show_optim=False,
        seed_start=0,
        calls=100,
    )
    main(
        model=Models.HUMANOID_10DOF,
        iterations=iteration,
        print_level=5,
        ignore_already_run=True,
        show_optim=False,
        seed_start=0,
        calls=100,
    )
    main(
        model=Models.UPPER_LIMB_XYZ_VARIABLES,
        iterations=500,
        print_level=5,
        ignore_already_run=True,
        show_optim=False,
        seed_start=0,
        calls=100,
    )
    main(
        model=Models.UPPER_LIMB_XYZ_VARIABLES,
        iterations=200,
        print_level=5,
        ignore_already_run=True,
        show_optim=False,
        seed_start=0,
        calls=100,
    )


