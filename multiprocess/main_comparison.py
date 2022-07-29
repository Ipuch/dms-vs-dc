"""

"""
import os
from utils import generate_calls, run_pool, run_the_missing_ones
from robot_leg import LegOCP, ArmOCP, Models
from multiprocessing import Pool, cpu_count
from datetime import date
from bioptim import OdeSolver, RigidBodyDynamics
from pathlib import Path
from bioptim import DefectType

from run_leg_ocp import main as main_leg_ocp
from run_arm_ocp import main as main_arm_ocp


def main():
    Date = date.today()
    Date = Date.strftime("%d-%m-%y")
    model = Models.ARM.value

    out_path_raw = Path(Path(__file__).parent.__str__() + f"/../../robot-leg-results/arm_{Date}")
    try:
        os.mkdir(out_path_raw)
    except:
        print("../robot-leg-results/raw_" + Date + " is already created ")

    cpu_number = cpu_count()
    n_thread = 2
    param = dict(
        model_str=[
            model,
        ],
        ode_solver=[
            # OdeSolver.RK4(n_integration_steps=1),
            OdeSolver.RK4(n_integration_steps=5),
            # OdeSolver.RK4(n_integration_steps=10),
            # OdeSolver.RK8(n_integration_steps=1),
            OdeSolver.RK8(n_integration_steps=5),
            # OdeSolver.RK8(n_integration_steps=10),
            # OdeSolver.CVODES(),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=2),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=3),
            OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=5),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=6),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=7),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=8),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=9),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=2, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=3, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=4, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=5, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=6, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=7, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=8, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.EXPLICIT, polynomial_degree=9, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=2),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=3),
            OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=5),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=6),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=7),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=8),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=9),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=2, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=3, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=4, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=5, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=6, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=7, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=8, method="radau"),
            # OdeSolver.IRK(defects_type=DefectType.IMPLICIT, polynomial_degree=9, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=2),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=3),
            OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=4),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=5),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=6),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=7),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=8),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=9),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=2, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=3, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=4, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=5, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=6, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=7, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=8, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.IMPLICIT, polynomial_degree=9, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=2),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=3),
            OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=4),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=5),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=6),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=7),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=8),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=9),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=2, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=3, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=4, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=5, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=6, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=7, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=8, method="radau"),
            # OdeSolver.COLLOCATION(defects_type=DefectType.EXPLICIT, polynomial_degree=9, method="radau"),
        ],
        n_shooting=[50],
        n_thread=[n_thread],
        dynamic_type=[
            RigidBodyDynamics.ODE,
            # RigidBodyDynamics.DAE_INVERSE_DYNAMICS_JERK
        ],
        out_path=[out_path_raw.absolute().__str__()],
    )
    calls = int(10)

    my_calls = generate_calls(
        call_number=calls,
        parameters=param,
    )

    my_pool_number = int(cpu_number / n_thread)

    # main_arm_ocp(my_calls[0])
    run_pool(
        running_function=main_arm_ocp,
        calls=my_calls,
        pool_nb=4,
    )

    # run_the_missing_ones(
    #     out_path_raw, Date, n_shooting, ode_solver, nsteps, n_thread, model_str, my_pool_number
    # )


if __name__ == "__main__":
    main()
