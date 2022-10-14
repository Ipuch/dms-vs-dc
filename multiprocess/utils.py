from typing import Union, Callable
from multiprocessing import Pool, Process
import os
from pathlib import Path
import pickle
from itertools import product


def generate_calls(
    call_number: int,
    parameters: dict,
    seed_start: int = 0,
) -> list:
    """
    Generate the list of calls to be used in multiprocessing

    Parameters
    ----------
    call_number : Union[int, list]
        The int of list of irand to be run
    parameters : dict
        The parameters to be used in the calls containing the following keys:
        - model_str : str
            The model to be used
        ode_solver : list[OdeSolver]
            The list of ode solvers to be used
        - dynamics_types : RigidBodyDynamics
            The dynamics to be used
        - nstep : int
            The number of steps to be used
        - n_threads : int
            The number of threads to be used
        - out_path_raw : str
            The path to the output folder
        - extra_obj : bool
            Whether to use the extra objective or not
        - Date : str
            The date to be used in the output folder
        - n_shooting : tuple
            The number of shooting nodes to be used
    seed_start : int, optional
        The start seed to be used, by default 0

    Returns
    -------
    all_calls: list
        The list of calls to be run
    """
    call_lists = set_product_list(parameters)
    all_calls = [
        [*call, seed_start + i_rand] for i_rand in range(call_number) for call in call_lists
    ]

    return all_calls


def run_pool(running_function: Callable, calls: list, pool_nb: int):
    """
    Run the pool of processes

    Parameters
    ----------
    running_function : Callable
        The function to be run in the pool
    calls : list
        The list of calls to be run
    pool_nb : int
        The number of processes to be used in parallel
    """
    # run_humanoid(calls[0])
    with Pool(pool_nb) as p:  # should be 4
        p.map(running_function, calls)


def run_the_missing_ones(
    out_path_raw: str,
    Date,
    n_shooting: tuple,
    dynamics_types: list,
    ode_solver: list,
    nstep: int,
    n_threads: int,
    model_str: str,
    extra_obj: bool,
    pool_nb: int,
):
    """
    This function is used to run the process that were not run during the previous pool of processes

    Parameters
    ----------
    out_path_raw : str
        The path to store the raw results
    Date : str
        The date of the run
    n_shooting : tuple
        The number of shooting points for each phase
    dynamics_types : list
        The list of dynamics types to be run such as MillerDynamics.EXPLICIT, MillerDynamics.IMPLICIT, MillerDynamics.ROOT_EXPLICIT
    ode_solver : list
        The list of OdeSolver to be run such as OdeSolver.RK4, OdeSolver.RK2
    nstep : int
        The number of intermediate steps between two shooting points
    n_threads : int
        The number of threads to be used
    model_str : str
        The path to the bioMod model
    extra_obj : bool
        Whether to run with the extra objective or not (minimizing extra controls for implicit formulations)
    """
    # Run the one that did not run
    files = os.listdir(out_path_raw)
    files.sort()

    new_calls = {dynamics_types[0].value: [], dynamics_types[1].value: []}
    for i, file in enumerate(files):
        if file.endswith(".pckl"):
            p = Path(f"{out_path_raw}/{file}")
            file_path = open(p, "rb")
            data = pickle.load(file_path)
            if (
                data["dynamics_type"].value == dynamics_types[0].value
                or data["dynamics_type"].value == dynamics_types[1].value
            ):
                new_calls[data["dynamics_type"].value].append(data["irand"])

    list_100 = [i for i in range(0, 100)]

    dif_list = list(set(list_100) - set(new_calls[dynamics_types[0].value]))
    if dif_list:
        calls = generate_calls(
            dif_list,
            Date,
            n_shooting,
            [dynamics_types[1]],
            [ode_solver[1]],
            nstep,
            n_threads,
            out_path_raw,
            model_str,
            extra_obj,
        )
        run_pool(calls, pool_nb)

    dif_list = list(set(list_100) - set(new_calls[dynamics_types[1].value]))

    if dif_list:
        calls = generate_calls(
            dif_list,
            Date,
            n_shooting,
            [dynamics_types[1]],
            [ode_solver[1]],
            nstep,
            n_threads,
            out_path_raw,
            model_str,
            extra_obj,
        )
        run_pool(calls, pool_nb)


def set_product_list(parameters_compared: dict = None):
    """
    Set the list of parameters to be used in the product function

    Parameters
    ----------
    parameters_compared : dict, optional
        The parameters to be compared, by default None

    Returns
    -------
    list_parameters: list
        The list of parameters to be used in the product function

    """

    vals = parameters_compared.values()
    keys = parameters_compared.keys()
    list_combinations = [instance for instance in product(*vals)]

    return list_combinations
