from typing import Union, Callable
from multiprocessing import Pool
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
    all_calls = [[*call, seed_start + i_rand] for i_rand in range(call_number) for call in call_lists]

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
    with Pool(pool_nb) as p:
        p.map(running_function, calls)


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
