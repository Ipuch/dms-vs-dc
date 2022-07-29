# nested loops
def generate_strings(letters, transitions, k):
    def helper(s):
        if len(s) == k:
            yield s
        elif len(s) < k:
            for letter in transitions[s[-1]]:
                yield from helper(s + letter)

    for letter in letters:
        yield from helper(letter)


# Example: note that you don't have to use a list of characters, since a string is also a sequence of characters.

letters = "abcd"
transitions = {"a": "bc", "b": "a", "c": "d", "d": "bcd"}
for s in generate_strings(letters, transitions, 4):
    print(s)

from bioptim import OdeSolver
from itertools import product

nstep = 1
ode_solver_list = [
    OdeSolver.RK4(n_integration_steps=nstep),
    OdeSolver.RK4(n_integration_steps=nstep * 2),
    OdeSolver.RK8(n_integration_steps=nstep),
    OdeSolver.CVODES(),
    OdeSolver.IRK(polynomial_degree=3, method="legendre"),
    OdeSolver.IRK(polynomial_degree=9, method="legendre"),
    OdeSolver.COLLOCATION(polynomial_degree=3, method="legendre"),
    OdeSolver.COLLOCATION(polynomial_degree=9, method="legendre"),
]
tol_list = [1, 0.01, 0.001, 1e-5, 1e-8]
shooting_list = [5, 10, 20, 40]
# nested loops

param = {"ode_solver": ode_solver_list, "tolerance": tol_list, "nodes": shooting_list}


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def generate_simulation(param):
    def helper(s):
        if len(s) == k:
            yield s
        elif len(s) < k:
            for letter in transitions[s[-1]]:
                yield from helper(s + letter)

    for letter in letters:
        yield from helper(letter)


# Example: note that you don't have to use a list of characters, since a string is also a sequence of characters.

letters = "abcd"
transitions = {"a": "bc", "b": "a", "c": "d", "d": "bcd"}
for s in generate_strings(letters, transitions, 4):
    print(s)
