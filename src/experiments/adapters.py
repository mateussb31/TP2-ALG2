import os
from dataclasses import dataclass

from src.experiments.tsplib_io import load_instance as _load

FAKE = os.environ.get("TP2_FAKE") == "1"
FAST = os.environ.get("TP2_FAST") == "1"


@dataclass
class Solution:
    tour: list
    cost: float


def load_instance(path, dist):
    # se a pessoa 1 tiver TSPInstance.from_file, troque aqui
    return _load(path, dist)


def trivial(inst):
    tour = list(range(inst.n))
    d = inst.distance_matrix
    return Solution(tour, float(sum(d[tour[i], tour[(i + 1) % inst.n]] for i in range(inst.n))))


def heuristic(name):
    if name == "trivial":
        return trivial
    if FAKE:
        from src.experiments import fake_algos
        return fake_algos.HEURISTICS[name]
    if name == "nn":
        from src.algorithms.heuristics.nearest_neighbor import solve
    elif name == "hull":
        from src.algorithms.heuristics.convex_hull_insertion import solve
    elif name == "tat":
        from src.algorithms.heuristics.twice_around_the_tree import solve
    elif name == "chr":
        from src.algorithms.heuristics.christofides import solve
    else:
        raise ValueError(name)
    return solve


def bnb():
    if FAKE:
        from src.experiments import fake_algos
        return fake_algos.bnb
    if FAST:
        from src.algorithms.exact import branch_and_bound_fast as branch_and_bound
    else:
        from src.algorithms.exact import branch_and_bound

    def run(inst, initial_solution, time_limit):
        return branch_and_bound.solve(inst, initial_solution=initial_solution, time_limit=time_limit)

    return run