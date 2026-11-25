import random
import time
from dataclasses import dataclass


@dataclass
class Sol:
    tour: list
    cost: float


@dataclass
class Res:
    solution: Sol
    cost: float
    nodes_expanded: int
    nodes_pruned: int
    max_depth: int
    execution_time: float
    status: str


def cost_of(inst, tour):
    d = inst.distance_matrix
    return float(sum(d[tour[i], tour[(i + 1) % inst.n]] for i in range(inst.n)))


def make(seed, delay):
    def solve(inst):
        time.sleep(delay)
        tour = list(range(inst.n))
        random.Random(seed + inst.n).shuffle(tour)
        return Sol(tour, cost_of(inst, tour))
    return solve


HEURISTICS = {"nn": make(1, 0.01), "hull": make(2, 0.01), "tat": make(3, 0.02), "chr": make(4, 0.05)}


def bnb(inst, initial_solution, time_limit):
    work, nodes = inst.n * 0.05, inst.n ** 3
    if work > time_limit:
        time.sleep(max(time_limit, 0))
        return Res(initial_solution, initial_solution.cost, nodes // 3, nodes // 5, inst.n // 2, time_limit, "timeout")
    time.sleep(work)
    return Res(initial_solution, initial_solution.cost, nodes, nodes * 6 // 10, inst.n, work, "completed")