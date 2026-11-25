import gc
import json
import sys
import time
import traceback

import numpy as np

from src.experiments import adapters
from src.experiments.tsplib_io import ROOT

COLS = ["instance", "group", "n", "algorithm", "warm_start", "gc", "reps", "init_time_s", "init_cost",
        "total_time_s", "time_std_s", "nodes_expanded", "nodes_pruned", "max_depth", "peak_mem_mb",
        "best_cost", "final_cost", "status", "finished_at"]


def blank_row(spec):
    row = {c: None for c in COLS}
    row.update(instance=spec["instance"], group=spec["group"], n=spec["n"], algorithm=spec["algorithm"],
               warm_start=spec["warm_start"], gc="on" if spec["gc"] else "off", reps=spec["reps"])
    return row


def peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 ** 2 if sys.platform == "darwin" else r / 1024
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024 ** 2


def check_tour(inst, tour):
    t = list(tour)
    if len(t) == inst.n + 1 and t[0] == t[-1]:
        t = t[:-1]
    if sorted(t) != list(range(inst.n)):
        return None
    d = inst.distance_matrix
    return float(sum(d[t[i], t[(i + 1) % inst.n]] for i in range(inst.n)))


def run_heuristic(spec, inst, fns, row):
    solve, times, sol = fns[0], [], None
    for _ in range(spec["reps"]):
        t0 = time.perf_counter()
        sol = solve(inst)
        times.append(time.perf_counter() - t0)
    cost = check_tour(inst, sol.tour)
    row["total_time_s"] = float(np.median(times))
    row["time_std_s"] = float(np.std(times))
    if cost is None or abs(cost - sol.cost) > 1e-6 * max(1.0, cost):
        row["status"] = "invalid"
        return
    row["best_cost"] = row["final_cost"] = cost
    row["status"] = "completed"


def run_bnb(spec, inst, fns, row):
    init_fn, bnb_fn = fns
    t0 = time.perf_counter()
    init = init_fn(inst)
    init_time = time.perf_counter() - t0
    init_cost = check_tour(inst, init.tour)
    row["init_time_s"], row["init_cost"] = init_time, init_cost
    if init_cost is None:
        row["status"] = "invalid"
        return
    remaining = max(spec["time_limit"] - init_time, 1.0)
    t1 = time.perf_counter()
    res = bnb_fn(inst, init, remaining)
    row["total_time_s"] = init_time + (time.perf_counter() - t1)
    for k in ("nodes_expanded", "nodes_pruned", "max_depth"):
        row[k] = getattr(res, k, None)
    raw = str(getattr(res, "status", "")).lower()
    status = "completed" if raw.startswith(("complete", "optimal")) else "timeout" if "time" in raw else (raw or "error")
    best = init_cost
    sol = getattr(res, "solution", None)
    if sol is not None:
        best = check_tour(inst, sol.tour)
        if best is None:
            row["status"] = "invalid"
            return
    row["best_cost"] = best
    row["final_cost"] = best if status == "completed" else None
    row["status"] = status

def warm_up(fns, spec):
    from src.experiments.tsplib_io import Instance, distance_matrix
    c = np.random.default_rng(0).uniform(0, 100, (8, 2))
    tiny = Instance("warmup", 8, c, distance_matrix(c, spec["dist"]))
    first = fns[0](tiny)
    if len(fns) > 1:
        fns[1](tiny, first, 5)

def main():
    spec = json.loads(sys.argv[1])
    row = blank_row(spec)
    try:
        inst = adapters.load_instance(ROOT / spec["path"], spec["dist"])
        if spec["algorithm"] == "bnb":
            fns = (adapters.heuristic(spec["warm_start"]), adapters.bnb())
        else:
            fns = (adapters.heuristic(spec["algorithm"]),)
        warm_up(fns, spec)
        gc.collect()
        if not spec["gc"]:
            gc.disable()
        base = peak_rss_mb()
        (run_bnb if spec["algorithm"] == "bnb" else run_heuristic)(spec, inst, fns, row)
        row["peak_mem_mb"] = round(peak_rss_mb() - base, 3)
    except Exception:
        row["status"] = "error"
        sys.stderr.write(traceback.format_exc())
    default = lambda o: o.item() if hasattr(o, "item") else str(o)
    sys.stdout.write("RESULT:" + json.dumps(row, default=default) + "\n")

if __name__ == "__main__":
    main()