import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.tsplib_io import load_instance, read_manifest

OUT = ROOT / "data" / "processed" / "results" / "reference_optima.csv"
KNOWN = {"eil51": 426, "berlin52": 7542, "st70": 675, "wi29": 27603, "dj38": 6656}
COLS = ["name", "n", "optimum", "status", "seconds", "cuts", "known", "matches_known"]


def solve_optimum(d, time_limit):
    n = len(d)
    iu, ju = np.triu_indices(n, k=1)
    m = len(iu)
    cost = d[iu, ju].astype(float)
    rows = np.concatenate([iu, ju])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    incidence = sp.csr_matrix((np.ones(2 * m), (rows, cols)), shape=(n, m))
    cons = [LinearConstraint(incidence, 2, 2)]
    start, cuts = time.perf_counter(), 0
    while True:
        left = time_limit - (time.perf_counter() - start)
        if left <= 0:
            return None, "timeout", cuts
        res = milp(cost, constraints=cons, integrality=np.ones(m), bounds=Bounds(0, 1),
                   options={"time_limit": left, "mip_rel_gap": 0.0})
        if res.status != 0 or res.x is None:
            return None, "timeout" if res.status == 1 else "failed", cuts
        x = res.x > 0.5
        g = sp.coo_matrix((np.ones(int(x.sum())), (iu[x], ju[x])), shape=(n, n))
        k, labels = connected_components(g, directed=False)
        if k == 1:
            return float(cost[x].sum()), "optimal", cuts
        for c in range(k):
            members = labels == c
            inside = (members[iu] & members[ju]).astype(float)
            cons.append(LinearConstraint(sp.csr_matrix(inside.reshape(1, -1)), -np.inf, members.sum() - 1))
            cuts += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main,sens")
    ap.add_argument("--dist", default="nint")
    ap.add_argument("--time-limit", type=float, default=900)
    args = ap.parse_args()
    groups = set(args.group.split(","))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        with open(OUT, newline="") as f:
            done = {r["name"] for r in csv.DictReader(f)}
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for r in read_manifest():
            if r["group"] not in groups or r["name"] in done:
                continue
            inst = load_instance(ROOT / r["path"], args.dist)
            t0 = time.perf_counter()
            opt, status, cuts = solve_optimum(inst.distance_matrix, args.time_limit)
            known = KNOWN.get(r["name"]) if args.dist == "nint" else None
            match = (opt is not None and known is not None and abs(opt - known) < 1e-6)
            w.writerow({"name": r["name"], "n": inst.n, "optimum": opt, "status": status,
                        "seconds": round(time.perf_counter() - t0, 2), "cuts": cuts,
                        "known": known, "matches_known": match if known is not None else ""})
            f.flush()
            flag = "" if known is None else ("  ok" if match else "  DIFERENTE DO OTIMO CONHECIDO")
            print(f"{r['name']:<24} n={inst.n:<4} opt={opt} {status} {time.perf_counter() - t0:.1f}s{flag}")


if __name__ == "__main__":
    main()