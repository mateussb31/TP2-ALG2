import argparse
import cProfile
import csv
import gc
import pstats
import re
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import adapters
from src.experiments.tsplib_io import read_manifest

OUT = ROOT / "data" / "processed" / "results"
WATCH = re.compile(r"mst|spanning|prim|kruskal|match|blossom|euler|shortcut|hull|bound", re.I)


def solver(name, args):
    if name == "bnb":
        init, run = adapters.heuristic(args.warm), adapters.bnb()
        return lambda inst: run(inst, init(inst), args.time_limit)
    return adapters.heuristic(name)


def profile_one(fn, inst):
    pr = cProfile.Profile()
    pr.enable()
    fn(inst)
    pr.disable()
    st = pstats.Stats(pr).stats
    tot = sum(v[2] for v in st.values()) or 1e-12
    root = max(v[3] for v in st.values()) or 1e-12
    rows = []
    for (file, line, name), (cc, nc, tt, ct, _) in st.items():
        if tt / tot >= 0.01 or (ct / root >= 0.02 and WATCH.search(name + file)):
            rows.append({"function": name, "file": Path(file).name, "line": line, "ncalls": nc,
                         "tottime_s": tt, "cumtime_s": ct, "share_tot": tt / tot, "share_cum": ct / root})
    return rows


def memory_one(fn, inst):
    gc.collect()
    tracemalloc.start()
    fn(inst)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def gc_compare(fn, inst, reps):
    out = {"on": [], "off": []}
    fn(inst)
    for i in range(reps):
        order = ("on", "off") if i % 2 == 0 else ("off", "on")
        for mode in order:
            gc.collect()
            if mode == "off":
                gc.disable()
            t0 = time.perf_counter()
            fn(inst)
            out[mode].append(time.perf_counter() - t0)
            gc.enable()
    return out


def append_rows(path, rows):
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=["tat", "chr"])
    ap.add_argument("--group", default="scale")
    ap.add_argument("--warm", default="nn")
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--max-n", type=int, default=10 ** 9)
    ap.add_argument("--gc-reps", type=int, default=5)
    args = ap.parse_args()
    groups = set(args.group.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    for r in read_manifest():
        if r["group"] not in groups or int(r["n"]) > args.max_n:
            continue
        inst = adapters.load_instance(ROOT / r["path"], "nint")
        for a in args.algos:
            fn = solver(a, args)
            base = {"instance": r["name"], "n": inst.n, "algorithm": a}
            append_rows(OUT / "profiling_results.csv", [{**base, **x} for x in profile_one(fn, inst)])
            peak = memory_one(fn, inst)
            matrix = inst.distance_matrix.nbytes
            append_rows(OUT / "memory_results.csv", [{**base, "py_peak_kb": peak / 1024,
                                                      "matrix_kb": matrix / 1024, "peak_over_matrix": peak / matrix}])
            g = gc_compare(fn, inst, args.gc_reps)
            append_rows(OUT / "gc_results.csv", [{**base, "gc_on_median_s": float(np.median(g["on"])),
                                                  "gc_off_median_s": float(np.median(g["off"]))}])
            print(f"{r['name']} {a}: pico python {peak / 1024:.0f} kb ({peak / matrix:.1f}x a matriz)", flush=True)


if __name__ == "__main__":
    main()