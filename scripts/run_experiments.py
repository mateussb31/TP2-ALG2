import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.tsplib_io import read_manifest
from src.experiments.worker import COLS, blank_row

NAMES = ["trivial", "nn", "hull", "tat", "chr"]
RESULTS = ROOT / "data" / "processed" / "results"
DONE = {"completed", "timeout", "killed"}
LOCK = threading.Lock()


def collect_env(args):
    import numpy
    import scipy
    info = {"args": vars(args), "python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "started": datetime.now().isoformat(timespec="seconds")}
    try:
        import psutil
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except ImportError:
        pass
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                                     stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return info


def spec_key(s):
    return (s["instance"], s["algorithm"], s["warm_start"], "on" if s["gc"] else "off")


def row_key(r):
    return (r["instance"], r["algorithm"], r["warm_start"], r["gc"])


def make_specs(args):
    groups = set(args.group.split(","))
    specs = []
    for r in read_manifest():
        n = int(r["n"])
        if r["group"] not in groups or not args.min_n <= n <= args.max_n:
            continue
        base = {"instance": r["name"], "path": r["path"], "group": r["group"], "n": n, "dist": args.dist,
                "gc": not args.gc_off, "time_limit": args.time_limit}
        if args.suite == "approx":
            specs += [{**base, "algorithm": a, "warm_start": "none", "reps": args.reps} for a in NAMES]
        else:
            specs += [{**base, "algorithm": "bnb", "warm_start": w, "reps": 1} for w in NAMES]
    specs.sort(key=lambda s: s["n"], reverse=(args.order == "large"))
    return specs


def done_keys(path):
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {row_key(r) for r in csv.DictReader(f) if r["status"] in DONE}


def append(path, row):
    with LOCK:
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow({c: "NA" if row.get(c) is None else row[c] for c in COLS})


def save_log(spec, text):
    logs = RESULTS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{spec['instance']}_{spec['algorithm']}_{spec['warm_start']}.err").write_text(text[-4000:])


def run_job(spec, hard):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    cmd = [sys.executable, "-m", "src.experiments.worker", json.dumps(spec)]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=hard, cwd=ROOT, env=env)
    except subprocess.TimeoutExpired:
        row = blank_row(spec)
        row["status"], row["total_time_s"] = "killed", time.perf_counter() - t0
        return row
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("RESULT:"):
            row = json.loads(line[7:])
            if row["status"] in ("error", "invalid"):
                save_log(spec, p.stderr)
            return row
    row = blank_row(spec)
    row["status"] = "error"
    save_log(spec, p.stderr)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["approx", "warmstart"], required=True)
    ap.add_argument("--group", default="main", help="main, sens, scale (separados por virgula)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=1800)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dist", default="nint")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=10 ** 9)
    ap.add_argument("--order", choices=["large", "small"], default="large")
    ap.add_argument("--gc-off", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = "algorithm_results.csv" if args.suite == "approx" else "warm_start_results.csv"
    out = Path(args.out) if args.out else RESULTS / name
    (RESULTS / f"environment_{args.suite}.json").write_text(json.dumps(collect_env(args), indent=2, default=str))

    done = done_keys(out)
    specs = [s for s in make_specs(args) if spec_key(s) not in done]
    print(f"{len(specs)} execucoes pendentes ({len(done)} ja feitas) -> {out}", flush=True)
    hard = lambda s: args.time_limit + 120 if s["algorithm"] == "bnb" else 900
    count = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_job, s, hard(s)): s for s in specs}
        for fut in as_completed(futs):
            row = fut.result()
            row["finished_at"] = datetime.now().isoformat(timespec="seconds")
            append(out, row)
            count += 1
            t = row.get("total_time_s")
            print(f"[{count}/{len(specs)}] {row['instance']} {row['algorithm']} {row['warm_start']} "
                  f"-> {row['status']} {t if t is None else round(t, 2)}s", flush=True)


if __name__ == "__main__":
    main()