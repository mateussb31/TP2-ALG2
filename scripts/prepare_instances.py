import argparse
import csv
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.tsplib_io import generate, read_tsplib, write_tsplib

RAW = ROOT / "data" / "raw" / "tsplib"
OUT = ROOT / "data" / "processed" / "instances"
HEIDELBERG = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{}.tsp.gz"
WATERLOO = "https://www.math.uwaterloo.ca/tsp/world/{}.tsp"
SOURCES = {"wi29": WATERLOO, "dj38": WATERLOO, "eil51": HEIDELBERG, "berlin52": HEIDELBERG, "st70": HEIDELBERG}
SUBSETS = [("eil51", 12), ("berlin52", 14), ("st70", 16), ("eil51", 18), ("berlin52", 20),
           ("st70", 22), ("eil51", 25), ("berlin52", 28), ("st70", 32), ("eil51", 36)]
KINDS = ["uniform", "clustered", "elongated", "circle"]
SENS_SIZES = [20, 30, 40, 60, 75]
SCALE_SIZES = [10, 20, 30, 40, 50, 60, 75, 100, 150, 200, 300]


def download(name, template):
    url = template.format(name)
    dest = RAW / Path(url).name
    if not dest.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print("baixando", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    return dest


def add(manifest, name, coords, group, source, comment):
    path = OUT / f"{name}.tsp"
    write_tsplib(path, name, coords, comment)
    manifest.append({"name": name, "path": path.relative_to(ROOT).as_posix(),
                     "group": group, "n": len(coords), "source": source})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", nargs="*", default=[], help="nomes de instancias da tsplib para checar o tipo")
    ap.add_argument("--no-supp", action="store_true", help="nao gera as instancias suplementares")
    args = ap.parse_args()

    if args.inspect:
        for name in args.inspect:
            _, kind, _, dim = read_tsplib(download(name, HEIDELBERG))
            print(f"{name}: dimension={dim} edge_weight_type={kind}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    manifest, full = [], {}
    for name, template in SOURCES.items():
        _, kind, coords, _ = read_tsplib(download(name, template))
        assert kind == "EUC_2D", f"{name} nao e EUC_2D"
        full[name] = coords
        add(manifest, name, coords, "main", "original", "copia da instancia original")

    for i, (base, k) in enumerate(SUBSETS):
        idx = np.sort(np.random.default_rng(i).choice(len(full[base]), size=k, replace=False))
        add(manifest, f"{base}_n{k}", full[base][idx], "main", f"subconjunto de {base}",
            f"{k} pontos de {base}, semente {i}")

    if not args.no_supp:
        for kind in KINDS:
            for n in SENS_SIZES:
                for s in range(3):
                    add(manifest, f"syn_{kind}_n{n}_s{s}", generate(kind, n, 100 * s + n),
                        "sens", f"sintetica {kind}", f"{kind} n={n} semente={100 * s + n}")
        for n in SCALE_SIZES:
            add(manifest, f"scale_n{n}", generate("uniform", n, 7), "scale", "sintetica uniform", f"uniform n={n}")

    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "path", "group", "n", "source"])
        w.writeheader()
        w.writerows(manifest)
    main_rows = [m for m in manifest if m["group"] == "main"]
    small = sum(m["n"] <= 40 for m in main_rows)
    print(f"main: {len(main_rows)} instancias, {small} com n<=40 ({100 * small / len(main_rows):.0f}%)")
    print(f"total no manifesto: {len(manifest)}")


if __name__ == "__main__":
    main()