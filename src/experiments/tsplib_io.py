import csv
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Instance:
    name: str
    n: int
    coordinates: np.ndarray
    distance_matrix: np.ndarray


def read_tsplib(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    name, kind, dim, coords, reading = path.name.split(".")[0], None, None, [], False
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line == "EOF":
                break
            if line.startswith("NODE_COORD_SECTION"):
                reading = True
            elif reading and line:
                p = line.split()
                coords.append((float(p[1]), float(p[2])))
            elif ":" in line:
                key, value = (s.strip() for s in line.split(":", 1))
                if key == "NAME":
                    name = value
                elif key == "EDGE_WEIGHT_TYPE":
                    kind = value
                elif key == "DIMENSION":
                    dim = int(value)
    return name, kind, np.array(coords), dim


def distance_matrix(coords, mode="nint"):
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    if mode == "nint":
        return np.floor(d + 0.5).astype(np.int64)
    return d


def load_instance(path, dist="nint"):
    name, kind, coords, _ = read_tsplib(path)
    if kind != "EUC_2D":
        raise ValueError(f"{name}: edge_weight_type {kind} != EUC_2D")
    return Instance(name, len(coords), coords, distance_matrix(coords, dist))


def write_tsplib(path, name, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"NAME : {name}\nCOMMENT : {comment}\nTYPE : TSP\nDIMENSION : {len(coords)}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, 1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        f.write("EOF\n")


def read_manifest():
    with open(ROOT / "data" / "processed" / "instances" / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def generate(kind, n, seed):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        p = rng.uniform(0, 1000, (n, 2))
    elif kind == "clustered":
        k = int(rng.integers(2, 7))
        sigma = float(rng.choice([15, 30, 60]))
        centers = rng.uniform(100, 900, (k, 2))
        p = centers[rng.integers(0, k, n)] + rng.normal(0, sigma, (n, 2))
    elif kind == "elongated":
        ratio = float(rng.choice([3, 6, 12, 25]))
        p = rng.uniform(0, 1, (n, 2)) * [1000, 1000 / ratio]
    elif kind == "circle":
        a = rng.uniform(0, 2 * np.pi, n)
        p = 500 + 450 * np.column_stack([np.cos(a), np.sin(a)])
    else:
        raise ValueError(kind)
    return np.round(np.clip(p, 0, 1000), 2)