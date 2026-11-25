import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.tsplib_io import read_manifest, read_tsplib

GRID = 4


def features(c):
    n = len(c)
    lo, hi = c.min(axis=0), c.max(axis=0)
    w, h = np.maximum(hi - lo, 1e-9)
    try:
        hull = len(ConvexHull(c).vertices) / n
    except Exception:
        hull = 1.0
    ix = np.minimum(((c[:, 0] - lo[0]) / w * GRID).astype(int), GRID - 1)
    iy = np.minimum(((c[:, 1] - lo[1]) / h * GRID).astype(int), GRID - 1)
    p = np.bincount(ix * GRID + iy, minlength=GRID * GRID)
    p = p[p > 0] / n
    entropy = float(-(p * np.log(p)).sum() / np.log(GRID * GRID))
    nn = cKDTree(c).query(c, k=2)[0][:, 1]
    d = squareform(pdist(c)) + 1e-9
    np.fill_diagonal(d, 0)
    t = minimum_spanning_tree(d).toarray()
    deg = ((t > 0) | (t.T > 0)).sum(axis=1)
    return {
        "aspect_ratio": float(max(w, h) / min(w, h)),
        "hull_fraction": float(hull),
        "spatial_entropy": entropy,
        "clark_evans": float(nn.mean() / (0.5 / np.sqrt(n / (w * h)))),
        "nn_cv": float(nn.std() / nn.mean()),
        "mst_max_degree": int(deg.max()),
        "mst_leaf_fraction": float((deg == 1).mean()),
    }


def main():
    rows = []
    for r in read_manifest():
        _, _, coords, _ = read_tsplib(ROOT / r["path"])
        rows.append({"name": r["name"], "group": r["group"], "n": len(coords), **features(coords)})
    out = ROOT / "data" / "processed" / "results" / "geometric_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(len(rows), "instancias ->", out)


if __name__ == "__main__":
    main()