import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_analysis import FEATS, HEUR, RES, bh, family

gaps = pd.read_csv(ROOT / "report" / "tables" / "gaps.csv")
feat = pd.read_csv(RES / "geometric_features.csv").rename(columns={"name": "instance"})
d = gaps.merge(feat[["instance"] + FEATS], on="instance")
d["family"] = d.instance.map(family)

print("medias das features por familia:")
print(d[d.algorithm == "chr"].groupby("family")[FEATS].mean().round(2).to_string())

sub = d[d.family != "circle"]
rows = []
for a in HEUR:
    g = sub[sub.algorithm == a]
    res = [stats.spearmanr(g[f], g.gap_pct) for f in FEATS]
    q = bh([r.pvalue for r in res])
    for f, r, qq in zip(FEATS, res, q):
        rows.append({"algorithm": a, "feature": f, "spearman": r.statistic, "p_bh": qq,
                     "instances": len(g)})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "report" / "tables" / "sensitivity_spearman_sem_circle.csv", index=False)
print("\nspearman sem circle, p_bh < 0.05:")
print(out[out.p_bh < 0.05].round(4).to_string(index=False))
print("\ninstancias por algoritmo:", len(sub[sub.algorithm == "chr"]))