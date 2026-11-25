import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "processed" / "results"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
FEATS = ["n", "aspect_ratio", "hull_fraction", "spatial_entropy", "clark_evans",
         "nn_cv", "mst_max_degree", "mst_leaf_fraction"]
HEUR = ["nn", "hull", "tat", "chr"]
ORDER = ["trivial", "nn", "hull", "tat", "chr"]
FIXED = {
    "n": lambda n: n,
    "n log n": lambda n: n * np.log(n),
    "n^2": lambda n: n ** 2,
    "n^2 log n": lambda n: n ** 2 * np.log(n),
    "n^3": lambda n: n ** 3,
    "2^n": lambda n: 2.0 ** n,
}


def read(name):
    p = RES / name
    return pd.read_csv(p, na_values=["NA"]) if p.exists() else None


def fit_models(n, t):
    n, t = np.asarray(n, float), np.asarray(t, float)
    ok = np.isfinite(t) & (t > 0)
    n, t = n[ok], t[ok]
    o = np.argsort(n)
    n, t = n[o], t[o]
    lt, N = np.log(t), len(t)
    tss, tss_lin = np.sum((lt - lt.mean()) ** 2), np.sum((t - t.mean()) ** 2)
    rows, curves = [], {}

    def add(name, k, pred):
        rss = np.sum((lt - np.log(pred)) ** 2) + 1e-12
        aicc = N * np.log(rss / N) + 2 * k + 2 * k * (k + 1) / (N - k - 1) if N - k - 1 > 0 else np.nan
        rows.append({"model": name, "params": k, "r2_log": 1 - rss / tss,
                     "r2_lin": 1 - np.sum((t - pred) ** 2) / tss_lin,
                     "rmse": float(np.sqrt(np.mean((t - pred) ** 2))), "aicc": aicc})
        curves[name] = pred

    for name, g in FIXED.items():
        add(name, 1, np.exp(np.mean(lt - np.log(g(n)))) * g(n))
    lr = stats.linregress(np.log(n), lt)
    add(f"a*n^b (b={lr.slope:.2f})", 2, np.exp(lr.intercept) * n ** lr.slope)
    table = pd.DataFrame(rows).sort_values("aicc").reset_index(drop=True)
    return table, n, t, curves, lr


def plot_fit(n, t, table, curves, title, path, ylabel):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(n, t, color="black", zorder=3, label="medido")
    for name in table.model[:3]:
        ax.plot(n, curves[name], label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def complexity(d, ycol, group_col, tag, ylabel, min_points=5):
    tables, exps = [], []
    for a, g in d.groupby(group_col):
        if len(g) < min_points:
            continue
        table, n, t, curves, lr = fit_models(g["n"], g[ycol])
        ci = stats.t.ppf(0.975, len(n) - 2) * lr.stderr
        exps.append({"series": a, "exponent": lr.slope, "ci95": ci, "r2_loglog": lr.rvalue ** 2, "points": len(n)})
        table.insert(0, "series", a)
        tables.append(table)
        plot_fit(n, t, table, curves, f"{tag}: {a}", FIG / f"fit_{tag}_{a}.png", ylabel)
    if tables:
        pd.concat(tables).to_csv(TAB / f"complexity_{tag}.csv", index=False)
        pd.DataFrame(exps).to_csv(TAB / f"exponents_{tag}.csv", index=False)
        print(f"\nexpoentes empiricos ({tag}):\n", pd.DataFrame(exps).round(3).to_string(index=False))


def heuristic_gaps(algo, ref):
    d = algo[algo.algorithm.isin(ORDER) & (algo.status == "completed")]
    d = d.groupby(["instance", "algorithm"], as_index=False).agg(cost=("final_cost", "first"))
    r = ref[ref.status == "optimal"].rename(columns={"name": "instance"})[["instance", "optimum"]]
    d = d.merge(r, on="instance")
    d["gap_pct"] = 100 * (d.cost - d.optimum) / d.optimum
    return d


def bh(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    m, order = len(p), np.argsort(p)
    adj, prev = np.empty(m), 1.0
    for rank, i in zip(range(m, 0, -1), order[::-1]):
        prev = min(prev, p[i] * m / rank)
        adj[i] = prev
    return adj

def family(name):
    if name.startswith("syn_"):
        return name.split("_")[1]
    return name.split("_")[0]

def sensitivity(gaps, feat):
    data = gaps.merge(feat.rename(columns={"name": "instance"})[["instance"] + FEATS], on="instance")
    corr_rows, model_rows, cv_rows = [], [], []
    for a, g in data[data.algorithm.isin(HEUR)].groupby("algorithm"):
        if len(g) < 12:
            continue
        y, X = g.gap_pct.to_numpy(), g[FEATS].to_numpy(float)
        rho_p = [stats.spearmanr(g[f], y) for f in FEATS]
        adj = bh([r.pvalue for r in rho_p])
        for f, r, q in zip(FEATS, rho_p, adj):
            corr_rows.append({"algorithm": a, "feature": f, "spearman": r.statistic, "p": r.pvalue, "p_bh": q})
        cv = KFold(5, shuffle=True, random_state=0)
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=0)
        grp = g.instance.map(family).to_numpy()
        for name, m in (("ridge", ridge), ("tree", tree)):
            cv_rows.append({"algorithm": a, "model": name, "r2_cv": r2_score(y, cross_val_predict(m, X, y, cv=cv)),
                            "instances": len(y)})
            cv_rows.append({"algorithm": a, "model": name + "_family_out",
                            "r2_cv": r2_score(y, cross_val_predict(m, X, y, cv=GroupKFold(5), groups=grp)),
                            "instances": len(y)})
            m.fit(X, y)
        for f, w in zip(FEATS, ridge[-1].coef_):
            model_rows.append({"algorithm": a, "model": "ridge_std_coef", "feature": f, "weight": w})
        for f, w in zip(FEATS, tree.feature_importances_):
            model_rows.append({"algorithm": a, "model": "tree_importance", "feature": f, "weight": w})
        (TAB / f"tree_{a}.txt").write_text(export_text(tree, feature_names=FEATS))
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(TAB / "sensitivity_spearman.csv", index=False)
    pd.DataFrame(model_rows).to_csv(TAB / "sensitivity_models.csv", index=False)
    cvt = pd.DataFrame(cv_rows)
    cvt.to_csv(TAB / "sensitivity_cv.csv", index=False)
    print("\nr2 em validacao cruzada (valores baixos ou negativos = features nao explicam o gap):\n",
          cvt.round(3).to_string(index=False))
    if len(corr):
        piv = corr.pivot(index="algorithm", columns="feature", values="spearman")[FEATS]
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(piv.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(FEATS)))
        ax.set_xticklabels(FEATS, rotation=45, ha="right")
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels(piv.index)
        fig.colorbar(im, label="spearman com o gap")
        fig.tight_layout()
        fig.savefig(FIG / "sensitivity_heatmap.png", dpi=200)
        plt.close(fig)


def warmstart(ws):
    ws = ws[ws.group == "main"].copy()
    ws["init_time_s"] = ws.init_time_s.fillna(0)
    ws["completed"] = ws.status == "completed"
    summary = ws.groupby("warm_start").agg(
        runs=("instance", "count"), completed=("completed", "sum"),
        timeouts=("status", lambda s: int((s == "timeout").sum())),
        median_nodes=("nodes_expanded", "median"), median_pruned=("nodes_pruned", "median"),
        median_total_s=("total_time_s", "median"), median_init_s=("init_time_s", "median")).reindex(ORDER)
    summary.to_csv(TAB / "warmstart_summary.csv")
    print("\nresumo warm-start:\n", summary.round(3).to_string())
    piv = {c: ws.pivot(index="instance", columns="warm_start", values=c) for c in ("nodes_expanded", "total_time_s")}
    comp = ws.pivot(index="instance", columns="warm_start", values="completed").fillna(False).astype(bool)
    rows = []
    for w in ORDER[1:]:
        if w not in comp.columns or "trivial" not in comp.columns:
            continue
        both = comp["trivial"] & comp[w]
        if both.sum() < 3:
            continue
        nt, nw = piv["nodes_expanded"].loc[both, "trivial"], piv["nodes_expanded"].loc[both, w]
        tt, tw = piv["total_time_s"].loc[both, "trivial"], piv["total_time_s"].loc[both, w]
        try:
            p = stats.wilcoxon(tt, tw).pvalue
        except ValueError:
            p = np.nan
        rows.append({"strategy": w, "pairs": int(both.sum()),
                     "nodes_ratio_geomean": float(np.exp(np.mean(np.log(np.maximum(nw, 1) / np.maximum(nt, 1))))),
                     "time_speedup_geomean": float(np.exp(np.mean(np.log(tt / tw)))), "wilcoxon_p_time": p})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(TAB / "warmstart_vs_trivial.csv", index=False)
    print("\nwarm-start contra trivial (so instancias que ambos terminaram):\n", cmp_df.round(4).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for w in ORDER:
        g = ws[ws.warm_start == w].sort_values("n")
        axes[0].plot(g.n, g.nodes_expanded, "o-", label=w)
        axes[1].plot(g.n, g.total_time_s, "o-", label=w)
    for ax, lab in zip(axes, ("nos expandidos", "tempo total (s)")):
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel(lab)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "warmstart.png", dpi=200)
    plt.close(fig)
    done = ws[ws.completed]
    complexity(done.assign(series=done.warm_start), "total_time_s", "series", "bnb_time", "tempo (s)")
    complexity(done.assign(series=done.warm_start), "nodes_expanded", "series", "bnb_nodes", "nos expandidos")


def sanity(algo, ws, ref):
    opt = ref[ref.status == "optimal"].set_index("name").optimum
    for label, d in (("heuristicas", algo), ("b&b", ws)):
        d = d.dropna(subset=["best_cost"])
        d = d[d.instance.isin(opt.index)]
        below = d[d.best_cost < d.instance.map(opt) - 1e-6]
        if len(below):
            print(f"ATENCAO: {label} com custo abaixo do otimo (bug de distancia ou de custo):")
            print(below[["instance", "algorithm", "warm_start", "best_cost"]].to_string(index=False))
    c = ws[ws.status == "completed"]
    c = c[c.instance.isin(opt.index)]
    bad = c[(c.final_cost - c.instance.map(opt)).abs() > 1e-6]
    print("b&b 'completed' com custo != otimo:", len(bad), "de", len(c))
    if len(bad):
        print(bad[["instance", "warm_start", "final_cost"]].to_string(index=False))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    algo, ws, ref, feat = (read(x) for x in ("algorithm_results.csv", "warm_start_results.csv",
                                             "reference_optima.csv", "geometric_features.csv"))
    mem = read("memory_results.csv")
    if algo is not None:
        scale = algo[(algo.group == "scale") & (algo.status == "completed") & algo.algorithm.isin(HEUR)]
        complexity(scale, "total_time_s", "algorithm", "heur_time", "tempo (s)")
    if mem is not None:
        complexity(mem[mem.algorithm.isin(HEUR)], "py_peak_kb", "algorithm", "heur_mem", "pico python (kb)")
        mem.groupby("algorithm").peak_over_matrix.describe().to_csv(TAB / "memory_copies.csv")
    if algo is not None and ref is not None:
        gaps = heuristic_gaps(algo, ref)
        gaps.to_csv(TAB / "gaps.csv", index=False)
        print("\ngap medio (%) por algoritmo:\n", gaps.groupby("algorithm").gap_pct.describe().round(2).to_string())
        if feat is not None:
            sensitivity(gaps, feat)
    if ws is not None:
        warmstart(ws)
        if ref is not None and algo is not None:
            sanity(algo, ws, ref)
    gcr = read("gc_results.csv")
    if gcr is not None:
        gcr["slowdown_with_gc_pct"] = 100 * (gcr.gc_on_median_s / gcr.gc_off_median_s - 1)
        gcr.to_csv(TAB / "gc_effect.csv", index=False)


if __name__ == "__main__":
    main()