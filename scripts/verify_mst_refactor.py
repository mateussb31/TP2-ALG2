#!/usr/bin/env python3
"""Verifica a otimização da MST do limite inferior do B&B (versão _fast).

Compara a versão BASE (src/algorithms/exact/bounds.py e branch_and_bound.py,
que geraram os resultados oficiais) com a versão OTIMIZADA (bounds_fast.py e
branch_and_bound_fast.py) e confirma que o limite calculado é o mesmo; logo
nodes_expanded, nodes_pruned, max_depth e o custo saem idênticos e só o tempo
muda.

Uso (da raiz do repositório, com o venv ativo, sem outras execuções pesadas
rodando ao mesmo tempo, para os tempos serem comparáveis):
    python scripts/verify_mst_refactor.py
    python scripts/verify_mst_refactor.py --profile --out docs/refactor_mst_verificacao.txt
    python scripts/verify_mst_refactor.py --sizes 12 13 14 --seeds 0 1 2

Etapas:
  1. mst_weight base vs otimizado: mesmo peso em subconjuntos aleatórios, e
     mesma decisão `peso >= cutoff` com early-exit.
  2. remaining_bound (base) vs remaining_bound_shared (otimizado), em estados
     de busca simulados, com cutoffs aleatórios e em ordem arbitrária (testa
     o reuso do SharedMST): mesma decisão de poda e mesmo valor quando < cutoff.
  3. B&B completo, base vs otimizado, em instâncias aleatórias pequenas, com
     cold-start e warm-start (NN): contadores e custo idênticos; tempos e
     speedup impressos. Execução com timeout é "inconclusiva" (com timeout o
     otimizado, mais rápido, expande mais nós: isso é esperado).
  4. (--profile) cProfile da base e da otimizada na maior instância.

Sai com código 0 se tudo bater e 1 caso contrário.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print(f"  [FALHOU] {msg}")
    return cond


class Tee:
    """Duplica a saída padrão em um arquivo UTF-8."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def make_instance(TSPInstance, n, seed):
    rng = np.random.default_rng(seed)
    return TSPInstance(f"v{n}_s{seed}", rng.integers(0, 1000, (n, 2)).astype(float))


# ---------------------------------------------------------------------------
def step1_mst(old_b, new_b, D, trials=400):
    print("\n1. mst_weight: base vs otimizado")
    lo, ln = old_b.LowerBound(D), new_b.LowerBound(D)
    rng = random.Random(1)
    n = D.shape[0]
    before = len(FAILS)
    for _ in range(trials):
        k = rng.randint(2, n)
        vs = rng.sample(range(n), k)
        full_old = lo.mst_weight(vs)
        full_new = ln.mst_weight(vs)
        check(full_old == full_new, f"peso difere: {full_old} vs {full_new} (k={k})")

        cutoff = rng.randint(max(1, full_old // 2), full_old * 2 + 1)
        a = lo.mst_weight(vs, cutoff)
        b = ln.mst_weight(vs, cutoff)
        check((a >= cutoff) == (b >= cutoff), f"decisao difere com cutoff={cutoff}")
        if a < cutoff:
            check(a == b == full_old, f"valor difere abaixo do cutoff: {a} vs {b}")
    print(f"  {trials} subconjuntos aleatorios: {'OK' if len(FAILS) == before else 'FALHAS'}")


def step2_bound(old_b, new_b, D, trials=300):
    print("\n2. remaining_bound (base) vs remaining_bound_shared (otimizado)")
    lo, ln = old_b.LowerBound(D), new_b.LowerBound(D)
    rng = random.Random(2)
    n = D.shape[0]
    before = len(FAILS)
    for _ in range(trials):
        depth = rng.randint(1, n - 4)  # prefixo com `depth` cidades
        path = [0] + rng.sample(range(1, n), depth - 1)
        visited = [False] * n
        for v in path:
            visited[v] = True
        unv = [u for u in range(n) if not visited[u]]
        shared = new_b.SharedMST(unv + [0])
        base = lo.mst_weight(unv + [0])

        order = unv[:]
        rng.shuffle(order)  # ordem arbitraria de filhos: testa o reuso
        for v in order:
            visited[v] = True
            child = [u for u in unv if u != v]
            cutoff = rng.choice(
                [float("inf"), rng.randint(1, base * 2 + 2), rng.randint(max(1, base // 2), base + 5)]
            )
            a = lo.remaining_bound(v, 0, visited, child, cutoff)
            b = ln.remaining_bound_shared(v, 0, visited, child, cutoff, shared)
            check((a >= cutoff) == (b >= cutoff), f"poda difere: cutoff={cutoff} a={a} b={b}")
            if a < cutoff:
                check(a == b, f"limite difere: {a} vs {b}")
            visited[v] = False
    print(f"  {trials} estados simulados: {'OK' if len(FAILS) == before else 'FALHAS'}")


def step3_bnb(old_s, new_s, TSPInstance, nn, sizes, seeds, tlimit):
    print("\n3. B&B completo: contadores identicos, tempo diferente")
    print(
        f"  {'instancia':<9}{'inicio':<6}{'expandidos base/otim':>26}{'podados base/otim':>28}"
        f"{'t base':>9}{'t otim':>9}{'speedup':>9}"
    )
    speedups = []
    for n in sizes:
        for seed in seeds:
            inst = make_instance(TSPInstance, n, seed)
            for label, ws in (("cold", None), ("NN", nn.solve(inst))):
                ro = old_s.solve(inst, initial_solution=ws, time_limit=tlimit)
                rn = new_s.solve(inst, initial_solution=ws, time_limit=tlimit)
                name = f"n{n}_s{seed}"
                if ro.timed_out or rn.timed_out:
                    print(f"  {name:<9}{label:<6}  inconclusiva (timeout: base={ro.timed_out}, otim={rn.timed_out})")
                    continue
                same = (
                    ro.nodes_expanded == rn.nodes_expanded
                    and ro.nodes_pruned == rn.nodes_pruned
                    and ro.max_depth == rn.max_depth
                    and ro.cost == rn.cost
                )
                check(
                    same,
                    f"{name}/{label}: contadores diferem "
                    f"(exp {ro.nodes_expanded}/{rn.nodes_expanded}, "
                    f"pod {ro.nodes_pruned}/{rn.nodes_pruned}, "
                    f"prof {ro.max_depth}/{rn.max_depth}, custo {ro.cost}/{rn.cost})",
                )
                sp = ro.execution_time / rn.execution_time if rn.execution_time > 0 else float("inf")
                speedups.append(sp)
                print(
                    f"  {name:<9}{label:<6}{ro.nodes_expanded:>13}/{rn.nodes_expanded:<12}"
                    f"{ro.nodes_pruned:>14}/{rn.nodes_pruned:<13}"
                    f"{ro.execution_time:>8.2f}s{rn.execution_time:>8.2f}s{sp:>8.2f}x"
                    + ("" if same else "  <-- DIFERE")
                )
    if speedups:
        print(f"\n  speedup: minimo {min(speedups):.2f}x | mediana {float(np.median(speedups)):.2f}x | maximo {max(speedups):.2f}x")


def step4_profile(old_s, new_s, TSPInstance, n, seed, tlimit):
    print(f"\n4. cProfile (n={n}, seed={seed}, cold-start): top por tempo proprio")
    inst = make_instance(TSPInstance, n, seed)
    for label, mod in (("BASE", old_s), ("OTIMIZADA", new_s)):
        pr = cProfile.Profile()
        pr.enable()
        res = mod.solve(inst, time_limit=tlimit)
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(8)
        print(f"\n--- {label}: {res.execution_time:.2f}s, {res.nodes_expanded} nos ---")
        print("\n".join(l for l in buf.getvalue().splitlines() if l.strip())[:2200])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--time-limit", type=float, default=300.0, help="s por execucao do B&B")
    ap.add_argument("--profile", action="store_true", help="imprime cProfile base/otimizada")
    ap.add_argument("--out", default=None, help="tambem grava a saida neste arquivo (UTF-8)")
    args = ap.parse_args()

    out_file = None
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(path, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, out_file)

    try:
        from src.algorithms.exact import bounds as old_b, branch_and_bound as old_s
        from src.algorithms.exact import bounds_fast as new_b, branch_and_bound_fast as new_s
        from src.algorithms.heuristics import nearest_neighbor as nn
        from src.tsp.instance import TSPInstance

        print("base: bounds.py / branch_and_bound.py  |  otimizada: bounds_fast.py / branch_and_bound_fast.py")

        D = make_instance(TSPInstance, 40, 7).distance_matrix
        step1_mst(old_b, new_b, D)
        step2_bound(old_b, new_b, D)
        step3_bnb(old_s, new_s, TSPInstance, nn, args.sizes, args.seeds, args.time_limit)
        if args.profile:
            step4_profile(old_s, new_s, TSPInstance, max(args.sizes), args.seeds[0], args.time_limit)

        print("\n" + "=" * 60)
        if FAILS:
            print(f"{len(FAILS)} falha(s):")
            for f in FAILS[:15]:
                print("  -", f)
            return 1
        print("OK: limites identicos, contadores identicos.")
        return 0
    finally:
        sys.stdout = sys.__stdout__
        if out_file:
            out_file.close()


if __name__ == "__main__":
    sys.exit(main())
