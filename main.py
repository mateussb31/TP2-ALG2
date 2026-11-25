#!/usr/bin/env python3
"""Teste funcional integrado do projeto TSP (corretude, não performance).

Uso (da raiz do repositório):
    python main.py                  # tudo, com as instâncias reais de data/raw/tsplib
    python main.py --skip-real      # só instâncias sintéticas (mais rápido)
    python main.py --bnb-time 10    # tempo (s) de cada B&B nas instâncias reais

O que verifica:
  1. imports de todos os módulos (nos caminhos do roteiro);
  2. distance / solution / instance;
  3. mst, matching e eulerian isolados (contra força bruta e networkx);
  4. as 4 heurísticas: tour válido, custo exato, start_node, razões 2x / 1,5x;
  5. branch-and-bound contra força bruta, cold-start e as 5 estratégias de
     warm-start; o warm-start com o tour ótimo nunca expande mais nós;
  6. timeout do branch-and-bound (contadores e melhor solução preservados);
  7. instâncias reais EUC_2D de data/raw/tsplib (gap em relação ao ótimo);
  8. integração com o runner da Pessoa 2 (src/experiments/adapters.py).

Sai com código 0 se tudo passar e 1 se algo falhar.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import importlib
import inspect
import itertools
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import networkx as nx
except ImportError:  # só usado como verificador independente
    nx = None

# Ótimos conhecidos da TSPLIB (EUC_2D). Conferir na página da TSPLIB se algum
# check de "ótimo" falhar de forma suspeita.
KNOWN_OPTIMA = {
    "berlin52": 7542,
    "eil51": 426,
    "st70": 675,
    "dj38": 6656,
    "wi29": 27603,
}


# --------------------------------------------------------------------------
# Mini framework de testes
# --------------------------------------------------------------------------
class Runner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures = []
        self.section_name = ""
        self.quiet = False

    def section(self, title):
        self.section_name = title
        print(f"\n=== {title} ===")

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            if not self.quiet:
                print(f"  [OK]      {name}")
        else:
            self.failed += 1
            suffix = f" -- {detail}" if detail else ""
            self.failures.append(f"{self.section_name} :: {name}{suffix}")
            print(f"  [FALHOU]  {name}{suffix}")
        return bool(cond)

    def skip(self, name, why=""):
        self.skipped += 1
        print(f"  [PULADO]  {name}" + (f" -- {why}" if why else ""))

    def info(self, text):
        print(f"  [info]    {text}")

    @contextlib.contextmanager
    def batch(self, label):
        """Agrupa muitas verificações numa única linha de resumo."""
        p0, f0 = self.passed, self.failed
        old, self.quiet = self.quiet, True
        try:
            yield
        finally:
            self.quiet = old
            dp, df = self.passed - p0, self.failed - f0
            tag = "OK" if df == 0 else "FALHOU"
            print(f"  [{tag}]".ljust(12) + f"{label} ({dp} ok, {df} falhas)")

    def guard(self, name, fn, *args, **kwargs):
        """Executa fn; qualquer exceção vira uma falha registrada."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            frame = traceback.extract_tb(e.__traceback__)[-1]
            where = f"{os.path.basename(frame.filename)}:{frame.lineno}"
            self.check(name, False, f"exceção {type(e).__name__}: {e} ({where})")
            return None


R = Runner()


def load_first(label, candidates):
    errs = []
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            R.check(f"importar {label} ({name})", True)
            return mod
        except Exception as e:  # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__}: {e}")
    R.check(f"importar {label}", False, " | ".join(errs))
    return None


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def cost_of(D, tour):
    n = len(tour)
    return int(sum(int(D[tour[i], tour[(i + 1) % n]]) for i in range(n)))


def brute_force(D):
    """(custo ótimo, tour ótimo) por enumeração. Só para n pequeno (<= 9)."""
    n = D.shape[0]
    Dl = D.tolist()
    best, best_tour = None, None
    for p in itertools.permutations(range(1, n)):
        tour = (0,) + p
        c = sum(Dl[tour[i]][tour[(i + 1) % n]] for i in range(n))
        if best is None or c < best:
            best, best_tour = c, list(tour)
    return int(best), best_tour


def brute_matching(Dl, vs):
    if not vs:
        return 0
    a, rest = vs[0], vs[1:]
    return min(
        Dl[a][b] + brute_matching(Dl, rest[:i] + rest[i + 1:])
        for i, b in enumerate(rest)
    )


def is_spanning_tree(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        a, b = find(e[0]), find(e[1])
        if a == b:
            return False
        parent[a] = b
    return len(edges) == n - 1


def call_with_start(fn, inst, start):
    """Chama fn(inst, start_node=start) se a assinatura aceitar start_node."""
    if "start_node" in inspect.signature(fn).parameters:
        return fn(inst, start_node=start)
    return fn(inst)


def validate(label, sol, inst, start=None):
    """Valida uma Solution (ou objeto equivalente) de forma independente."""
    n = inst.n
    if sol is None:
        return R.check(f"{label}: retornou uma solução", False)
    if not (hasattr(sol, "tour") and hasattr(sol, "cost")):
        return R.check(f"{label}: objeto tem tour e cost", False, repr(sol))

    tour = [int(v) for v in sol.tour]
    valid = sorted(tour) == list(range(n))
    if hasattr(sol, "is_valid"):
        valid = valid and bool(sol.is_valid(n))
    if not R.check(f"{label}: tour é permutação de 0..{n - 1}", valid, f"tour={tour[:20]}"):
        return False

    real = cost_of(inst.distance_matrix, tour)
    ok = R.check(
        f"{label}: custo confere com a matriz",
        sol.cost is not None and int(sol.cost) == real,
        f"cost={sol.cost} recalculado={real}",
    )
    if start is not None:
        ok &= R.check(f"{label}: tour começa em {start}", tour[0] == start, f"começa em {tour[0]}")
    return ok


def build_zoo(TSPInstance, seed):
    """Instâncias pequenas (n <= 8) com casos de borda, todas com força bruta viável."""
    rng = np.random.default_rng(seed)
    zoo = []

    def add(name, coords):
        zoo.append((name, TSPInstance(name, np.asarray(coords, dtype=float))))

    add("n1", [[5, 5]])
    add("n2", [[0, 0], [3, 4]])
    add("n3", [[0, 0], [3, 4], [6, 0]])
    add("quadrado4", [[0, 0], [10, 0], [10, 10], [0, 10]])
    for n in (5, 6, 7, 8):
        add(f"aleatoria{n}", rng.integers(0, 1000, (n, 2)))
    add("colinear6", [[i * 10, 0] for i in range(6)])
    add("duplicados6", [[0, 0], [0, 0], [10, 10], [10, 10], [20, 0], [5, 5]])
    add("identicos5", [[7, 7]] * 5)
    add(
        "agrupada8",
        np.vstack([rng.normal(100, 5, (4, 2)), rng.normal(800, 5, (4, 2))]).round(),
    )
    add(
        "fecho_com_internos8",
        [[0, 0], [100, 0], [100, 100], [0, 100], [50, 50], [30, 60], [70, 40], [20, 20]],
    )
    return zoo


def tsplib_header(path):
    info = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION") or i > 60:
                break
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip().upper()] = v.strip()
    return info


def plain_tsp(path, tmpdir):
    """Devolve um caminho .tsp descompactado (TSPInstance.from_file não lê .gz)."""
    path = Path(path)
    if path.suffix != ".gz":
        return path
    dest = Path(tmpdir) / (path.name.split(".")[0] + ".tsp")
    with gzip.open(path, "rb") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)
    return dest


# --------------------------------------------------------------------------
# 2. distance / solution / instance
# --------------------------------------------------------------------------
def test_distance(dist):
    f = dist.compute_distance_matrix

    D = f(np.array([[0, 0], [3, 4], [1, 1], [0, 0.5]]))
    R.check("forma (N, N)", D.shape == (4, 4))
    R.check("dtype inteiro", np.issubdtype(D.dtype, np.integer), str(D.dtype))
    R.check("(0,0)-(3,4) = 5", D[0, 1] == 5)
    R.check("nint: sqrt(2)=1,41 -> 1", D[0, 2] == 1)
    R.check("nint: 0,5 arredonda para cima -> 1", D[0, 3] == 1)

    D2 = f(np.array([[0, 0], [1.5, 0], [2.4, 0]]))
    R.check("nint: 1,5 -> 2", D2[0, 1] == 2, f"{D2[0, 1]}")
    R.check("nint: 2,4 -> 2", D2[0, 2] == 2, f"{D2[0, 2]}")
    R.check("nint: 0,9 -> 1", D2[1, 2] == 1, f"{D2[1, 2]}")

    rng = np.random.default_rng(0)
    c = rng.integers(0, 500, (40, 2)).astype(float)
    M = f(c)
    expected = np.array(
        [
            [int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5) for b in c]
            for a in c
        ]
    )
    R.check("igual à fórmula int(sqrt(dx²+dy²)+0,5) (40 pontos)", np.array_equal(M, expected))
    R.check("simétrica", np.array_equal(M, M.T))
    R.check("diagonal nula", np.all(np.diag(M) == 0))

    try:
        f(np.zeros((3, 3)))
        R.check("entrada com formato inválido levanta ValueError", False, "não levantou")
    except ValueError:
        R.check("entrada com formato inválido levanta ValueError", True)


def test_solution(Solution):
    R.check("is_valid: permutação válida", Solution([2, 0, 1, 3]).is_valid(4))
    R.check("is_valid: repetido é inválido", not Solution([0, 0, 1, 2]).is_valid(4))
    R.check("is_valid: fora do intervalo é inválido", not Solution([0, 1, 2, 4]).is_valid(4))
    R.check("is_valid: tamanho errado é inválido", not Solution([0, 1, 2]).is_valid(4))
    R.check("is_valid: n=0 é inválido", not Solution([]).is_valid(0))
    R.check("is_valid: aceita inteiros do NumPy", Solution(list(np.arange(5))).is_valid(5))

    D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    s = Solution([0, 1, 2])
    c = s.compute_cost(D)
    R.check("compute_cost: 0-1-2-0 = 1+3+2 = 6", c == 6, f"{c}")
    R.check("compute_cost guarda em .cost", s.cost == 6)
    R.check("compute_cost: tour de 1 vértice = 0", Solution([0]).compute_cost(D) == 0)
    R.check("custo é inteiro", isinstance(c, (int, np.integer)), type(c).__name__)


def test_instance(TSPInstance, tmp):
    content = (
        "NAME : teste5\nCOMMENT : cinco vertices\nTYPE : TSP\nDIMENSION : 5\n"
        "EDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
        "1 0 0\n2 3 4\n3 10 0\n4 10 10\n5 0 1.0e+01\nEOF\n"
    )
    p = Path(tmp) / "teste5.tsp"
    p.write_text(content)
    inst = TSPInstance.from_file(str(p))
    R.check("name lido", inst.name == "teste5", inst.name)
    R.check("n = 5", inst.n == 5)
    R.check("coordinates (5, 2)", np.asarray(inst.coordinates).shape == (5, 2))
    R.check("coordenada do vértice 2 = (3, 4)", list(inst.coordinates[1]) == [3.0, 4.0])
    R.check("notação científica lida (5 -> (0, 10))", list(inst.coordinates[4]) == [0.0, 10.0])
    R.check("distance_matrix (5, 5)", inst.distance_matrix.shape == (5, 5))
    R.check("d(1,2) = 5", inst.distance_matrix[0, 1] == 5)
    R.check("matriz simétrica", np.array_equal(inst.distance_matrix, inst.distance_matrix.T))

    p2 = Path(tmp) / "geo.tsp"
    p2.write_text(content.replace("EUC_2D", "GEO"))
    try:
        TSPInstance.from_file(str(p2))
        R.check("EDGE_WEIGHT_TYPE != EUC_2D levanta ValueError", False, "não levantou")
    except ValueError:
        R.check("EDGE_WEIGHT_TYPE != EUC_2D levanta ValueError", True)

    p3 = Path(tmp) / "dim.tsp"
    p3.write_text(content.replace("DIMENSION : 5", "DIMENSION : 7"))
    try:
        TSPInstance.from_file(str(p3))
        R.check("DIMENSION inconsistente levanta ValueError", False, "não levantou")
    except ValueError:
        R.check("DIMENSION inconsistente levanta ValueError", True)


# --------------------------------------------------------------------------
# 3. mst / matching / eulerian
# --------------------------------------------------------------------------
def call_mst(f, inst, start=None):
    D = inst.distance_matrix
    try:
        r = f(D) if start is None else f(D, start)
    except (TypeError, AttributeError, IndexError):
        r = f(inst) if start is None else f(inst, start)
    if hasattr(r, "edges"):
        return list(r.edges), r.cost
    return list(r[0]), r[1]


def test_mst(mst_mod, zoo, TSPInstance):
    f = mst_mod.minimum_spanning_tree
    rng = np.random.default_rng(11)
    cases = list(zoo) + [
        (f"aleatoria25_{s}", TSPInstance("r", rng.integers(0, 1000, (25, 2)).astype(float)))
        for s in range(3)
    ]
    with R.batch(f"MST em {len(cases)} instâncias"):
        for name, inst in cases:
            n, D = inst.n, inst.distance_matrix
            edges, cost = call_mst(f, inst)
            R.check(f"{name}: {n - 1} arestas", len(edges) == n - 1, f"{len(edges)}")
            R.check(f"{name}: é árvore geradora", is_spanning_tree(n, edges))
            R.check(
                f"{name}: pesos das arestas = D[u,v]",
                all(int(w) == int(D[u, v]) for u, v, w in edges),
            )
            R.check(f"{name}: custo = soma dos pesos", int(cost) == sum(int(w) for _, _, w in edges))
            if nx is not None and n >= 2:
                G = nx.Graph()
                G.add_nodes_from(range(n))
                G.add_weighted_edges_from(
                    (i, j, int(D[i, j])) for i in range(n) for j in range(i + 1, n)
                )
                ref = int(nx.minimum_spanning_tree(G).size(weight="weight"))
                R.check(f"{name}: custo = MST do networkx", int(cost) == ref, f"{cost} vs {ref}")
            if n >= 2:
                try:
                    _, cost2 = call_mst(f, inst, n // 2)
                    R.check(f"{name}: custo independe da raiz", int(cost2) == int(cost))
                except TypeError:
                    pass  # assinatura sem start_node
    if nx is None:
        R.info("networkx ausente: comparação da MST com networkx foi pulada")


def mst_odd_vertices(n, edges):
    deg = [0] * n
    for e in edges:
        deg[e[0]] += 1
        deg[e[1]] += 1
    return [v for v in range(n) if deg[v] % 2 == 1]


def test_matching(match_mod, mst_mod, TSPInstance):
    f = match_mod.min_weight_perfect_matching
    rng = np.random.default_rng(5)

    with R.batch("emparelhamento perfeito mínimo vs força bruta"):
        for s in range(6):
            n = 14
            inst = TSPInstance("m", rng.integers(0, 1000, (n, 2)).astype(float))
            D, Dl = inst.distance_matrix, inst.distance_matrix.tolist()
            vs = sorted(int(v) for v in rng.choice(n, size=8, replace=False))
            m = f(inst, vs)
            covered = sorted(x for e in m for x in (e[0], e[1]))
            R.check(f"s{s}: cada vértice em exatamente uma aresta", covered == vs, f"{covered} vs {vs}")
            R.check(f"s{s}: pesos = D[u,v]", all(int(w) == int(D[u, v]) for u, v, w in m))
            total = sum(int(w) for _, _, w in m)
            opt = brute_matching(Dl, vs)
            R.check(f"s{s}: custo ótimo", total == opt, f"{total} vs ótimo {opt}")

    with R.batch("matching sobre vértices ímpares da MST"):
        for s in range(4):
            inst = TSPInstance("m2", rng.integers(0, 1000, (20, 2)).astype(float))
            edges, _ = call_mst(mst_mod.minimum_spanning_tree, inst)
            odd = mst_odd_vertices(inst.n, edges)
            R.check(f"s{s}: nº de vértices ímpares é par", len(odd) % 2 == 0)
            m = f(inst, odd)
            covered = sorted(x for e in m for x in (e[0], e[1]))
            R.check(f"s{s}: emparelhamento perfeito", covered == sorted(odd))

    inst = TSPInstance("m3", rng.integers(0, 1000, (6, 2)).astype(float))
    R.check("lista vazia -> []", list(f(inst, [])) == [])
    for label, arg in (
        ("quantidade ímpar", [0, 1, 2]),
        ("vértices repetidos", [0, 0, 1, 2]),
        ("índice fora do intervalo", [0, 99]),
    ):
        try:
            f(inst, arg)
            R.check(f"{label} levanta ValueError", False, "não levantou")
        except ValueError:
            R.check(f"{label} levanta ValueError", True)


def check_circuit(label, find, n, edges):
    c = find(n, edges)
    m = len(edges)
    ok = R.check(f"{label}: {m + 1} vértices no circuito", len(c) == m + 1, f"len={len(c)}")
    if not ok:
        return
    R.check(f"{label}: circuito fecha (primeiro == último)", c[0] == c[-1])
    used = Counter(tuple(sorted((a, b))) for a, b in zip(c, c[1:]))
    want = Counter(tuple(sorted((int(e[0]), int(e[1])))) for e in edges)
    R.check(f"{label}: cada aresta usada exatamente uma vez", used == want)


def test_eulerian(eul_mod, mst_mod, match_mod, TSPInstance):
    find = eul_mod.find_eulerian_circuit
    check_circuit("triângulo", find, 3, [(0, 1), (1, 2), (2, 0)])
    check_circuit("aresta paralela", find, 3, [(0, 1), (1, 2), (2, 0), (0, 1), (1, 0)])
    check_circuit("laço", find, 2, [(0, 0), (0, 1), (1, 0)])
    check_circuit("arestas com peso (u, v, w)", find, 3, [(0, 1, 5), (1, 2, 5), (2, 0, 5)])

    rng = np.random.default_rng(3)
    inst = TSPInstance("e", rng.integers(0, 1000, (16, 2)).astype(float))
    edges, _ = call_mst(mst_mod.minimum_spanning_tree, inst)
    doubled = [(u, v) for u, v, _ in edges] * 2
    check_circuit("MST duplicada (twice-around-the-tree)", find, inst.n, doubled)

    odd = mst_odd_vertices(inst.n, edges)
    matching = match_mod.min_weight_perfect_matching(inst, odd)
    christo = [(u, v) for u, v, _ in edges] + [(u, v) for u, v, _ in matching]
    check_circuit("MST + emparelhamento (Christofides)", find, inst.n, christo)

    for label, n, es in (
        ("grau ímpar", 3, [(0, 1), (1, 2)]),
        ("grafo desconexo", 6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]),
    ):
        try:
            find(n, es)
            R.check(f"{label} levanta ValueError", False, "não levantou")
        except ValueError:
            R.check(f"{label} levanta ValueError", True)

    R.check("sem arestas -> lista vazia", list(find(3, [])) == [])


# --------------------------------------------------------------------------
# 4. heurísticas
# --------------------------------------------------------------------------
def test_heuristics(algos, zoo, TSPInstance, mst_mod):
    with R.batch(f"{len(algos)} heurísticas x {len(zoo)} instâncias x vários start_node"):
        for name, inst in zoo:
            n = inst.n
            for aname, fn in algos.items():
                for start in sorted({0, n // 2, n - 1}):
                    label = f"{aname}/{name}/start={start}"
                    sol = R.guard(label, call_with_start, fn, inst, start)
                    if sol is not None:
                        validate(label, sol, inst, start)

    # Razões de aproximação e limite inferior, contra o ótimo por força bruta.
    # Folga de n unidades: o arredondamento EUC_2D pode violar a desigualdade
    # triangular por 1 em cada atalho.
    with R.batch("custo >= ótimo e razões 2x (TAT) / 1,5x (Christofides)"):
        for name, inst in zoo:
            n = inst.n
            if n < 3:
                continue
            opt, _ = brute_force(inst.distance_matrix)
            for aname, fn in algos.items():
                sol = R.guard(f"{aname}/{name}", call_with_start, fn, inst, 0)
                if sol is None or not hasattr(sol, "tour"):
                    continue
                c = cost_of(inst.distance_matrix, [int(v) for v in sol.tour])
                R.check(f"{aname}/{name}: custo >= ótimo", c >= opt, f"{c} < {opt}")
                if aname.startswith("TwiceAround"):
                    R.check(f"{aname}/{name}: custo <= 2*ótimo", c <= 2 * opt + n, f"{c} vs 2*{opt}")
                if aname == "Christofides":
                    R.check(
                        f"{aname}/{name}: custo <= 1,5*ótimo",
                        c <= 1.5 * opt + n,
                        f"{c} vs 1.5*{opt}",
                    )

    # Instância média: sem força bruta, mas custo >= MST (a MST é limite inferior do ótimo).
    rng = np.random.default_rng(21)
    inst = TSPInstance("u30", rng.integers(0, 1000, (30, 2)).astype(float))
    _, mst_cost = call_mst(mst_mod.minimum_spanning_tree, inst)
    with R.batch("instância de 30 pontos: validade e custo >= peso da MST"):
        for aname, fn in algos.items():
            sol = R.guard(f"{aname}/u30", call_with_start, fn, inst, 3)
            if sol is not None and validate(f"{aname}/u30", sol, inst, 3):
                R.check(f"{aname}/u30: custo >= MST", int(sol.cost) >= int(mst_cost), f"{sol.cost} < {mst_cost}")
                if aname.startswith("TwiceAround"):
                    R.check(f"{aname}/u30: custo <= 2*MST", int(sol.cost) <= 2 * int(mst_cost) + inst.n)


# --------------------------------------------------------------------------
# 5. branch-and-bound
# --------------------------------------------------------------------------
RESULT_FIELDS = ("solution", "cost", "nodes_expanded", "nodes_pruned", "max_depth", "execution_time", "timed_out")


def check_bnb_result(label, res, inst, opt=None, expect_completed=True):
    n = inst.n
    missing = [f for f in RESULT_FIELDS if not hasattr(res, f)]
    if not R.check(f"{label}: BnBResult tem todos os campos", not missing, f"faltam {missing}"):
        return False
    if res.solution is None:
        return R.check(f"{label}: retornou solução", False)
    ok = validate(label, res.solution, inst)
    R.check(f"{label}: cost == solution.cost", int(res.cost) == int(res.solution.cost), f"{res.cost} vs {res.solution.cost}")
    R.check(f"{label}: cost é inteiro", isinstance(res.cost, (int, np.integer)), type(res.cost).__name__)
    R.check(f"{label}: nodes_expanded >= 1", res.nodes_expanded >= 1, f"{res.nodes_expanded}")
    R.check(f"{label}: nodes_pruned >= 0", res.nodes_pruned >= 0)
    R.check(f"{label}: 1 <= max_depth <= n", 1 <= res.max_depth <= n, f"{res.max_depth}")
    R.check(f"{label}: execution_time >= 0", res.execution_time >= 0)
    if expect_completed:
        R.check(f"{label}: não estourou o tempo", res.timed_out is False)
        if hasattr(res, "status"):
            R.check(f"{label}: status == 'completed'", res.status == "completed", f"{res.status}")
        if opt is not None:
            R.check(f"{label}: custo == ótimo", int(res.cost) == int(opt), f"{res.cost} vs ótimo {opt}")
    return ok


def test_bnb_exact(bnb, algos, zoo, Solution):
    small = [(nm, inst) for nm, inst in zoo if inst.n <= 8]
    with R.batch(f"B&B vs força bruta em {len(small)} instâncias (cold + 5 warm-starts)"):
        for name, inst in small:
            n, D = inst.n, inst.distance_matrix
            opt, opt_tour = brute_force(D)

            cold = R.guard(f"cold/{name}", bnb.solve, inst)
            if cold is None:
                continue
            check_bnb_result(f"cold/{name}", cold, inst, opt)

            warms = {}
            triv = Solution(list(range(n)))
            triv.compute_cost(D)
            warms["trivial"] = triv
            for aname, fn in algos.items():
                s = R.guard(f"warm-{aname}/{name} (heurística)", call_with_start, fn, inst, 0)
                if s is not None:
                    warms[aname] = s
            for wname, ws in warms.items():
                res = R.guard(f"warm-{wname}/{name}", bnb.solve, inst, initial_solution=ws)
                if res is not None:
                    check_bnb_result(f"warm-{wname}/{name}", res, inst, opt)
                    R.check(f"warm-{wname}/{name}: custo <= custo inicial", int(res.cost) <= cost_of(D, [int(v) for v in ws.tour]))

            # warm-start com o tour ótimo: custo ótimo e nunca expande mais nós que o cold-start
            best_ws = Solution(list(opt_tour))
            best_ws.compute_cost(D)
            res = R.guard(f"warm-ótimo/{name}", bnb.solve, inst, initial_solution=best_ws)
            if res is not None:
                check_bnb_result(f"warm-ótimo/{name}", res, inst, opt)
                R.check(
                    f"warm-ótimo/{name}: nós expandidos <= cold-start",
                    res.nodes_expanded <= cold.nodes_expanded,
                    f"{res.nodes_expanded} > {cold.nodes_expanded}",
                )

            if n >= 3:
                res = R.guard(f"time_limit=None/{name}", bnb.solve, inst, None, None)
                if res is not None:
                    R.check(f"time_limit=None/{name}: custo ótimo", int(res.cost) == opt)

    n4 = [inst for nm, inst in zoo if nm == "quadrado4"][0]
    try:
        bnb.solve(n4, initial_solution=Solution([0, 0, 1, 2]))
        R.check("initial_solution inválida levanta ValueError", False, "não levantou")
    except ValueError:
        R.check("initial_solution inválida levanta ValueError", True)


def test_bnb_timeout(bnb, nn_mod, TSPInstance):
    rng = np.random.default_rng(99)
    inst = TSPInstance("u40", rng.integers(0, 1000, (40, 2)).astype(float))
    limit = 0.5

    t0 = time.perf_counter()
    res = bnb.solve(inst, time_limit=limit)
    el = time.perf_counter() - t0
    R.check("cold-start n=40: estourou o tempo", res.timed_out is True)
    R.check("cold-start n=40: status == 'timeout'", getattr(res, "status", None) == "timeout", f"{getattr(res, 'status', None)}")
    R.check("cold-start n=40: voltou em tempo razoável", el < 5.0, f"{el:.2f}s")
    R.check("cold-start n=40: contadores preservados", res.nodes_expanded > 0 and res.max_depth >= 1, f"{res.nodes_expanded}/{res.max_depth}")
    if res.solution is not None:
        validate("cold-start n=40 (parcial)", res.solution, inst)
    R.info(f"cold-start n=40 em {limit}s: custo={res.cost} expandidos={res.nodes_expanded} podados={res.nodes_pruned}")

    warm = nn_mod.solve(inst)
    res = bnb.solve(inst, initial_solution=warm, time_limit=limit)
    R.check("warm-start n=40: estourou o tempo", res.timed_out is True)
    R.check("warm-start n=40: devolve solução", res.solution is not None)
    if res.solution is not None:
        validate("warm-start n=40 (parcial)", res.solution, inst)
        R.check("warm-start n=40: custo <= custo da heurística", int(res.cost) <= int(warm.cost), f"{res.cost} > {warm.cost}")
    R.info(f"warm-start(NN) n=40 em {limit}s: custo={res.cost} (NN={warm.cost}) expandidos={res.nodes_expanded} podados={res.nodes_pruned}")


# --------------------------------------------------------------------------
# 7. instâncias reais
# --------------------------------------------------------------------------
def test_real(algos, bnb, TSPInstance, Solution, data_dir, tmp, bnb_time):
    files = sorted(p for p in Path(data_dir).glob("*.tsp*") if p.suffix in (".tsp", ".gz"))
    if not files:
        R.skip("instâncias reais", f"nenhum arquivo em {data_dir}")
        return

    loaded = []
    for p in files:
        stem = p.name.split(".")[0].lower()
        info = tsplib_header(p)
        typ = info.get("EDGE_WEIGHT_TYPE", "?").upper()
        if typ != "EUC_2D":
            R.skip(f"{stem}", f"EDGE_WEIGHT_TYPE={typ} (só EUC_2D é suportado)")
            continue
        inst = R.guard(f"carregar {stem}", lambda: TSPInstance.from_file(str(plain_tsp(p, tmp))))
        if inst is None:
            continue
        R.check(f"{stem}: n == DIMENSION", inst.n == int(info.get("DIMENSION", -1)), f"{inst.n} vs {info.get('DIMENSION')}")
        R.check(f"{stem}: matriz simétrica, diagonal nula", np.array_equal(inst.distance_matrix, inst.distance_matrix.T) and not np.diag(inst.distance_matrix).any())
        loaded.append((stem, inst))

    if not loaded:
        return

    names = list(algos)
    print(f"\n  {'instância':<10}{'n':>4}{'ótimo':>8}  " + "  ".join(f"{a[:12]:>18}" for a in names))
    for stem, inst in loaded:
        opt = KNOWN_OPTIMA.get(stem)
        cells = []
        for aname, fn in algos.items():
            t0 = time.perf_counter()
            sol = R.guard(f"{aname}/{stem}", call_with_start, fn, inst, 0)
            dt = time.perf_counter() - t0
            if sol is None:
                cells.append(f"{'ERRO':>18}")
                continue
            ok = validate(f"{aname}/{stem}", sol, inst, 0)
            if ok and opt:
                R.check(f"{aname}/{stem}: custo >= ótimo ({opt})", int(sol.cost) >= opt, f"{sol.cost} < {opt}")
                if aname.startswith("TwiceAround"):
                    R.check(f"{aname}/{stem}: custo <= 2*ótimo", int(sol.cost) <= 2 * opt + inst.n)
                if aname == "Christofides":
                    R.check(f"{aname}/{stem}: custo <= 1,5*ótimo", int(sol.cost) <= 1.5 * opt + inst.n)
            gap = f"{100.0 * (int(sol.cost) - opt) / opt:+.1f}%" if opt else "-"
            cells.append(f"{int(sol.cost):>8} {gap:>6} {dt:>3.1f}s"[:18].rjust(18))
        print(f"  {stem:<10}{inst.n:>4}{(opt or '-'):>8}  " + "  ".join(cells))

    # B&B na menor instância real: cold + 5 warm-starts, com limite de tempo
    stem, inst = min(loaded, key=lambda t: t[1].n)
    opt = KNOWN_OPTIMA.get(stem)
    D = inst.distance_matrix
    print(f"\n  B&B em {stem} (n={inst.n}), limite {bnb_time}s por execução:")
    starts = {"cold": None}
    triv = Solution(list(range(inst.n)))
    triv.compute_cost(D)
    starts["trivial"] = triv
    for aname, fn in algos.items():
        starts[aname] = R.guard(f"warm {aname}/{stem}", call_with_start, fn, inst, 0)
    for sname, ws in starts.items():
        if sname != "cold" and ws is None:
            continue
        res = R.guard(f"B&B {sname}/{stem}", bnb.solve, inst, ws, bnb_time)
        if res is None:
            continue
        label = f"B&B {sname}/{stem}"
        check_bnb_result(label, res, inst, opt, expect_completed=False)
        if opt and res.solution is not None:
            R.check(f"{label}: custo >= ótimo", int(res.cost) >= opt, f"{res.cost} < {opt}")
            if not res.timed_out:
                R.check(f"{label}: terminou e achou o ótimo", int(res.cost) == opt, f"{res.cost} vs {opt}")
        if ws is not None and res.solution is not None:
            R.check(f"{label}: custo <= custo inicial", int(res.cost) <= cost_of(D, [int(v) for v in ws.tour]))
        R.info(
            f"{sname:<9} custo={res.cost} expandidos={res.nodes_expanded} podados={res.nodes_pruned} "
            f"prof={res.max_depth} t={res.execution_time:.2f}s timeout={res.timed_out}"
        )


# --------------------------------------------------------------------------
# 8. integração com o runner da Pessoa 2
# --------------------------------------------------------------------------
def test_adapters(TSPInstance, tmp):
    if os.environ.get("TP2_FAKE") == "1":
        R.skip("integração com adapters", "TP2_FAKE=1 está ativo (algoritmos falsos)")
        return
    try:
        from src.experiments import adapters, tsplib_io
    except Exception as e:  # noqa: BLE001
        R.check("importar src.experiments.adapters", False, f"{type(e).__name__}: {e}")
        return
    R.check("importar src.experiments.adapters", True)

    rng = np.random.default_rng(8)
    coords = rng.integers(0, 1000, (8, 2)).astype(float)
    path = Path(tmp) / "integ8.tsp"
    tsplib_io.write_tsplib(path, "integ8", coords, "teste de integracao")
    inst = adapters.load_instance(path, "nint")
    ref = TSPInstance.from_file(str(path))
    R.check(
        "adapters.load_instance e TSPInstance.from_file dão a mesma matriz",
        np.array_equal(inst.distance_matrix, ref.distance_matrix),
    )

    sols = {}
    for name in ("trivial", "nn", "hull", "tat", "chr"):
        fn = R.guard(f"adapters.heuristic('{name}')", adapters.heuristic, name)
        if fn is None:
            continue
        sol = R.guard(f"adapters '{name}' executa", fn, inst)
        if sol is not None and validate(f"adapters '{name}'", sol, inst):
            sols[name] = sol

    opt, _ = brute_force(inst.distance_matrix)
    run = R.guard("adapters.bnb()", adapters.bnb)
    if run is None:
        return
    for name, ws in sols.items():
        res = R.guard(f"adapters B&B (warm {name})", run, inst, ws, 30)
        if res is None:
            continue
        label = f"adapters B&B (warm {name})"
        missing = [f for f in RESULT_FIELDS + ("status",) if not hasattr(res, f)]
        if not R.check(f"{label}: resultado tem os campos lidos pelo worker", not missing, f"faltam {missing}"):
            continue
        raw = str(res.status).lower()
        R.check(f"{label}: status interpretado como 'completed'", raw.startswith(("complete", "optimal")), f"status={res.status!r}")
        R.check(f"{label}: custo == ótimo", int(res.cost) == opt, f"{res.cost} vs {opt}")
        validate(label, res.solution, inst)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Teste funcional integrado do projeto TSP")
    ap.add_argument("--skip-real", action="store_true", help="não usar instâncias de data/raw/tsplib")
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "tsplib"))
    ap.add_argument("--bnb-time", type=float, default=5.0, help="limite (s) de cada B&B nas instâncias reais")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    t_start = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="tsp_main_")
    try:
        # 1. imports
        R.section("1. Imports")
        dist = load_first("distance", ["src.tsp.distance"])
        inst_mod = load_first("instance", ["src.tsp.instance"])
        sol_mod = load_first("solution", ["src.tsp.solution"])
        mst_mod = load_first("mst", ["src.graph.mst", "graph.mst"])
        match_mod = load_first("matching", ["src.graph.matching", "graph.matching"])
        eul_mod = load_first("eulerian", ["src.graph.eulerian", "graph.eulerian"])
        nn_mod = load_first("nearest_neighbor", ["src.algorithms.heuristics.nearest_neighbor"])
        hull_mod = load_first("convex_hull_insertion", ["src.algorithms.heuristics.convex_hull_insertion"])
        tat_mod = load_first(
            "twice_around_the_tree",
            [
                "src.algorithms.heuristics.twice_around_the_tree",
                "src.algorithms.heuristics.twice_around_tree",
            ],
        )
        chr_mod = load_first("christofides", ["src.algorithms.heuristics.christofides"])
        load_first("bounds", ["src.algorithms.exact.bounds"])
        bnb = load_first("branch_and_bound", ["src.algorithms.exact.branch_and_bound"])

        TSPInstance = getattr(inst_mod, "TSPInstance", None) if inst_mod else None
        Solution = getattr(sol_mod, "Solution", None) if sol_mod else None
        if TSPInstance is None or Solution is None:
            print("\nTSPInstance/Solution indisponíveis: nada mais pode ser testado.")
            return finish(t_start)

        # 2. fundação
        R.section("2. distance / solution / instance")
        if dist:
            R.guard("distance", test_distance, dist)
        R.guard("solution", test_solution, Solution)
        R.guard("instance", test_instance, TSPInstance, tmp)

        zoo = build_zoo(TSPInstance, args.seed)

        # 3. grafos
        R.section("3. mst / matching / eulerian")
        if mst_mod:
            R.guard("mst", test_mst, mst_mod, zoo, TSPInstance)
        else:
            R.skip("mst", "módulo não importou")
        if match_mod and mst_mod:
            R.guard("matching", test_matching, match_mod, mst_mod, TSPInstance)
        else:
            R.skip("matching", "dependência não importou")
        if eul_mod and mst_mod and match_mod:
            R.guard("eulerian", test_eulerian, eul_mod, mst_mod, match_mod, TSPInstance)
        else:
            R.skip("eulerian", "dependência não importou")

        # 4. heurísticas
        R.section("4. Heurísticas")
        algos = {}
        if nn_mod:
            algos["NN"] = nn_mod.solve
        if hull_mod:
            algos["Hull"] = hull_mod.solve
        if tat_mod:
            algos["TwiceAround"] = tat_mod.solve
        if chr_mod:
            algos["Christofides"] = chr_mod.solve
        if algos and mst_mod:
            R.guard("heurísticas", test_heuristics, algos, zoo, TSPInstance, mst_mod)
        else:
            R.skip("heurísticas", "nenhuma heurística (ou a MST) importou")

        # 5/6. branch-and-bound
        R.section("5. Branch-and-Bound (corretude contra força bruta)")
        if bnb:
            R.guard("B&B exato", test_bnb_exact, bnb, algos, zoo, Solution)
        else:
            R.skip("B&B", "módulo não importou")

        R.section("6. Branch-and-Bound (timeout)")
        if bnb and nn_mod:
            R.guard("B&B timeout", test_bnb_timeout, bnb, nn_mod, TSPInstance)
        else:
            R.skip("B&B timeout", "dependência não importou")

        # 7. instâncias reais
        R.section("7. Instâncias reais (TSPLIB EUC_2D)")
        if args.skip_real:
            R.skip("instâncias reais", "--skip-real")
        elif algos and bnb:
            R.guard("instâncias reais", test_real, algos, bnb, TSPInstance, Solution, args.data_dir, tmp, args.bnb_time)
        else:
            R.skip("instâncias reais", "dependência não importou")

        # 8. integração com a Pessoa 2
        R.section("8. Integração com o runner da Pessoa 2 (adapters)")
        R.guard("adapters", test_adapters, TSPInstance, tmp)

        return finish(t_start)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def finish(t_start):
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"RESUMO: {R.passed} ok, {R.failed} falhas, {R.skipped} puladas  ({elapsed:.1f}s)")
    if R.failures:
        print("\nFalhas:")
        for f in R.failures:
            print(f"  - {f}")
        return 1
    print("Tudo certo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
