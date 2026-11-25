"""Branch-and-Bound exato (busca em profundidade) para o TSP.

Estratégia de busca: DFS com ordenação "vizinho mais próximo primeiro".
  - Memória O(n): só o prefixo atual, o vetor de visitados e contadores. Uma
    busca best-first guardaria uma fronteira de tamanho exponencial.
  - A DFS chega a um tour completo rapidamente (parecido com o vizinho mais
    próximo), o que dá um limite superior cedo mesmo sem warm-start.
  - O tour é um ciclo, então a cidade 0 é fixada como origem (sem perda de
    generalidade) e as n! rotações viram (n-1)!.

Poda: um filho é cortado quando g + limite_inferior >= melhor custo. Os
custos são inteiros, então só interessa um tour estritamente melhor.

Otimização da MST do limite inferior (ver bounds.py): todos os filhos de um
nó avaliam a MST do mesmo conjunto (U_pai ∪ {origem}); ela é calculada uma
vez por nó e compartilhada (SharedMST). Os limites, e portanto os contadores
de nós, não mudam em relação à versão sem compartilhamento.

Warm-start: `initial_solution` inicializa o melhor custo e o melhor tour.
Se a busca não achar nada melhor, o tour inicial é devolvido (e é ótimo).

Timeout: `time.perf_counter()` é consultado a cada 128 avaliações de nó. Ao
estourar, a busca para e devolve o melhor estado, com os contadores até ali.

Contadores:
  nodes_expanded: nós (prefixos) visitados pela busca;
  nodes_pruned:   filhos gerados e cortados pelo limite, sem visitá-los;
  max_depth:      maior número de cidades num prefixo (n = tour completo).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from src.algorithms.exact.bounds_fast import LowerBound, SharedMST
    from src.tsp.solution import Solution
except ImportError:  # execução fora do pacote / layout alternativo
    from bounds_fast import LowerBound, SharedMST
    from solution import Solution

_CHECK_EVERY = 128  # avaliações de nó entre consultas ao relógio


@dataclass
class BnBResult:
    """Resultado do Branch-and-Bound.

    Atributos:
        solution: melhor Solution encontrada (None só se não houve warm-start
            e o tempo acabou antes do primeiro tour completo).
        cost: custo da melhor solução (inf se solution é None).
        nodes_expanded, nodes_pruned, max_depth: contadores da árvore.
        execution_time: segundos de relógio dentro de solve().
        timed_out: True se o limite de tempo interrompeu a busca (o custo
            então é um limite superior, não necessariamente o ótimo).
    """

    solution: Optional[Solution]
    cost: float
    nodes_expanded: int
    nodes_pruned: int
    max_depth: int
    execution_time: float
    timed_out: bool

    @property
    def status(self) -> str:
        """'completed' ou 'timeout' (formato lido pelo runner de experimentos)."""
        return "timeout" if self.timed_out else "completed"


class _Timeout(Exception):
    pass


class _Search:
    def __init__(self, lb: LowerBound, best_cost, best_tour, deadline: float):
        self.lb = lb
        self.n = lb.n
        self.Dl = lb.Dl
        self.best_cost = best_cost
        self.best_tour: Optional[List[int]] = best_tour
        self.deadline = deadline

        self.expanded = 0
        self.pruned = 0
        self.max_depth = 0
        self.ticks = 0
        self.timed_out = False

        self.path: List[int] = [0]
        self.visited = [False] * self.n
        self.visited[0] = True

    def run(self) -> None:
        try:
            self._dfs(0, 1, 0)
        except _Timeout:
            self.timed_out = True

    def _dfs(self, cur: int, depth: int, g: int) -> None:
        self.expanded += 1
        if depth > self.max_depth:
            self.max_depth = depth

        n = self.n
        Dl = self.Dl

        if depth == n:  # tour completo: fecha o ciclo na origem
            total = g + Dl[cur][0]
            if total < self.best_cost:
                self.best_cost = total
                self.best_tour = self.path[:]
            return

        path = self.path
        visited = self.visited
        lb = self.lb
        row = Dl[cur]

        # Filhos em ordem crescente de distância a partir de `cur`.
        children = [w for w in lb.neighbors[cur] if not visited[w]]

        # Cidades não visitadas neste nó (mesmo conjunto de `children`, em
        # ordem crescente). Cada filho v avalia a MST sobre
        # {v} ∪ (unv \ {v}) ∪ {0} = unv ∪ {0}, que não depende de v:
        # compartilha-se uma única MST entre todos os filhos.
        unv = [u for u in range(n) if not visited[u]]
        shared = SharedMST(unv + [0])

        for i, v in enumerate(children):
            g2 = g + row[v]
            best = self.best_cost
            if g2 >= best:
                # Ordenado por distância: os demais filhos custam >= g2.
                self.pruned += len(children) - i
                break

            self.ticks += 1
            if not (self.ticks % _CHECK_EVERY) and time.perf_counter() >= self.deadline:
                raise _Timeout

            visited[v] = True
            path.append(v)

            unvisited = [u for u in unv if u != v]
            bound = lb.remaining_bound_shared(v, 0, visited, unvisited, best - g2, shared)

            if g2 + bound >= best:
                self.pruned += 1
            else:
                self._dfs(v, depth + 1, g2)

            path.pop()
            visited[v] = False


def _tour_cost(D: np.ndarray, tour: List[int]) -> int:
    t = np.asarray(tour, dtype=np.int64)
    return int(D[t, np.roll(t, -1)].sum())


def solve(instance, initial_solution=None, time_limit: Optional[float] = 1800) -> BnBResult:
    """Resolve o TSP de forma exata (ou até estourar `time_limit`).

    Args:
        instance: objeto com `n` e `distance_matrix`.
        initial_solution: objeto com `tour` (warm-start). O custo é recalculado
            a partir do tour e da matriz. None = cold-start (limite infinito).
        time_limit: segundos; None = sem limite.

    Returns:
        BnBResult. Se `timed_out` for True, `cost` é o melhor custo achado até
        a interrupção, não necessariamente o ótimo.
    """
    t0 = time.perf_counter()
    n = int(instance.n)
    D = instance.distance_matrix

    if n <= 0:
        raise ValueError("instância vazia.")

    best_cost = float("inf")
    best_tour: Optional[List[int]] = None

    if initial_solution is not None:
        tour = [int(v) for v in initial_solution.tour]
        if sorted(tour) != list(range(n)):
            raise ValueError("initial_solution.tour não é uma permutação de 0..n-1.")
        best_tour = tour
        best_cost = _tour_cost(D, tour)

    deadline = float("inf") if time_limit is None else t0 + float(time_limit)

    lb = LowerBound(D)
    search = _Search(lb, best_cost, best_tour, deadline)
    search.run()

    if search.best_tour is None:
        solution, cost = None, float("inf")
    else:
        cost = int(search.best_cost)
        solution = Solution(list(search.best_tour), cost)

    return BnBResult(
        solution=solution,
        cost=cost,
        nodes_expanded=search.expanded,
        nodes_pruned=search.pruned,
        max_depth=search.max_depth,
        execution_time=time.perf_counter() - t0,
        timed_out=search.timed_out,
    )
