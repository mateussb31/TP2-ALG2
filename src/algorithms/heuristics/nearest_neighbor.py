"""Heurística do vizinho mais próximo (nearest neighbor) para o TSP.

Partindo de `start_node`, visita repetidamente a cidade ainda não visitada
mais próxima da cidade atual, até percorrer todas. O ciclo fecha voltando
ao ponto de partida. Empates são resolvidos pelo menor índice.

Complexidade: O(n^2) tempo (n passos, cada um com um argmin vetorizado
sobre uma linha da matriz) e O(n) memória extra.
"""

from __future__ import annotations

import numpy as np

try:
    from src.tsp.solution import Solution
except ImportError:  # execução fora de um pacote
    from solution import Solution


def solve(instance, start_node: int = 0) -> Solution:
    """Resolve o TSP pelo vizinho mais próximo.

    Args:
        instance: objeto com `n` e `distance_matrix`.
        start_node: vértice no qual o tour começa.

    Returns:
        Solution com tour válido (começando em start_node) e custo exato
        calculado via instance.distance_matrix.
    """
    n = int(instance.n)
    D = instance.distance_matrix

    if not (0 <= start_node < n):
        raise ValueError(f"start_node deve estar em [0, {n - 1}]; recebido {start_node}")

    visited = np.zeros(n, dtype=bool)
    tour = [start_node]
    visited[start_node] = True
    current = start_node

    big = np.iinfo(np.int64).max  # nunca escolhido enquanto houver não visitados

    for _ in range(n - 1):
        candidates = np.where(visited, big, D[current])
        nxt = int(np.argmin(candidates))  # empate: menor índice
        tour.append(nxt)
        visited[nxt] = True
        current = nxt

    solution = Solution(tour)
    solution.compute_cost(D)
    return solution
