"""Heurística de inserção a partir do fecho convexo para o TSP.

1. Calcula o fecho convexo das coordenadas e usa seus vértices (na ordem do
   perímetro) como subtour inicial.
2. Insere iterativamente o vértice k e a aresta (i, j) do tour atual que
   minimizam  delta = d(i,k) + d(k,j) - d(i,j).
3. Rotaciona o ciclo para começar em `start_node` e devolve uma Solution
   com custo calculado exatamente via instance.distance_matrix.

Implementação eficiente: para cada vértice restante guardamos a melhor
aresta de inserção. Após cada inserção, só é preciso (a) comparar com as duas
arestas novas e (b) recalcular por completo os vértices cuja melhor aresta
foi a que acabou de ser quebrada.
"""

from __future__ import annotations

import numpy as np

try:
    from src.tsp.solution import Solution
except ImportError:  # execução fora de um pacote
    from solution import Solution


def _initial_subtour(instance, start_node: int) -> list:
    """Vértices do fecho convexo na ordem do perímetro (anti-horária).

    Em casos degenerados (pontos colineares/coincidentes, fecho com menos
    de 3 vértices ou falha do Qhull), usa `start_node` e o vértice mais
    distante dele como subtour inicial.
    """
    from scipy.spatial import ConvexHull

    D = instance.distance_matrix
    n = instance.n

    try:
        hull = ConvexHull(np.asarray(instance.coordinates, dtype=np.float64))
        vertices = [int(v) for v in hull.vertices]  # ordem anti-horária em 2D
        if len(vertices) >= 3:
            return vertices
    except Exception:
        pass

    a = start_node
    b = int(np.argmax(D[a]))
    if b == a or D[a, b] == 0:
        b = (a + 1) % n
    return [a, b]


def solve(instance, start_node: int = 0) -> Solution:
    """Resolve o TSP por inserção a partir do fecho convexo.

    Args:
        instance: objeto com `n`, `coordinates` e `distance_matrix`.
        start_node: vértice no qual o tour retornado deve começar.

    Returns:
        Solution com tour válido (começando em start_node) e custo exato.
    """
    n = int(instance.n)
    D = instance.distance_matrix

    if not (0 <= start_node < n):
        raise ValueError(f"start_node deve estar em [0, {n - 1}]; recebido {start_node}")

    if n <= 3:
        tour = list(range(n))
    else:
        tour = _initial_subtour(instance, start_node)

        in_tour = np.zeros(n, dtype=bool)
        in_tour[tour] = True
        remaining = np.flatnonzero(~in_tour)

        best_delta = np.full(n, np.inf, dtype=np.float64)
        best_i = np.zeros(n, dtype=np.int64)  # extremo inicial da melhor aresta
        best_j = np.zeros(n, dtype=np.int64)  # extremo final da melhor aresta

        if remaining.size > 0:
            t = np.asarray(tour, dtype=np.int64)
            nxt = np.roll(t, -1)
            base = D[t, nxt]
            M = D[np.ix_(remaining, t)] + D[np.ix_(remaining, nxt)] - base[None, :]
            idx = np.argmin(M, axis=1)
            best_delta[remaining] = M[np.arange(remaining.size), idx]
            best_i[remaining] = t[idx]
            best_j[remaining] = nxt[idx]

        while True:
            k = int(np.argmin(best_delta))
            if not np.isfinite(best_delta[k]):
                break  # nenhum vértice restante

            a = int(best_i[k])
            b = int(best_j[k])

            # Insere k entre a e b (a aresta a -> b é consecutiva no tour).
            pos = tour.index(a)
            tour.insert(pos + 1, k)
            best_delta[k] = np.inf

            cand = np.flatnonzero(np.isfinite(best_delta))
            if cand.size == 0:
                break

            stale_mask = (best_i[cand] == a) & (best_j[cand] == b)
            stale = cand[stale_mask]
            fresh = cand[~stale_mask]

            # Vértices cuja melhor aresta continua existindo: comparar só
            # com as duas arestas novas (a, k) e (k, b).
            if fresh.size > 0:
                d1 = D[fresh, a] + D[fresh, k] - D[a, k]
                better = d1 < best_delta[fresh]
                sel = fresh[better]
                best_delta[sel] = d1[better]
                best_i[sel] = a
                best_j[sel] = k

                d2 = D[fresh, k] + D[fresh, b] - D[k, b]
                better = d2 < best_delta[fresh]
                sel = fresh[better]
                best_delta[sel] = d2[better]
                best_i[sel] = k
                best_j[sel] = b

            # Vértices cuja melhor aresta foi quebrada: recalcular tudo.
            if stale.size > 0:
                t = np.asarray(tour, dtype=np.int64)
                nxt = np.roll(t, -1)
                base = D[t, nxt]
                M = D[np.ix_(stale, t)] + D[np.ix_(stale, nxt)] - base[None, :]
                idx = np.argmin(M, axis=1)
                best_delta[stale] = M[np.arange(stale.size), idx]
                best_i[stale] = t[idx]
                best_j[stale] = nxt[idx]

    # Rotaciona o ciclo para começar em start_node.
    s = tour.index(start_node)
    tour = tour[s:] + tour[:s]

    solution = Solution(tour)
    solution.compute_cost(D)
    return solution
