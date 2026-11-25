"""Algoritmo de Christofides (1,5-aproximação) para o TSP métrico.

Passos:
  1. Constrói a MST com Prim (módulo mst).
  2. Identifica os vértices de grau ímpar na MST (quantidade sempre par).
  3. Calcula o emparelhamento perfeito de peso mínimo entre eles
     (módulo matching).
  4. Une as arestas da MST com as do emparelhamento: o multigrafo resultante
     é conexo e tem todos os graus pares, logo é euleriano.
  5. Extrai um circuito euleriano com Hierholzer (módulo eulerian).
  6. Atalho (shortcutting): mantém só a primeira ocorrência de cada vértice.
  7. Fecha o ciclo (implícito em Solution) e calcula o custo exato via
     instance.distance_matrix.

Garantia: com desigualdade triangular e emparelhamento ótimo, custo <= 1,5 *
ótimo. O arredondamento EUC_2D pode violá-la por uma unidade em casos raros.
Se o emparelhamento for heurístico, a garantia deixa de valer.
"""

from __future__ import annotations

try:
    from graph.eulerian import find_eulerian_circuit
    from graph.matching import min_weight_perfect_matching
    from graph.mst import minimum_spanning_tree
    from src.tsp.solution import Solution
except ImportError:  # execução fora do pacote / layout alternativo
    from graph.eulerian import find_eulerian_circuit
    from graph.matching import min_weight_perfect_matching
    from graph.mst import minimum_spanning_tree
    from solution import Solution


def solve(instance, start_node: int = 0) -> Solution:
    """Resolve o TSP pelo algoritmo de Christofides.

    Args:
        instance: objeto com `n` e `distance_matrix`.
        start_node: vértice em que o tour começa (e raiz da MST).

    Returns:
        Solution com tour válido (começando em start_node) e custo exato.
    """
    n = int(instance.n)
    D = instance.distance_matrix

    if not (0 <= start_node < n):
        raise ValueError(f"start_node deve estar em [0, {n - 1}]; recebido {start_node}")

    # 1. MST
    mst = minimum_spanning_tree(D, start_node)

    # 2. Vértices de grau ímpar na MST
    degree = [0] * n
    for u, v, _ in mst.edges:
        degree[u] += 1
        degree[v] += 1
    odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]

    # 3. Emparelhamento perfeito de peso mínimo entre os vértices ímpares
    matching = min_weight_perfect_matching(instance, odd_vertices)

    # 4. Multigrafo euleriano = arestas da MST + arestas do emparelhamento
    multigraph = [(u, v) for u, v, _ in mst.edges]
    multigraph.extend((u, v) for u, v, _ in matching)

    # 5. Circuito euleriano começando em start_node
    circuit = find_eulerian_circuit(n, multigraph, start=start_node)

    # 6. Atalho: mantém só a primeira ocorrência de cada vértice
    seen = [False] * n
    tour = []
    for v in circuit:
        if not seen[v]:
            seen[v] = True
            tour.append(v)

    # 7. O ciclo fecha implicitamente (último -> primeiro)
    solution = Solution(tour)
    solution.compute_cost(D)
    return solution
