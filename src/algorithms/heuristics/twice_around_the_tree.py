"""Heurística Twice-around-the-tree (2-aproximação) para o TSP.

Passos:
  1. Constrói a MST com Prim (módulo mst).
  2. Duplica cada aresta da MST: o multigrafo resultante tem todos os graus
     pares e é conexo, logo é euleriano.
  3. Encontra um circuito euleriano com Hierholzer (módulo eulerian).
  4. Atalho (shortcutting): percorre o circuito mantendo só a primeira
     ocorrência de cada vértice.
  5. Fecha o ciclo (o último vértice volta ao primeiro, implícito em
     Solution) e calcula o custo exato via instance.distance_matrix.

Garantia: com desigualdade triangular, custo <= 2 * ótimo. O arredondamento
EUC_2D da TSPLIB pode violá-la por uma unidade em casos raros.

Complexidade: O(n^2) para a MST sobre matriz densa, O(n) para o circuito e
o atalho (o multigrafo tem 2(n-1) arestas).
"""

from __future__ import annotations

try:
    from graph.eulerian import find_eulerian_circuit
    from graph.mst import minimum_spanning_tree
    from src.tsp.solution import Solution
except ImportError:  # execução fora do pacote / layout alternativo
    from graph.eulerian import find_eulerian_circuit
    from graph.mst import minimum_spanning_tree
    from solution import Solution



def solve(instance, start_node: int = 0) -> Solution:
    """Resolve o TSP pela heurística twice-around-the-tree.

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

    # 2. Duplica as arestas da MST (u, v) -> duas cópias paralelas
    doubled = [(u, v) for u, v, _ in mst.edges] * 2

    # 3. Circuito euleriano começando em start_node
    circuit = find_eulerian_circuit(n, doubled, start=start_node)

    # 4. Atalho: mantém só a primeira ocorrência de cada vértice
    seen = [False] * n
    tour = []
    for v in circuit:
        if not seen[v]:
            seen[v] = True
            tour.append(v)

    # 5. O ciclo fecha implicitamente (último -> primeiro)
    solution = Solution(tour)
    solution.compute_cost(D)
    return solution
