"""Emparelhamento perfeito de peso mínimo entre vértices de grau ímpar.

Usa networkx (algoritmo blossom). Instalação: pip install networkx
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import networkx as nx
from networkx.algorithms.matching import min_weight_matching


def min_weight_perfect_matching(
    instance, odd_vertices: Iterable[int]
) -> List[Tuple[int, int, int]]:
    """Emparelhamento perfeito de custo mínimo entre `odd_vertices`.

    O subgrafo induzido pelos vértices dados é completo, com peso de cada
    aresta (u, v) igual a instance.distance_matrix[u, v].

    Args:
        instance: objeto com `n` e `distance_matrix`.
        odd_vertices: vértices de grau ímpar (quantidade par, sem repetição).

    Returns:
        Lista de arestas (u, v, peso) com u < v, ordenada, onde cada vértice
        de `odd_vertices` aparece exatamente uma vez.

    Raises:
        ValueError: entrada inválida (repetidos, fora do intervalo ou
            quantidade ímpar) ou emparelhamento não perfeito.
    """
    D = instance.distance_matrix
    vertices = [int(v) for v in odd_vertices]

    if len(set(vertices)) != len(vertices):
        raise ValueError("odd_vertices contém vértices repetidos.")
    if any(v < 0 or v >= instance.n for v in vertices):
        raise ValueError(f"odd_vertices deve conter índices em [0, {instance.n - 1}].")
    if len(vertices) % 2 != 0:
        raise ValueError("odd_vertices deve ter quantidade par de vértices.")
    if not vertices:
        return []

    G = nx.Graph()
    G.add_nodes_from(vertices)
    for i, u in enumerate(vertices):
        for v in vertices[i + 1:]:
            G.add_edge(u, v, weight=int(D[u, v]))

    # networkx >= 3.0 sempre devolve emparelhamento de cardinalidade máxima
    # (perfeito, num grafo completo de ordem par); versões antigas precisam
    # de maxcardinality=True.
    try:
        matching = min_weight_matching(G, weight="weight")
    except TypeError:
        matching = min_weight_matching(G, maxcardinality=True, weight="weight")

    if 2 * len(matching) != len(vertices):
        raise ValueError("O emparelhamento obtido não é perfeito.")

    edges = []
    for u, v in matching:
        a, b = (u, v) if u < v else (v, u)
        edges.append((int(a), int(b), int(D[a, b])))
    edges.sort()
    return edges
