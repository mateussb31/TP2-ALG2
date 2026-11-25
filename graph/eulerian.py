"""Circuito euleriano em multigrafo (algoritmo de Hierholzer)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def find_eulerian_circuit(
    num_vertices: int,
    edges: Sequence[Tuple[int, int]],
    start: Optional[int] = None,
) -> List[int]:
    """Encontra um circuito euleriano com Hierholzer (versão iterativa).

    Args:
        num_vertices: número de vértices (rótulos 0..num_vertices-1).
        edges: arestas (u, v) do multigrafo não direcionado; arestas
            repetidas (paralelas) e laços são permitidos. Só os dois
            primeiros elementos de cada tupla são usados, então tuplas
            (u, v, peso) também funcionam.
        start: vértice inicial; por padrão, o primeiro vértice com aresta.

    Returns:
        Sequência de vértices do circuito, com len(edges) + 1 elementos e
        primeiro == último. Lista vazia se não houver arestas e `start` for
        None; [start] se não houver arestas e `start` for dado.

    Raises:
        ValueError: vértice fora do intervalo, algum grau ímpar, `start`
            sem arestas ou arestas em mais de uma componente conexa.
    """
    m = len(edges)
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_vertices)]
    degree = [0] * num_vertices

    for eid, e in enumerate(edges):
        u, v = int(e[0]), int(e[1])
        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(f"Aresta {e} com vértice fora de [0, {num_vertices - 1}].")
        adj[u].append((v, eid))
        adj[v].append((u, eid))  # laço (u == v) entra duas vezes: grau +2
        degree[u] += 1
        degree[v] += 1

    odd = [v for v in range(num_vertices) if degree[v] % 2 != 0]
    if odd:
        raise ValueError(f"Grafo não euleriano: vértices de grau ímpar {odd[:10]}.")

    if m == 0:
        return [] if start is None else [start]

    if start is None:
        start = next(v for v in range(num_vertices) if degree[v] > 0)
    elif not (0 <= start < num_vertices) or degree[start] == 0:
        raise ValueError(f"start={start} inválido ou sem arestas incidentes.")

    used = [False] * m
    ptr = [0] * num_vertices
    stack = [start]
    circuit: List[int] = []

    while stack:
        u = stack[-1]
        # Pula arestas já usadas (cada aresta aparece em duas listas).
        while ptr[u] < len(adj[u]) and used[adj[u][ptr[u]][1]]:
            ptr[u] += 1

        if ptr[u] < len(adj[u]):
            v, eid = adj[u][ptr[u]]
            used[eid] = True
            ptr[u] += 1
            stack.append(v)
        else:
            circuit.append(stack.pop())

    if len(circuit) != m + 1:
        raise ValueError("Grafo não conexo: nem todas as arestas foram percorridas.")

    circuit.reverse()
    return circuit
