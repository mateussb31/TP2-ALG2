"""Calculo de árvore geradora mínima (minimum spanning tree -> MST) usando algoritmo de Prim.

Seleciona um vértice qualquer.
Escolhe a aresta de menor custo que liga um vértice já selecionado a um não selecionado.
Adicione o vértice selecionado a árvore geradora.
"""

import heapq
from dataclasses import dataclass
from typing import List, Tuple
 
import numpy as np
 
 
@dataclass
class MST:
    """Resultado do Prim.
 
    Atributos:
        edges: lista de (pai, filho, peso), com ints do Python.
        cost: peso total da árvore (int do Python).
        parent: lista com o pai de cada vértice (-1 na raiz).
    """
 
    edges: List[Tuple[int, int, int]]
    cost: int
    parent: List[int]
 
    def adjacency(self) -> List[List[int]]:
        """Lista de adjacência (não direcionada) da árvore."""
        adj = [[] for _ in self.parent]
        for u, v, _ in self.edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
 
 
def minimum_spanning_tree(distance_matrix: np.ndarray, start_node: int = 0) -> MST:
    """Calcula a MST pelo algoritmo de Prim (heap com chaves preguiçosas).
 
    Args:
        distance_matrix: matriz (n, n) simétrica de pesos inteiros.
        start_node: vértice raiz (não altera o custo, só a ordem das arestas).
 
    Returns:
        MST com as n-1 arestas, o custo total e o vetor de pais.
    """
    D = np.asarray(distance_matrix)
    n = D.shape[0]
    if not (0 <= start_node < n):
        raise ValueError(f"start_node deve estar em [0, {n - 1}]; recebido {start_node}")
 
    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
 
    key[start_node] = 0.0
    heap = [(0.0, start_node)]
 
    edges: List[Tuple[int, int, int]] = []
    cost = 0
 
    while heap:
        _, u = heapq.heappop(heap)
        if in_tree[u]:
            continue  # entrada obsoleta
        in_tree[u] = True
 
        p = int(parent[u])
        if p != -1:
            w = int(D[p, u])
            edges.append((p, int(u), w))
            cost += w
 
        # Relaxa as arestas de u para os vértices ainda fora da árvore.
        for v in np.flatnonzero(~in_tree & (D[u] < key)):
            key[v] = D[u, v]
            parent[v] = u
            heapq.heappush(heap, (float(D[u, v]), int(v)))
 
    return MST(edges=edges, cost=cost, parent=[int(p) for p in parent])
 
