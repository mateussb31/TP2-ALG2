"""Limites inferiores admissíveis para o Branch-and-Bound do TSP.

Contexto: a busca fixa a cidade 0 como origem e constrói o tour por
prefixos (0, v1, ..., vk). Dado um prefixo que termina em `cur`, com
conjunto U de cidades ainda não visitadas, o restante do tour é um caminho
hamiltoniano de `cur` até a origem passando por todo U. Os dois limitantes
abaixo estimam por baixo o custo desse restante:

1. Soma dos menores arcos de saída (barato, O(|U|) a O(|U|^2)):
   - `cur` precisa sair para algum vértice de U;
   - cada u em U precisa sair para algum vértice de (U \\ {u}) ∪ {origem}.
   Os arcos de saída de vértices distintos são distintos, então a soma dos
   mínimos é um limite inferior válido.

2. Peso da MST sobre {cur} ∪ U ∪ {origem} (mais forte, mais caro):
   o caminho restante é um caminho hamiltoniano nesse conjunto, e todo
   caminho hamiltoniano é uma árvore geradora; logo custa >= MST.

O limite final é o máximo dos dois. Ambos são admissíveis, então podar
quando g + limite >= melhor_custo nunca descarta um tour estritamente melhor.

Nenhuma cópia da matriz é feita por nó: `D` é usada por referência e a MST
só indexa linhas (vetores de tamanho |S|), sem montar submatrizes.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

# Abaixo disso o limite barato já é (quase) exato; a MST não compensa.
_MST_MIN_UNVISITED = 3


class LowerBound:
    """Pré-computa estruturas fixas da instância e avalia limites por nó."""

    def __init__(self, distance_matrix: np.ndarray):
        D = np.asarray(distance_matrix)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"distance_matrix deve ser quadrada; recebido {D.shape}")

        self.n = int(D.shape[0])
        self.D = D
        # Lista de listas: acesso escalar bem mais rápido que D[i, j] do NumPy.
        self.Dl = D.tolist()
        # Vizinhos de cada vértice em ordem crescente de distância (sem o
        # próprio vértice). Serve para o limite barato e para ordenar filhos.
        self.neighbors: List[List[int]] = []
        for i in range(self.n):
            order = np.argsort(D[i], kind="stable").tolist()
            order.remove(i)
            self.neighbors.append(order)

        if np.issubdtype(D.dtype, np.integer):
            self._big = int(np.iinfo(np.int64).max // 4)
        else:
            self._big = np.inf

    def min_out_sum(
        self,
        cur: int,
        start: int,
        visited: Sequence[bool],
        unvisited: Sequence[int],
    ) -> float:
        """Limite 1: soma dos menores arcos de saída. Exige `unvisited` não vazio."""
        Dl = self.Dl
        neighbors = self.neighbors

        row_cur = Dl[cur]
        total = min(row_cur[u] for u in unvisited)  # saída de `cur`

        for u in unvisited:
            row = Dl[u]
            for w in neighbors[u]:
                # Sucessor válido de u: outro vértice não visitado ou a origem.
                if w == start or not visited[w]:
                    total += row[w]
                    break
        return total

    def mst_weight(self, vertices: Sequence[int], cutoff: float = float("inf")) -> float:
        """Limite 2: peso da MST (Prim vetorizado) sobre `vertices`.

        Para assim que o peso parcial atinge `cutoff`: como as arestas
        restantes são não negativas, o limite final seria >= cutoff de qualquer
        forma (basta para podar).
        """
        idx = np.asarray(vertices, dtype=np.intp)
        k = idx.size
        if k <= 1:
            return 0

        D = self.D
        big = self._big

        key = D[idx[0], idx]  # indexação avançada: já devolve cópia (k,)
        in_tree = np.zeros(k, dtype=bool)
        in_tree[0] = True
        total = 0

        for _ in range(k - 1):
            masked = np.where(in_tree, big, key)
            j = int(masked.argmin())
            total += masked[j].item()
            if total >= cutoff:
                return total
            in_tree[j] = True
            np.minimum(key, D[idx[j], idx], out=key)

        return total

    def remaining_bound(
        self,
        cur: int,
        start: int,
        visited: Sequence[bool],
        unvisited: Sequence[int],
        cutoff: float = float("inf"),
    ) -> float:
        """Limite inferior do custo restante (de `cur`, por todo `unvisited`, até `start`).

        `cutoff` é a folga até o melhor custo conhecido (melhor - g). Se o
        limite barato já atinge `cutoff`, devolve sem calcular a MST.
        Exige cur != start (vale para todo nó interno da busca).
        """
        if not unvisited:
            return self.Dl[cur][start]  # só falta fechar o ciclo: exato

        lb = self.min_out_sum(cur, start, visited, unvisited)
        if lb >= cutoff or len(unvisited) < _MST_MIN_UNVISITED:
            return lb

        vertices = [cur]
        vertices.extend(unvisited)
        vertices.append(start)
        mst = self.mst_weight(vertices, cutoff)
        return mst if mst > lb else lb
