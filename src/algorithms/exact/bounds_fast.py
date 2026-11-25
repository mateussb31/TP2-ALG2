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

Nenhuma cópia da matriz inteira é feita por nó: `D` é usada por referência.

----------------------------------------------------------------------------
Otimização guiada por profiling da MST (o limite calculado NÃO muda)
----------------------------------------------------------------------------
O profiling do B&B aponta `mst_weight` como a subrotina dominante. Duas
mudanças, ambas exatas (mesmo valor de limite, logo mesmos nodes_expanded e
nodes_pruned; só o tempo muda):

(a) Compartilhar a MST entre irmãos (ganho estrutural).
    Os filhos de um nó (cur, U) são v em U; o filho v avalia a MST sobre
    {v} ∪ (U \\ {v}) ∪ {origem} = U ∪ {origem}. Esse conjunto NÃO depende de
    v: todos os filhos do mesmo nó pedem a MST do mesmo conjunto. Antes ela
    era recalculada |U| vezes; agora é calculada no máximo uma vez por nó
    (SharedMST), de forma preguiçosa, só se algum filho precisar dela.

(b) Prim com submatriz (ganho de constante).
    A submatriz dos vértices é extraída uma única vez (um gather em C) e o
    laço de Prim opera sobre ela, mascarando colunas já na árvore com `big`.
    Elimina, por iteração, o np.where com alocação, a indexação avançada
    D[idx[j], idx] e a máscara booleana.

Por que as decisões de poda continuam idênticas mesmo com o early-exit:
`cutoff` = melhor_custo - g. O limite é usado só em `g + limite >= melhor`,
isto é, `limite >= cutoff`. Uma soma parcial já >= cutoff implica MST >= cutoff
(arestas não negativas), então o resultado da comparação é o mesmo para a
soma parcial e para a MST completa. E como `cutoff` só diminui entre irmãos
(g cresce na ordem dos filhos e melhor_custo só cai), um valor parcial
calculado para um cutoff maior continua valendo para os menores; a condição
é verificada explicitamente em `shared_mst_weight`, sem depender dessa ordem.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

# Abaixo disso o limite barato já é (quase) exato; a MST não compensa.
_MST_MIN_UNVISITED = 3


class SharedMST:
    """MST de um conjunto de vértices, compartilhada entre os filhos de um nó.

    Atributos:
        vertices: vértices do conjunto (U ∪ {origem}).
        value: último valor calculado (None até o primeiro uso).
        cutoff: cutoff com que `value` foi calculado.
        exact: True se `value` é o peso completo da MST; False se é uma soma
            parcial já >= cutoff (válida como limite para qualquer cutoff
            menor ou igual).
    """

    __slots__ = ("vertices", "value", "cutoff", "exact")

    def __init__(self, vertices: List[int]):
        self.vertices = vertices
        self.value = None
        self.cutoff = float("inf")
        self.exact = False


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
            self._big = int(np.iinfo(D.dtype).max // 4)
        else:
            self._big = np.inf

    # ------------------------------------------------------------------
    # Limite 1: soma dos menores arcos de saída (inalterado)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Limite 2: MST (Prim sobre submatriz)
    # ------------------------------------------------------------------
    def mst_weight(self, vertices: Sequence[int], cutoff: float = float("inf")) -> float:
        """Peso da MST (Prim vetorizado) sobre `vertices`.

        Para assim que o peso parcial atinge `cutoff`: como as arestas
        restantes são não negativas, o limite final seria >= cutoff de qualquer
        forma (basta para podar). Sem `cutoff`, devolve o peso completo.

        Invariante do laço: `key[x]` é a menor distância de x à árvore se x
        está fora dela e `big` se x já está nela. A coluna de cada vértice
        que entra na árvore é mascarada com `big` em `sub`, então nenhuma
        atualização posterior consegue rebaixar `key` de um vértice dentro.
        """
        idx = np.asarray(vertices, dtype=np.intp)
        k = idx.size
        if k <= 1:
            return 0

        big = self._big
        sub = self.D[np.ix_(idx, idx)]  # um único gather; já é uma cópia (k, k)

        key = sub[0].copy()  # raiz = vertices[0]
        key[0] = big
        sub[:, 0] = big
        total = 0

        for _ in range(k - 1):
            j = int(key.argmin())
            total += key[j].item()
            if total >= cutoff:
                return total
            key[j] = big
            sub[:, j] = big
            np.minimum(key, sub[j], out=key)

        return total

    def shared_mst_weight(self, shared: SharedMST, cutoff: float) -> float:
        """MST do conjunto compartilhado, calculada no máximo uma vez por nó.

        Reaproveita o valor guardado se ele é exato, ou se é uma soma parcial
        calculada para um cutoff >= ao atual (então continua >= cutoff).
        """
        if shared.value is not None and (shared.exact or cutoff <= shared.cutoff):
            return shared.value
        w = self.mst_weight(shared.vertices, cutoff)
        shared.value = w
        shared.cutoff = cutoff
        shared.exact = w < cutoff  # sem early-exit => peso completo
        return w

    # ------------------------------------------------------------------
    # Limite combinado
    # ------------------------------------------------------------------
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

        Versão sem compartilhamento; mantida como referência e para usos
        isolados. A busca usa `remaining_bound_shared`.
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

    def remaining_bound_shared(
        self,
        cur: int,
        start: int,
        visited: Sequence[bool],
        unvisited: Sequence[int],
        cutoff: float,
        shared: SharedMST,
    ) -> float:
        """Igual a `remaining_bound`, mas a MST vem de `shared`.

        Pré-condição: `shared.vertices` é o conjunto {cur} ∪ unvisited ∪ {start}
        (na busca, U_pai ∪ {origem}, igual para todos os filhos do mesmo nó).
        """
        if not unvisited:
            return self.Dl[cur][start]

        lb = self.min_out_sum(cur, start, visited, unvisited)
        if lb >= cutoff or len(unvisited) < _MST_MIN_UNVISITED:
            return lb

        mst = self.shared_mst_weight(shared, cutoff)
        return mst if mst > lb else lb
