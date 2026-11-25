"""Solução (tour) para o TSP."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


class Solution:
    """Tour do TSP e seu custo.

    Atributos:
        tour: lista de índices de 0 a n-1 (ciclo implícito: o último
            vértice volta ao primeiro).
        cost: custo total do ciclo (None até ser calculado).
    """

    def __init__(
        self,
        tour: List[int],
        cost: Optional[Union[int, float]] = None,
    ):
        self.tour = list(tour)
        self.cost = cost

    def is_valid(self, n: int) -> bool:
        """True se o tour é um ciclo hamiltoniano sobre n vértices.

        Ou seja, contém exatamente os índices 0..n-1, cada um uma vez.
        """
        if n <= 0 or len(self.tour) != n:
            return False
        seen = [False] * n
        for v in self.tour:
            if not isinstance(v, (int, np.integer)) or v < 0 or v >= n:
                return False
            if seen[v]:
                return False
            seen[v] = True
        return True

    def compute_cost(self, distance_matrix: np.ndarray) -> Union[int, float]:
        """Calcula, armazena em self.cost e retorna o custo do ciclo."""
        t = np.asarray(self.tour, dtype=np.int64)
        if t.size == 0:
            self.cost = 0
            return self.cost
        nxt = np.roll(t, -1)
        total = distance_matrix[t, nxt].sum()
        self.cost = int(total) if np.issubdtype(
            np.asarray(distance_matrix).dtype, np.integer
        ) else float(total)
        return self.cost

    def __repr__(self) -> str:
        return f"Solution(n={len(self.tour)}, cost={self.cost})"
