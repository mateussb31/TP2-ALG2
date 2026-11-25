"""Instância de TSP lida de arquivos .tsp da TSPLIB."""

from __future__ import annotations

import numpy as np

from .distance import compute_distance_matrix


class TSPInstance:
    """Instância de TSP com coordenadas e matriz de distâncias.

    Atributos:
        name: nome da instância.
        n: número de cidades.
        coordinates: array (n, 2) de float64.
        distance_matrix: array (n, n) de inteiros (EUC_2D).
    """

    def __init__(self, name: str, coordinates: np.ndarray):
        coords = np.asarray(coordinates, dtype=np.float64)
        self.name = name
        self.n = int(coords.shape[0])
        self.coordinates = coords
        self.distance_matrix = compute_distance_matrix(coords)

    @classmethod
    def from_file(cls, filepath: str) -> "TSPInstance":
        """Faz o parse de um arquivo .tsp (EDGE_WEIGHT_TYPE: EUC_2D)."""
        name = None
        dimension = None
        edge_weight_type = None
        coords = []
        in_coords = False

        with open(filepath, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                if line == "EOF":
                    break

                if in_coords:
                    parts = line.split()
                    # Qualquer linha que não seja "id x y" encerra a seção.
                    if len(parts) < 3 or not _is_number(parts[0]):
                        in_coords = False
                    else:
                        coords.append((float(parts[1]), float(parts[2])))
                        continue

                if line.startswith("NODE_COORD_SECTION"):
                    in_coords = True
                    continue

                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().upper()
                    value = value.strip()
                    if key == "NAME":
                        name = value
                    elif key == "DIMENSION":
                        dimension = int(value)
                    elif key == "EDGE_WEIGHT_TYPE":
                        edge_weight_type = value.upper()

        if edge_weight_type != "EUC_2D":
            raise ValueError(
                f"EDGE_WEIGHT_TYPE não suportado: {edge_weight_type!r} "
                "(apenas EUC_2D)"
            )
        if dimension is None:
            raise ValueError("Campo DIMENSION ausente no arquivo.")
        if not coords:
            raise ValueError("Seção NODE_COORD_SECTION ausente ou vazia.")
        if len(coords) != dimension:
            raise ValueError(
                f"DIMENSION={dimension}, mas {len(coords)} coordenadas lidas."
            )

        return cls(name or "unnamed", np.array(coords, dtype=np.float64))

    def __repr__(self) -> str:
        return f"TSPInstance(name={self.name!r}, n={self.n})"


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False
