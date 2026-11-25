"""Cálculo da matriz de distâncias EUC_2D da TSPLIB."""

import numpy as np


def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """Retorna a matriz (N, N) de distâncias inteiras EUC_2D.

    Cada entrada é int(sqrt(dx^2 + dy^2) + 0.5), conforme a TSPLIB.
    A matriz é simétrica e tem diagonal nula.

    Args:
        coordinates: array (N, 2) com as coordenadas 2D das cidades.

    Returns:
        Array (N, N) de dtype int64.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"coordinates deve ter formato (N, 2); recebido {coords.shape}"
        )

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    # Distâncias são não negativas, então floor(x + 0.5) == int(x + 0.5).
    matrix = np.floor(dist + 0.5).astype(np.int64)
    np.fill_diagonal(matrix, 0)
    return matrix
