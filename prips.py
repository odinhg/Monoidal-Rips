import phat
import numpy as np
import networkx as nx
from scipy.spatial import distance
from operator import itemgetter
from itertools import product, chain
from typing import TypeAlias

Simplex: TypeAlias = tuple[int]
Filtration: TypeAlias = list[tuple[tuple[int], float]]
BoundaryMatrix: TypeAlias = list[tuple[int, list[int]]]
PersistenceDiagram: TypeAlias = list[tuple[int, tuple[float, float]]]


def f(s: Simplex, D: np.ndarray, p: float) -> float:
    """
    Compute p-Rips filtration value for a simplex s based on the distance matrix D.
    """
    if len(s) > 1:
        values = [D[t[0], t[1]] for t in zip(s, s[1:])]
        if p == np.inf:
            return max(values)
        return sum([v**p for v in values]) ** (1 / p)
    return D[s[0], s[0]]


def is_degenerate(s: Simplex) -> bool:
    """
    Checks if a simplex is degenerate (has adjacent elements that are equal).
    """
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            return True
    return False


def get_faces(s: Simplex, return_degerate: bool = False) -> list[Simplex]:
    """
    Boundary operator. Return (non-degenerate) faces of a simplex.
    """
    faces = [s[:k] + s[k + 1 :] for k in range(len(s))]
    if return_degerate:
        return faces
    return [face for face in faces if not is_degenerate(face)]


def filtration_from_array(
    X: np.ndarray, p: float = 1.0, q: float = 2.0, max_dim: int = 2
) -> Filtration:
    """
    Compute p-Rips filtration from a point cloud X. Use Lq norm to compute distance matrix. Returns a list of tuples on the form (simplex: tuple[int], filtration_value: float) sorted by filtration value increasingly.
    """
    D = distance.squareform(distance.pdist(X, metric="minkowski", p=q))
    indices = list(range(X.shape[0]))
    filtration_values = {}

    # Treat 0-, 1- and 2-simplices as special cases for speed.
    for i in indices:
        filtration_values[(i,)] = 0

    for s in product(indices, repeat=2):
        i, j = s
        if i == j:
            continue
        filtration_values[s] = D[i, j]

    if max_dim > 1:
        for s in product(indices, repeat=3):
            i, j, k = s
            if i == j or j == k:
                continue
            if p == np.inf:
                v = max(D[i, j], D[j, k], D[i, k], max(D[i, j], D[j, k]))
            elif p == 1:
                v = max(D[i, j], D[j, k], D[i, k], D[i, j] + D[j, k])
            else:
                v = max(
                    D[i, j], D[j, k], D[i, k], (D[i, j] ** p + D[j, k] ** p) ** (1 / p)
                )
            filtration_values[s] = v

    if max_dim > 2:
        for dim in range(3, max_dim + 1):
            for s in product(indices, repeat=dim + 1):
                if is_degenerate(s):
                    continue
                v = max(
                    [
                        filtration_values.get(face, 0)
                        for face in get_faces(s, return_degerate=True)
                    ]
                    + [f(s, D, p)]
                )
                filtration_values[s] = v

    filtration = sorted(filtration_values.items(), key=itemgetter(1))

    return filtration


def filtration_from_graph(
    A: np.ndarray, weighted: bool = True, p: float = 1.0
) -> Filtration:
    """Construct p-Rips filtration from a (directed) graph given as an adjacency matrix A. If weighted=True, use diagonal elements as filtration values for vertices. Edges not present in the graph are indicated by infinity in the adjacency matrix."""
    indices = list(range(A.shape[0]))
    filtration_values = {}

    for i in indices:
        if weighted:
            filtration_values[(i,)] = A[i, i]
        else:
            filtration_values[(i,)] = 0

    for s in product(indices, repeat=2):
        i, j = s
        if i == j:
            continue
        filtration_value = max(A[i, i], A[j, j], A[i, j])
        if filtration_value < np.inf:
            filtration_values[s] = A[i, j]

    for s in product(indices, repeat=3):
        i, j, k = s
        if i == j or j == k:
            continue
        # Skip if boundary is not contained in the graph
        if (i, j) not in filtration_values:
            continue
        if (j, k) not in filtration_values:
            continue
        if i != k and (i, k) not in filtration_values:
            continue
        if p == np.inf:
            v = max(A[i, j], A[j, k], (A[i, k] if i != k else 0))
        else:
            v = max(
                A[i, j],
                A[j, k],
                (A[i, k] if i != k else 0),
                (A[i, j] ** p + A[j, k] ** p) ** (1 / p),
            )
        filtration_values[s] = v
    filtration = sorted(filtration_values.items(), key=itemgetter(1))

    return filtration


def compute_boundary_matrix(filtration: Filtration) -> BoundaryMatrix:
    """Construct boundary matrix from filtration. Returns boundary matrix as a sparse matrix compatible with the phat package."""
    index_dict = {
        s[0]: j for j, s in enumerate(filtration)
    }  # Lookup table simplex --> index

    B = []
    for s, _ in filtration:
        dim = len(s) - 1
        if dim == 2:
            i, j, k = s
            if i == k:
                B.append((dim, [index_dict[(j, k)], index_dict[(i, j)]]))
            else:
                B.append(
                    (dim, [index_dict[(j, k)], index_dict[(i, k)], index_dict[(i, j)]])
                )
        elif dim == 1:
            i, j = s
            B.append((dim, [index_dict[(j,)], index_dict[(i,)]]))
        elif dim == 0:
            B.append((dim, []))
        else:
            face_indices = [index_dict[face] for face in get_faces(s)]
            B.append((dim, face_indices))

    return B


def compute_persistence(
    filtration: list[tuple[tuple[int], float]],
    min_persistence: float = 10e-4,
    max_dim: int = 1,
) -> PersistenceDiagram:
    """
    Compute persistent homology of given filtration using phat.
    """
    B = compute_boundary_matrix(filtration)
    boundary_matrix = phat.boundary_matrix(
        representation=phat.representations.vector_vector,
        columns=B,
    )
    pairs = boundary_matrix.compute_persistence_pairs(
        reduction=phat.reductions.standard_reduction
    )
    pairs.sort()

    diagram = []
    for birth_idx, death_idx in pairs:
        birth_simplex, birth_time = filtration[birth_idx]
        death_simplex, death_time = filtration[death_idx]
        dim = len(birth_simplex) - 1
        persistence = death_time - birth_time
        if persistence > min_persistence:
            diagram.append((dim, (birth_time, death_time)))

    # Add unpaired pairs (i.e., pairs on the form (birth, +inf))
    paired_simplices = set(chain.from_iterable(pairs))
    for i in range(len(filtration)):
        if i not in paired_simplices:
            # Simplex i corresponds to a zero column and is not the lowest 1 in any (non-zero) column
            birth_simplex, birth_time = filtration[i]
            dim = len(birth_simplex) - 1
            if dim > max_dim:
                continue
            diagram.append((dim, (birth_time, np.inf)))

    return diagram
