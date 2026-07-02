"""Ramer-Douglas-Peucker corner detector for L-shape seed.

Given an ordered set of 2-D scan points (as returned by
ScanResult.ordered_hit_points()), finds the index of the point with the
maximum perpendicular distance from the chord connecting the first and last
points.  That index approximates the L-shape corner (vértice de la L).

References
----------
Cellina et al. (arXiv 2025) — multi-hypothesis L-shape fitting
Zhang et al. CISRAM 2015   — Search-Based L-shape fitting
"""
from __future__ import annotations

import math
import numpy as np


def _perp_dist_to_chord(
    pts: np.ndarray,  # (N, 2)
    i0: int,
    i1: int,
) -> np.ndarray:  # (N,)
    """Perpendicular distances of pts[i0:i1+1] to the chord pts[i0]→pts[i1]."""
    a = pts[i1] - pts[i0]          # chord direction
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return np.zeros(i1 - i0 + 1)
    a_hat = a / norm
    diff = pts[i0 : i1 + 1] - pts[i0]   # (M, 2)
    proj = diff @ a_hat                   # scalar projection
    perp = diff - np.outer(proj, a_hat)   # (M, 2) perpendicular component
    return np.linalg.norm(perp, axis=1)


def rdp_corner(pts: np.ndarray, eps: float = 1e-4) -> int:
    """Return the index of the L-corner (max perpendicular distance to chord).

    Parameters
    ----------
    pts : (N, 2) ordered scan points (angular order preserved)
    eps : minimum distance to consider a real corner (metres)

    Returns
    -------
    corner_idx : int  index into pts of the estimated L-corner
    """
    if len(pts) < 3:
        return len(pts) // 2

    dists = _perp_dist_to_chord(pts, 0, len(pts) - 1)
    idx = int(np.argmax(dists))
    if dists[idx] < eps:
        # flat scan — return midpoint
        return len(pts) // 2
    return idx


def initial_theta(pts: np.ndarray, corner_idx: int) -> float:
    """Estimate initial orientation θ₀ (rad) from the dominant arm of the L.

    The dominant arm is the longer of the two sub-sequences split at corner_idx.
    θ₀ is the angle of the dominant arm direction.
    """
    if corner_idx <= 0 or corner_idx >= len(pts) - 1:
        corner_idx = max(1, min(corner_idx, len(pts) - 2))

    arm_a = pts[:corner_idx + 1]         # from start → corner
    arm_b = pts[corner_idx:]             # from corner → end

    def arm_angle(arm: np.ndarray) -> float:
        if len(arm) < 2:
            return 0.0
        d = arm[-1] - arm[0]
        return float(math.atan2(d[1], d[0]))

    len_a = float(np.linalg.norm(pts[corner_idx] - pts[0]))
    len_b = float(np.linalg.norm(pts[-1] - pts[corner_idx]))

    if len_a >= len_b:
        return arm_angle(arm_a)
    return arm_angle(arm_b)
