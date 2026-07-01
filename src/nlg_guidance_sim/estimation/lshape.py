"""L-shape fitting for ordered 2-D LiDAR point clouds.

Implements the efficient search-based method from:
  Xiao Zhang et al., "Efficient L-shape fitting of laser scanner data
  for vehicle pose estimation", CISRAM 2015.

For the NLG guidance application we fit an L (or degenerate line)
to the hit points returned by the synthetic RPLidar and extract:
  * Y   — lateral offset of the NLG midpoint from the guide axis [m]
  * psi — heading angle of the NLG axle w.r.t. the guide axis [rad]

Fallback: if too few hit points are available (< MIN_POINTS) the
estimator returns a centroid-only result flagged as low-confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_POINTS = 8          # below this we fall back to centroid
N_THETA = 180           # angular resolution for the search grid


@dataclass
class LShapeFitResult:
    Y_m: float                    # estimated lateral offset [m]
    psi_rad: float                # estimated heading [rad]
    psi_deg: float                # same in degrees
    corner_xy: np.ndarray         # estimated L-corner in world frame
    seg1_pts: np.ndarray          # points assigned to segment 1
    seg2_pts: np.ndarray          # points assigned to segment 2
    rmse_m: float                 # fit residual [m]
    confidence: float             # 0..1  (1 = full L-fit, <0.5 = fallback)
    method: str                   # 'lshape' | 'line' | 'centroid'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _line_residuals(pts: np.ndarray, theta: float) -> np.ndarray:
    """Perpendicular distances from points to a line with normal (cos θ, sin θ)."""
    n = np.array([math.cos(theta), math.sin(theta)])
    return pts @ n


def _fit_line_to_pts(pts: np.ndarray) -> tuple[float, float]:
    """Return (theta, rho) of the best-fit line using SVD.
    The line equation is  x*cos(θ) + y*sin(θ) = ρ.
    """
    if len(pts) < 2:
        return 0.0, float(np.mean(pts[:, 0])) if len(pts) else 0.0
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    tangent = Vt[0]                          # principal direction
    normal = np.array([-tangent[1], tangent[0]])
    theta = math.atan2(normal[1], normal[0])
    rho = float(centroid @ normal)
    return theta, rho


def _split_at_corner(pts: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Project points onto direction θ and find the split that minimises
    total perpendicular-distance variance on both sides (L-search criterion).
    Returns (seg1, seg2, corner_param).
    """
    t = np.array([math.cos(theta), math.sin(theta)])
    params = pts @ t
    order = np.argsort(params)
    pts_sorted = pts[order]
    params_sorted = params[order]
    n = len(pts_sorted)

    best_cost = np.inf
    best_k = n // 2

    for k in range(2, n - 2):
        seg1 = pts_sorted[:k]
        seg2 = pts_sorted[k:]
        _, r1 = _fit_line_to_pts(seg1)
        _, r2 = _fit_line_to_pts(seg2)
        res1 = _line_residuals(seg1, theta) - r1
        res2 = _line_residuals(seg2, theta) - r2
        cost = float(np.sum(res1 ** 2) + np.sum(res2 ** 2))
        if cost < best_cost:
            best_cost = cost
            best_k = k

    return pts_sorted[:best_k], pts_sorted[best_k:], float(params_sorted[best_k])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_lshape(hit_pts: np.ndarray) -> LShapeFitResult:
    """Fit an L-shape to *hit_pts* (N×2 array, world frame, ordered by angle).

    Strategy
    --------
    1. Search over N_THETA candidate orientations θ in [0, π).
    2. For each θ, split the sorted projection into two arms and compute
       the total perpendicular residual (CISRAM 2015 cost function).
    3. Pick the θ with minimum cost → defines the L orientation.
    4. Fit lines to each arm and intersect them → corner estimate.
    5. Derive Y and ψ from the corner and arm directions.
    """
    if len(hit_pts) < MIN_POINTS:
        return _fallback_centroid(hit_pts)

    thetas = np.linspace(0.0, math.pi, N_THETA, endpoint=False)
    best_cost = np.inf
    best_theta = 0.0

    for theta in thetas:
        t = np.array([math.cos(theta), math.sin(theta)])
        params = hit_pts @ t
        order = np.argsort(params)
        pts_s = hit_pts[order]
        n = len(pts_s)
        costs = []
        for k in range(2, n - 2):
            s1, s2 = pts_s[:k], pts_s[k:]
            _, r1 = _fit_line_to_pts(s1)
            _, r2 = _fit_line_to_pts(s2)
            res1 = _line_residuals(s1, theta) - r1
            res2 = _line_residuals(s2, theta) - r2
            costs.append(float(np.sum(res1 ** 2) + np.sum(res2 ** 2)))
        if costs:
            c = min(costs)
            if c < best_cost:
                best_cost = c
                best_theta = theta

    seg1, seg2, _ = _split_at_corner(hit_pts, best_theta)

    if len(seg1) < 2 or len(seg2) < 2:
        return _fallback_line(hit_pts)

    theta1, rho1 = _fit_line_to_pts(seg1)
    theta2, rho2 = _fit_line_to_pts(seg2)

    # Intersect the two lines to find the L-corner
    A = np.array([
        [math.cos(theta1), math.sin(theta1)],
        [math.cos(theta2), math.sin(theta2)],
    ])
    b = np.array([rho1, rho2])
    try:
        corner = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return _fallback_line(hit_pts)

    # psi = angle of the axle arm (the arm more perpendicular to X axis)
    # We pick the arm whose normal is closer to the Y direction.
    psi_candidate1 = theta1 - math.pi / 2.0
    psi_candidate2 = theta2 - math.pi / 2.0
    # Normalise to (-pi/2, pi/2)
    psi_rad = _normalise_angle(psi_candidate1)
    if abs(math.sin(theta2)) > abs(math.sin(theta1)):
        psi_rad = _normalise_angle(psi_candidate2)

    Y_m = float(corner[1])   # Y coordinate of corner = lateral offset estimate

    # RMSE over all points
    all_res = np.concatenate([
        _line_residuals(seg1, theta1) - rho1,
        _line_residuals(seg2, theta2) - rho2,
    ])
    rmse = float(np.sqrt(np.mean(all_res ** 2)))
    confidence = max(0.0, min(1.0, 1.0 - rmse / 0.05))   # normalised to 5 cm

    return LShapeFitResult(
        Y_m=Y_m,
        psi_rad=psi_rad,
        psi_deg=math.degrees(psi_rad),
        corner_xy=corner,
        seg1_pts=seg1,
        seg2_pts=seg2,
        rmse_m=rmse,
        confidence=confidence,
        method="lshape",
    )


def estimate_pose(hit_pts: np.ndarray) -> LShapeFitResult:
    """Convenience wrapper — alias for fit_lshape."""
    return fit_lshape(hit_pts)


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def _fallback_line(pts: np.ndarray) -> LShapeFitResult:
    theta, rho = _fit_line_to_pts(pts)
    psi_rad = _normalise_angle(theta - math.pi / 2.0)
    centroid = pts.mean(axis=0)
    res = _line_residuals(pts, theta) - rho
    rmse = float(np.sqrt(np.mean(res ** 2)))
    return LShapeFitResult(
        Y_m=float(centroid[1]),
        psi_rad=psi_rad,
        psi_deg=math.degrees(psi_rad),
        corner_xy=centroid,
        seg1_pts=pts,
        seg2_pts=np.empty((0, 2)),
        rmse_m=rmse,
        confidence=0.35,
        method="line",
    )


def _fallback_centroid(pts: np.ndarray) -> LShapeFitResult:
    if len(pts) == 0:
        centroid = np.zeros(2)
    else:
        centroid = pts.mean(axis=0)
    return LShapeFitResult(
        Y_m=float(centroid[1]) if len(pts) else 0.0,
        psi_rad=0.0,
        psi_deg=0.0,
        corner_xy=centroid,
        seg1_pts=pts,
        seg2_pts=np.empty((0, 2)),
        rmse_m=np.inf,
        confidence=0.0,
        method="centroid",
    )


def _normalise_angle(a: float) -> float:
    """Wrap angle to (-pi/2, pi/2)."""
    while a > math.pi / 2:
        a -= math.pi
    while a < -math.pi / 2:
        a += math.pi
    return a
