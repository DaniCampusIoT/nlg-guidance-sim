"""
Stage-2 Gauss-Newton / Levenberg-Marquardt refinement with fixed priors.

References
----------
* "Vehicle Speed Estimation Using Infrastructure-Mounted LiDAR via
   Rectangle Edge Matching", MDPI 2026  -- residual formulation.
* "LiDAR-Based Vehicle Detection and Tracking for Autonomous Racing",
   arXiv 2025                           -- fixed-dimension state vector.

The state vector is x = [xc, yc, theta] (3-DOF only).
Width W and Length L are *constants* (geometric priors from the preset).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class RefinedPose:
    """Output of the LM refinement step."""
    xc: float
    yc: float
    theta_rad: float
    width_m: float
    length_m: float
    rmse: float
    n_iter: int
    converged: bool
    cost_init: float
    cost_final: float


def _rect_residuals(
    state: np.ndarray,
    points: np.ndarray,
    half_w: float,
    half_l: float,
) -> np.ndarray:
    """Compute signed-distance residuals for all hit points.

    Each residual = signed distance from point to nearest side of the
    rigid rectangle (xc, yc, theta) with fixed half_w x half_l.

    Parameters
    ----------
    state   : [xc, yc, theta]
    points  : (N, 2)
    half_w  : W/2 (fixed)
    half_l  : L/2 (fixed)
    """
    xc, yc, theta = state
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Translate + rotate into rectangle frame
    dx = points[:, 0] - xc
    dy = points[:, 1] - yc
    u =  cos_t * dx + sin_t * dy   # longitudinal
    v = -sin_t * dx + cos_t * dy   # lateral

    # Closest side distance (signed, negative = inside)
    su = np.maximum(np.abs(u) - half_l, 0.0) * np.sign(u)
    sv = np.maximum(np.abs(v) - half_w, 0.0) * np.sign(v)

    # For points on the edge or inside, use the minimum wall distance
    inside = (np.abs(u) <= half_l) & (np.abs(v) <= half_w)
    wall_u = half_l - np.abs(u)
    wall_v = half_w - np.abs(v)
    min_wall = np.minimum(wall_u, wall_v)

    residuals = np.where(
        inside,
        -min_wall,                          # inside: negative residual
        np.sqrt(su**2 + sv**2),             # outside: Euclidean distance
    )
    return residuals


def _safe_n_iter(result) -> int:
    """Return iteration count from a least_squares result.

    ``result.njev`` (Jacobian evaluations) is ``None`` when
    ``method='lm'`` because scipy's MINPACK wrapper does not expose it.
    Fall back to ``result.nfev`` (function evaluations), which is always
    populated, and further fall back to 0 if even that is None.
    """
    njev = getattr(result, "njev", None)
    if njev is not None:
        return int(njev)
    nfev = getattr(result, "nfev", None)
    return int(nfev) if nfev is not None else 0


def refine_pose(
    xc0: float,
    yc0: float,
    theta0: float,
    width_m: float,
    length_m: float,
    points: np.ndarray,
    method: str = "lm",
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    max_nfev: int = 200,
) -> RefinedPose:
    """Refine NLG pose with Levenberg-Marquardt (default) or trf.

    Parameters
    ----------
    xc0, yc0, theta0 : initial state from Stage-1
    width_m, length_m: fixed priors (W, L)
    points           : (N, 2) LiDAR hit points
    method           : 'lm' or 'trf'
    """
    if len(points) < 4:
        return RefinedPose(
            xc=xc0, yc=yc0, theta_rad=theta0,
            width_m=width_m, length_m=length_m,
            rmse=np.inf, n_iter=0, converged=False,
            cost_init=np.inf, cost_final=np.inf,
        )

    half_w = width_m / 2.0
    half_l = length_m / 2.0
    x0 = np.array([xc0, yc0, theta0], dtype=float)

    # Evaluate initial cost
    r0 = _rect_residuals(x0, points, half_w, half_l)
    cost_init = float((r0**2).sum())

    result = least_squares(
        fun=_rect_residuals,
        x0=x0,
        args=(points, half_w, half_l),
        method=method,
        ftol=ftol,
        xtol=xtol,
        max_nfev=max_nfev,
    )

    xc, yc, theta = result.x
    theta = float(math.atan2(math.sin(theta), math.cos(theta)))
    rmse = float(np.sqrt((result.fun**2).mean()))
    cost_final = float((result.fun**2).sum())

    return RefinedPose(
        xc=float(xc), yc=float(yc), theta_rad=theta,
        width_m=width_m, length_m=length_m,
        rmse=rmse, n_iter=_safe_n_iter(result),
        converged=bool(result.success),
        cost_init=cost_init, cost_final=cost_final,
    )
