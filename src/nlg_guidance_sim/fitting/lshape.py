"""
Search-Based L-shape fitting (coarse, Stage-1).

Based on:
  Zhang et al., "Efficient L-Shape Fitting for Vehicle Detection Using
  Laser Scanners", CISRAM 2015.

The algorithm sweeps theta in [0 deg, 90 deg) and selects the orientation
that minimises the sum of squared perpendicular distances to the closest
side of the rigid bounding rectangle (W x L fixed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class LShapeResult:
    """Output of the Search-Based L-shape fitter."""
    xc: float        # centre x [m]  (midpoint of visible cluster)
    yc: float        # centre y [m]
    theta_rad: float # heading / orientation [rad]
    width_m: float   # fixed prior width (W) [m]
    length_m: float  # fixed prior length (L) [m]
    cost: float      # minimised cost value
    n_points: int    # number of hit points used


class LShapeFit:
    """Search-Based L-shape fitter with fixed (prior) dimensions.

    Parameters
    ----------
    width_m  : known NLG width  W (track_width or tire_width_m)
    length_m : known NLG length L (tire_length_m)
    theta_step_deg : angular resolution for coarse sweep
    """

    def __init__(
        self,
        width_m: float,
        length_m: float,
        theta_step_deg: float = 0.5,
    ) -> None:
        self.width_m = width_m
        self.length_m = length_m
        self.theta_step_deg = theta_step_deg

    # -- public --------------------------------------------------------------

    def fit(self, points: np.ndarray) -> LShapeResult:
        """Fit L-shape to *points* (N, 2) array of LiDAR hits.

        Returns the best LShapeResult found by the angular sweep.
        """
        if len(points) < 3:
            cx, cy = points.mean(axis=0) if len(points) else (0.0, 0.0)
            return LShapeResult(
                xc=float(cx), yc=float(cy),
                theta_rad=0.0,
                width_m=self.width_m, length_m=self.length_m,
                cost=np.inf, n_points=len(points),
            )

        cx, cy = float(points[:, 0].mean()), float(points[:, 1].mean())

        # Coarse sweep
        thetas = np.deg2rad(
            np.arange(0.0, 90.0, self.theta_step_deg)
        )
        costs = np.array([self._cost(points, t) for t in thetas])
        best_i = int(np.argmin(costs))

        # Refine with scalar minimisation around best coarse angle
        lo = thetas[max(0, best_i - 1)]
        hi = thetas[min(len(thetas) - 1, best_i + 1)]
        res = minimize_scalar(
            lambda t: self._cost(points, t),
            bounds=(lo, hi),
            method="bounded",
        )
        theta_opt = float(res.x)
        cost_opt = float(res.fun)

        return LShapeResult(
            xc=cx, yc=cy,
            theta_rad=theta_opt,
            width_m=self.width_m, length_m=self.length_m,
            cost=cost_opt, n_points=len(points),
        )

    # -- private -------------------------------------------------------------

    def _cost(self, points: np.ndarray, theta: float) -> float:
        """Sum of squared min-distances from each point to the nearest
        side of the rigid rectangle parametrised by theta."""
        c = np.array([math.cos(theta), math.sin(theta)])
        s = np.array([-math.sin(theta), math.cos(theta)])
        half_l = self.length_m / 2.0
        half_w = self.width_m / 2.0

        # Project onto local axes
        lx = points @ c          # longitudinal
        ly = points @ s          # lateral

        # Distances to each of the four sides
        d_top    = lx - half_l   # > 0 if outside front
        d_bot    = -half_l - lx  # > 0 if outside back
        d_right  = ly - half_w   # > 0 if outside right
        d_left   = -half_w - ly  # > 0 if outside left

        # Signed distance: negative = inside, positive = outside
        dx = np.maximum(d_top, d_bot)
        dy = np.maximum(d_right, d_left)

        # Points inside the rect: distance to nearest edge
        dist_sq = np.where(
            (dx < 0) & (dy < 0),
            np.minimum(dx**2, dy**2),
            np.maximum(dx, 0.0)**2 + np.maximum(dy, 0.0)**2,
        )
        return float(dist_sq.sum())
