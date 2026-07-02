"""Gauss-Newton / Levenberg-Marquardt solver for the rigid rectangle fit.

Wraps scipy.optimize.least_squares with method='lm' (Levenberg-Marquardt),
which is more robust to ill-conditioned Jacobians than plain Gauss-Newton
when the scan is partial (occlusions).

Reference: MDPI 2026 — Rectangle Edge Matching via Gauss-Newton.
Reference: Cellina et al. arXiv 2025 — Variance Minimisation L-shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from .rectangle_model import RigidRectangle


@dataclass
class FitResult:
    """Output of the GN/LM optimiser."""
    xc: float          # corner X (m)
    yc: float          # corner Y (m)
    theta: float       # heading  (rad)
    W: float           # width    (m)  — fixed from preset
    L: float           # length   (m)  — fixed from preset
    cost: float        # sum of squared residuals at solution
    rmse_m: float      # RMSE of point-to-edge distances (m)
    n_iter: int        # number of LM iterations
    converged: bool

    @property
    def Y_m(self) -> float:
        """Lateral offset: yc is approximately the lateral position of the corner.
        The NLG midpoint Y is yc + W/2 · cos(theta) for the heading direction.
        """
        return float(self.yc + (self.W / 2.0) * math.cos(self.theta))

    @property
    def psi_rad(self) -> float:
        return float(self.theta)

    @property
    def psi_deg(self) -> float:
        return float(math.degrees(self.theta))

    def center_xy(self) -> np.ndarray:
        """Geometric centre of the rectangle."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        h = np.array([c, s])
        n = np.array([-s, c])
        corner = np.array([self.xc, self.yc])
        return corner + (self.L / 2.0) * h + (self.W / 2.0) * n

    def rect_corners(self) -> np.ndarray:
        """Return (4, 2) array of the fitted rectangle corners."""
        rect = RigidRectangle(W=self.W, L=self.L)
        return rect.corners(np.array([self.xc, self.yc, self.theta]))


def run_lm(
    pts: np.ndarray,       # (N, 2) hit points
    rect: RigidRectangle,
    x0: np.ndarray,        # (3,) initial state [xc, yc, theta]
    max_iter: int = 50,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
) -> FitResult:
    """Run Levenberg-Marquardt optimisation and return FitResult."""

    def _residuals(state: np.ndarray) -> np.ndarray:
        return rect.residuals(state, pts)

    def _jac(state: np.ndarray) -> np.ndarray:
        return rect.jacobian(state, pts)

    result = least_squares(
        _residuals,
        x0,
        jac=_jac,
        method="lm",
        max_nfev=max_iter * 10,
        ftol=ftol,
        xtol=xtol,
    )

    residuals_final = _residuals(result.x)
    rmse = float(np.sqrt(np.mean(residuals_final ** 2)))
    xc, yc, theta = float(result.x[0]), float(result.x[1]), float(result.x[2])

    return FitResult(
        xc=xc,
        yc=yc,
        theta=theta,
        W=rect.W,
        L=rect.L,
        cost=float(result.cost),
        rmse_m=rmse,
        n_iter=int(result.nfev),
        converged=result.success,
    )
