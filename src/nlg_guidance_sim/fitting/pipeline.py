"""Two-stage fitting pipeline: RDP seed → Gauss-Newton / Levenberg-Marquardt.

Stage 1 (fast heuristic)
  Ramer-Douglas-Peucker finds the L-corner and computes θ₀ from the dominant arm.
  This seed prevents the LM optimiser from converging to the mirrored local minimum
  (the 'L-flip' problem identified in Zhang et al. CISRAM 2015).

Stage 2 (non-linear refinement)
  scipy.optimize.least_squares (method='lm') minimises the sum of squared
  point-to-nearest-edge distances, with W_real and L_real FIXED from the
  AircraftPreset dimensions.  Only (xc, yc, θ) are free.

References
----------
MDPI 2026  — Vehicle Speed Estimation via Rectangle Edge Matching (Gauss-Newton)
Cellina et al. arXiv 2025 — a-priori known vehicle size + Variance Minimisation seed
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .rdp import rdp_corner, initial_theta
from .rectangle_model import RigidRectangle
from .gauss_newton import FitResult, run_lm


@dataclass
class FittingResult:
    """Combined result from both pipeline stages."""
    seed_corner_idx: int          # RDP corner index in pts array
    seed_theta_rad: float         # θ₀ from RDP stage (rad)
    fit: FitResult                # LM optimisation output
    pts_used: np.ndarray          # (N, 2) points passed to the solver

    @property
    def Y_m(self) -> float:
        return self.fit.Y_m

    @property
    def psi_rad(self) -> float:
        return self.fit.psi_rad

    @property
    def psi_deg(self) -> float:
        return self.fit.psi_deg

    @property
    def rmse_m(self) -> float:
        return self.fit.rmse_m

    @property
    def converged(self) -> bool:
        return self.fit.converged


def run_fitting_pipeline(
    pts: np.ndarray,       # (N, 2) ordered hit points from ScanResult
    W_real: float,         # known wheel/NLG width  (m) — from AircraftPreset
    L_real: float,         # known wheel/NLG length (m) — from AircraftPreset
    min_pts: int = 6,
    max_iter: int = 60,
) -> FittingResult | None:
    """Run the full two-stage fitting pipeline.

    Returns None if there are fewer than min_pts hit points.
    """
    if len(pts) < min_pts:
        return None

    # ── Stage 1: RDP seed ────────────────────────────────────────────────────
    corner_idx = rdp_corner(pts, eps=1e-3)
    theta0     = initial_theta(pts, corner_idx)

    # Initial corner position = the RDP corner point
    xc0 = float(pts[corner_idx, 0])
    yc0 = float(pts[corner_idx, 1])
    x0  = np.array([xc0, yc0, theta0])

    # ── Stage 2: LM refinement with fixed W, L ───────────────────────────────
    rect = RigidRectangle(W=W_real, L=L_real)
    fit  = run_lm(pts, rect, x0, max_iter=max_iter)

    return FittingResult(
        seed_corner_idx=corner_idx,
        seed_theta_rad=theta0,
        fit=fit,
        pts_used=pts,
    )
