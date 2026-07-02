"""Derive PoseEstimate (Y, psi, confidence) from an LShapeFitResult.

Convention
----------
- The guide axis runs along X; the platform is centred at Y=0.
- Y_est  = signed lateral displacement of the L-corner from the guide axis.
- psi_est = heading angle of the NLG relative to the platform axis (rad).
  Positive = turned left (counter-clockwise viewed from above).

This module works exclusively with LShapeFitResult (the dataclass produced
by nlg_guidance_sim.estimation.lshape.fit_lshape).  It no longer references
the old LShapeResult / line1 / line2 / corner_angle_deg API.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nlg_guidance_sim.estimation.lshape import LShapeFitResult


@dataclass
class EstResult:
    """Pose estimate derived from a single LiDAR scan."""
    Y_m:        float           # lateral offset  [m]
    psi_rad:    float           # heading angle   [rad]
    psi_deg:    float           # heading angle   [deg]
    corner_xy:  np.ndarray      # estimated L-corner in world frame
    confidence: float           # 0–1
    method:     str             # propagated from LShapeFitResult.method

    def __str__(self) -> str:
        return (
            f"Y = {self.Y_m:+.4f} m   "
            f"\u03c8 = {self.psi_deg:+.2f} deg   "
            f"conf = {self.confidence:.2f}   "
            f"[{self.method}]"
        )


# Keep the old name as an alias so any external code using PoseEstimate still works
PoseEstimate = EstResult


def estimate_pose(
    result: LShapeFitResult,
    guide_axis_y: float = 0.0,
    rms_scale: float = 0.05,
) -> EstResult:
    """Convert an LShapeFitResult to an EstResult.

    The lateral offset Y is taken directly from the corner Y-coordinate
    (already computed inside fit_lshape).

    The heading psi is also already computed inside fit_lshape; we just
    apply a small sign correction so that a wheel turned left gives psi > 0.
    """
    Y_est = float(result.corner_xy[1]) - guide_axis_y

    # psi already normalised to (-π/2, π/2) by fit_lshape
    psi_rad = result.psi_rad

    # Confidence: penalise high RMS (normalised by rms_scale)
    rms_penalty = min(result.rmse_m / rms_scale, 1.0) if math.isfinite(result.rmse_m) else 1.0
    confidence  = max(0.0, result.confidence * (1.0 - 0.3 * rms_penalty))

    return EstResult(
        Y_m=Y_est,
        psi_rad=psi_rad,
        psi_deg=math.degrees(psi_rad),
        corner_xy=result.corner_xy.copy(),
        confidence=confidence,
        method=result.method,
    )
