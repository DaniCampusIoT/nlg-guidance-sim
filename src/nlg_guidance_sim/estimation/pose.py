"""Derive Y-offset and heading angle psi from an LShapeResult.

Convention
----------
- The guide axis runs along X; the platform is centred at Y=0.
- Y_est  = signed lateral displacement of the corner from the guide axis.
- psi_est = heading angle of the NLG relative to the platform axis (rad).
  Positive = turned left (counter-clockwise viewed from above).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nlg_guidance_sim.estimation.lshape import LShapeResult


@dataclass
class PoseEstimate:
    Y_m: float          # lateral offset  [m]
    psi_rad: float      # heading angle   [rad]
    psi_deg: float      # heading angle   [deg]
    corner_xy: np.ndarray
    confidence: float   # 0–1, based on fitting RMS and corner angle quality

    def __str__(self) -> str:
        return (
            f"Y = {self.Y_m:+.4f} m   "
            f"\u03c8 = {self.psi_deg:+.2f} deg   "
            f"conf = {self.confidence:.2f}"
        )


def estimate_pose(
    result: LShapeResult,
    guide_axis_y: float = 0.0,
    rms_scale: float = 0.005,
) -> PoseEstimate:
    """Convert an LShapeResult to a PoseEstimate.

    The two arms of the L are assumed to be:
      - arm 1: roughly parallel to X  (longitudinal face of the tyre)
      - arm 2: roughly parallel to Y  (lateral face / side wall)

    We pick the arm whose normal is more aligned with Y to derive psi.
    """
    # --- lateral offset from corner ---
    Y_est = float(result.corner_xy[1]) - guide_axis_y

    # --- heading from the arm that is more along X ---
    n1 = np.array([result.line1.a, result.line1.b])
    n2 = np.array([result.line2.a, result.line2.b])

    # arm aligned with X has its normal most aligned with Y-axis
    y_axis = np.array([0.0, 1.0])
    dot1 = abs(float(np.dot(n1, y_axis)))
    dot2 = abs(float(np.dot(n2, y_axis)))

    # choose the arm whose normal is most Y-aligned → that arm is longitudinal
    long_normal = n1 if dot1 >= dot2 else n2

    # The heading psi is the angle between X-axis and the arm direction.
    # arm direction is perpendicular to its normal.
    # normal = (a, b) → direction = (-b, a)
    arm_dir = np.array([-long_normal[1], long_normal[0]])
    # ensure it points in the +X direction
    if arm_dir[0] < 0:
        arm_dir = -arm_dir
    psi_rad = math.atan2(float(arm_dir[1]), float(arm_dir[0]))

    # --- confidence ---
    # penalise high RMS and penalise deviation of corner angle from 90 deg
    angle_err = abs(result.corner_angle_deg - 90.0)  # ideal = 0
    rms_penalty = min(result.total_rms / rms_scale, 1.0)
    angle_penalty = min(angle_err / 45.0, 1.0)
    confidence = max(0.0, 1.0 - 0.5 * rms_penalty - 0.5 * angle_penalty)

    return PoseEstimate(
        Y_m=Y_est,
        psi_rad=psi_rad,
        psi_deg=math.degrees(psi_rad),
        corner_xy=result.corner_xy.copy(),
        confidence=confidence,
    )
