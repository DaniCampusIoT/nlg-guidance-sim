"""
Full two-stage fitting pipeline.

Stage 1: Search-Based L-shape (LShapeFit) -> coarse pose (xc, yc, theta)
Stage 2: Levenberg-Marquardt refinement (refine_pose) -> precise pose

Usage
-----
    from nlg_guidance_sim.fitting.pipeline import run_pipeline, PipelineResult

    result = run_pipeline(
        hit_points=scan.ordered_hit_points(),
        width_m=preset.tire_width_m,
        length_m=preset.tire_length_m,
    )
    # Access the raw point cloud that was used:
    #   result.hit_points_array  ->  (N, 2) np.ndarray
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .lshape import LShapeFit, LShapeResult
from .optimizer import RefinedPose, refine_pose
from .segmentation import find_corner_candidates


@dataclass
class PipelineResult:
    """Complete output of the two-stage pipeline.

    Attributes
    ----------
    coarse            : LShapeResult — coarse pose from Stage-1 angular sweep
    refined           : RefinedPose  — LM-refined pose from Stage-2
    elapsed_ms        : wall-clock time for the full pipeline [ms]
    n_points_used     : number of LiDAR hit points processed
    hit_points_array  : (N, 2) array of the actual hit points used —
                        stored here so visualisation helpers can draw
                        the point cloud without re-running the scan.
                        Previously absent, causing fitting_plots.py to
                        wrongly read coarse.n_points (int) as a point array.
    corner_candidates : raw output of the segmentation / corner detector
    """
    # Coarse (Stage 1)
    coarse: LShapeResult
    # Refined (Stage 2)
    refined: RefinedPose
    # Diagnostics
    elapsed_ms: float
    n_points_used: int
    # Point cloud used by the pipeline — needed by fitting_plots.py
    hit_points_array: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    corner_candidates: list[dict] = field(default_factory=list)

    # -- Convenience accessors -----------------------------------------------

    @property
    def Y_m(self) -> float:
        """Lateral offset of the NLG midpoint from guide axis [m]."""
        return self.refined.yc

    @property
    def psi_deg(self) -> float:
        """Heading / obliquity angle [deg]."""
        import math
        return math.degrees(self.refined.theta_rad)

    @property
    def X_m(self) -> float:
        """Longitudinal position of the NLG midpoint [m]."""
        return self.refined.xc

    def summary(self) -> dict:
        return {
            "X_m": round(self.X_m, 4),
            "Y_m": round(self.Y_m, 4),
            "psi_deg": round(self.psi_deg, 3),
            "rmse_m": round(self.refined.rmse, 5),
            "converged": self.refined.converged,
            "lm_iters": self.refined.n_iter,
            "cost_init": round(self.coarse.cost, 6),
            "cost_final": round(self.refined.cost_final, 6),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "n_points": self.n_points_used,
        }


def run_pipeline(
    hit_points: np.ndarray,
    width_m: float,
    length_m: float,
    theta_step_deg: float = 0.5,
    rdp_epsilon: float = 0.02,
    seg_threshold: float = 0.04,
    lm_method: str = "lm",
    max_nfev: int = 200,
) -> PipelineResult:
    """Execute the two-stage L-shape fitting pipeline.

    Parameters
    ----------
    hit_points     : (N, 2) ordered LiDAR hit points (angular order preserved)
    width_m        : NLG width prior W [m]
    length_m       : NLG length prior L [m]
    theta_step_deg : angular resolution of coarse sweep [deg]
    rdp_epsilon    : RDP simplification threshold [m]
    seg_threshold  : Split-and-Merge perpendicular distance threshold [m]
    lm_method      : scipy least_squares method ('lm' or 'trf')
    max_nfev       : max function evaluations for LM
    """
    t0 = time.perf_counter()

    # Keep a clean copy for diagnostics / visualisation
    pts = np.asarray(hit_points, dtype=float)

    # Stage 0: corner candidates (for diagnostics / multi-hypothesis)
    corners = find_corner_candidates(
        pts, epsilon=rdp_epsilon, distance_threshold=seg_threshold
    )

    # Stage 1: Search-based L-shape (coarse)
    fitter = LShapeFit(
        width_m=width_m,
        length_m=length_m,
        theta_step_deg=theta_step_deg,
    )
    coarse = fitter.fit(pts)

    # Initial state: use best corner candidate if available, else centroid
    if corners:
        best = corners[0]
        xc0 = float(best["corner"][0])
        yc0 = float(best["corner"][1])
        theta0 = float(best["theta0_rad"])
    else:
        xc0, yc0, theta0 = coarse.xc, coarse.yc, coarse.theta_rad

    # Stage 2: LM refinement (precise)
    refined = refine_pose(
        xc0=xc0, yc0=yc0, theta0=theta0,
        width_m=width_m, length_m=length_m,
        points=pts,
        method=lm_method,
        max_nfev=max_nfev,
    )

    elapsed = (time.perf_counter() - t0) * 1000.0

    return PipelineResult(
        coarse=coarse,
        refined=refined,
        elapsed_ms=elapsed,
        n_points_used=len(pts),
        hit_points_array=pts,          # <-- nube completa, usada por fitting_plots
        corner_candidates=corners,
    )
