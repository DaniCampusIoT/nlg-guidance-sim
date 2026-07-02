"""Smoke tests for the fitting pipeline and EKF."""
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from nlg_guidance_sim.fitting.rdp import rdp_corner, initial_theta
from nlg_guidance_sim.fitting.rectangle_model import RigidRectangle
from nlg_guidance_sim.fitting.gauss_newton import run_lm
from nlg_guidance_sim.fitting.pipeline import run_fitting_pipeline
from nlg_guidance_sim.estimation.pose_ekf import PoseEKF


def _make_l_points(
    xc: float, yc: float, theta: float,
    W: float, L: float, n: int = 40, noise: float = 0.002,
) -> np.ndarray:
    """Generate synthetic L-shape points from a known ground-truth rectangle."""
    rng = np.random.default_rng(42)
    c, s = math.cos(theta), math.sin(theta)
    h = np.array([c,  s])
    n_vec = np.array([-s, c])
    corner = np.array([xc, yc])

    t_arm1 = np.linspace(0, L, n // 2)
    arm1   = corner[None, :] + np.outer(t_arm1, h)
    t_arm2 = np.linspace(0, W, n // 2)
    arm2   = corner[None, :] + np.outer(t_arm2, n_vec)

    pts = np.vstack([arm1, arm2])
    pts += rng.normal(0, noise, pts.shape)
    return pts


def test_rdp_corner_detects_corner():
    pts = _make_l_points(2.0, 0.1, math.radians(5), W=0.22, L=0.70)
    ci = rdp_corner(pts)
    # The corner should be near the midpoint of the array
    assert 5 < ci < len(pts) - 5, f"Corner index {ci} outside expected range"


def test_rigid_rectangle_residuals_zero_at_truth():
    xc, yc, theta = 2.0, 0.1, math.radians(5)
    W, L = 0.22, 0.70
    rect = RigidRectangle(W=W, L=L)
    # Points exactly on the edges
    c, s = math.cos(theta), math.sin(theta)
    h = np.array([c, s])
    nv = np.array([-s, c])
    corner = np.array([xc, yc])
    pts_on_edge = np.vstack([
        corner + np.outer(np.linspace(0, L, 10), h),
        corner + np.outer(np.linspace(0, W, 10), nv),
    ])
    state = np.array([xc, yc, theta])
    res = rect.residuals(state, pts_on_edge)
    assert np.max(np.abs(res)) < 1e-6, f"Max residual on exact edges: {np.max(np.abs(res)):.2e}"


def test_lm_converges_to_ground_truth():
    xc_true, yc_true, theta_true = 2.0, 0.12, math.radians(4)
    W, L = 0.22, 0.70
    pts = _make_l_points(xc_true, yc_true, theta_true, W=W, L=L, noise=0.003)
    rect = RigidRectangle(W=W, L=L)
    x0   = np.array([xc_true + 0.05, yc_true + 0.03, theta_true + math.radians(3)])
    res  = run_lm(pts, rect, x0)
    assert res.converged, "LM did not converge"
    assert abs(res.xc   - xc_true)    < 0.02, f"|xc error| = {abs(res.xc - xc_true):.4f} m"
    assert abs(res.yc   - yc_true)    < 0.02, f"|yc error| = {abs(res.yc - yc_true):.4f} m"
    assert abs(res.theta - theta_true) < math.radians(2), "theta error > 2 deg"


def test_pipeline_end_to_end():
    pts = _make_l_points(2.0, 0.1, math.radians(5), W=0.22, L=0.70)
    result = run_fitting_pipeline(pts, W_real=0.22, L_real=0.70)
    assert result is not None
    assert result.converged
    assert result.rmse_m < 0.01   # < 1 cm RMSE on synthetic data


def test_ekf_initializes_and_updates():
    ekf = PoseEKF()
    assert not ekf.initialized
    ekf.initialize(2.0, 0.1, math.radians(4))
    assert ekf.initialized
    for _ in range(10):
        ekf.predict(dt=0.05)
        ekf.update(2.0 + 0.01, 0.1 + 0.002)
    assert len(ekf.history) == 11
    assert abs(ekf.Y_m() - 0.1) < 0.05, f"EKF Y far from truth: {ekf.Y_m():.3f}"
