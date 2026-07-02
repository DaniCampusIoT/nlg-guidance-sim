"""
Smoke + unit tests for the two-stage fitting pipeline.
"""
from pathlib import Path
import sys
import math

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.fitting.segmentation import rdp_simplify, split_and_merge, find_corner_candidates
from nlg_guidance_sim.fitting.lshape import LShapeFit
from nlg_guidance_sim.fitting.optimizer import refine_pose
from nlg_guidance_sim.fitting.pipeline import run_pipeline
from nlg_guidance_sim.fitting.phases import classify_phase, Phase, y_alert, psi_alert


# ── helpers ──────────────────────────────────────────────────────────────────

def _rect_points(xc, yc, theta, W, L, n=80, noise=0.002, seed=0):
    """Generate noisy points on two visible sides of a rectangle."""
    rng = np.random.default_rng(seed)
    half_l, half_w = L / 2, W / 2
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Front edge + right side
    ts = np.linspace(-half_l, half_l, n // 2)
    front = np.column_stack([ts, np.full(n // 2, half_w)])
    right = np.column_stack([np.full(n // 2, half_l), np.linspace(-half_w, half_w, n // 2)])
    local = np.vstack([front, right])
    # Rotate and translate
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    world = (R @ local.T).T + np.array([xc, yc])
    world += rng.normal(0, noise, world.shape)
    return world


# ── segmentation tests ───────────────────────────────────────────────────────

def test_rdp_simplify_line():
    pts = np.column_stack([np.linspace(0, 1, 50), np.zeros(50)])
    simple = rdp_simplify(pts, epsilon=0.01)
    assert len(simple) <= 3  # almost a straight line


def test_rdp_simplify_corner():
    # L-shape: right + front side
    pts = np.vstack([
        np.column_stack([np.zeros(20), np.linspace(0, 1, 20)]),
        np.column_stack([np.linspace(0, 1, 20), np.ones(20)]),
    ])
    simple = rdp_simplify(pts, epsilon=0.01)
    assert len(simple) >= 3  # corner must survive


def test_split_and_merge_two_segments():
    pts = np.vstack([
        np.column_stack([np.linspace(0, 1, 20), np.zeros(20)]),
        np.column_stack([np.ones(20), np.linspace(0, 1, 20)]),
    ])
    segs = split_and_merge(pts, distance_threshold=0.03)
    assert len(segs) >= 1


def test_find_corner_candidates_lshape():
    pts = np.vstack([
        np.column_stack([np.linspace(0, 1, 30), np.zeros(30)]),
        np.column_stack([np.ones(30), np.linspace(0, 0.5, 30)]),
    ])
    cands = find_corner_candidates(pts)
    assert isinstance(cands, list)


# ── L-shape fitter tests ─────────────────────────────────────────────────────

def test_lshape_fit_zero_angle():
    W, L = 0.22, 0.70
    pts = _rect_points(2.0, 0.0, 0.0, W, L, noise=0.001)
    result = LShapeFit(width_m=W, length_m=L).fit(pts)
    assert result.n_points == len(pts)
    assert result.cost < 1.0


def test_lshape_fit_nonzero_angle():
    W, L = 0.22, 0.70
    theta_true = math.radians(5.0)
    pts = _rect_points(2.0, 0.1, theta_true, W, L, noise=0.002)
    result = LShapeFit(width_m=W, length_m=L).fit(pts)
    assert result.cost < 1.0


# ── optimizer tests ──────────────────────────────────────────────────────────

def test_refine_pose_converges():
    W, L = 0.22, 0.70
    xc_true, yc_true, theta_true = 2.0, 0.08, math.radians(4.0)
    pts = _rect_points(xc_true, yc_true, theta_true, W, L, noise=0.002)
    refined = refine_pose(
        xc0=xc_true + 0.05, yc0=yc_true + 0.03, theta0=theta_true + 0.05,
        width_m=W, length_m=L, points=pts,
    )
    assert refined.converged
    assert refined.rmse < 0.05
    assert abs(refined.yc - yc_true) < 0.05
    assert abs(math.degrees(refined.theta_rad) - math.degrees(theta_true)) < 2.0


# ── pipeline tests ───────────────────────────────────────────────────────────

def test_pipeline_summary_keys():
    W, L = 0.22, 0.70
    pts = _rect_points(2.0, 0.05, math.radians(3.0), W, L)
    result = run_pipeline(pts, width_m=W, length_m=L)
    s = result.summary()
    for key in ("X_m", "Y_m", "psi_deg", "rmse_m", "converged", "elapsed_ms"):
        assert key in s, f"Missing key: {key}"


def test_pipeline_y_accuracy():
    W, L = 0.22, 0.70
    yc_true = 0.12
    pts = _rect_points(2.0, yc_true, math.radians(4.0), W, L, noise=0.002)
    result = run_pipeline(pts, width_m=W, length_m=L)
    assert abs(result.Y_m - yc_true) < 0.08


# ── phase tests ───────────────────────────────────────────────────────────────

def test_phase_far():
    state = classify_phase(x_m=5.5, capture_x_m=2.0)
    assert state.phase == Phase.FAR


def test_phase_capture():
    state = classify_phase(x_m=2.0, capture_x_m=2.0)
    assert state.phase == Phase.CAPTURE


def test_phase_locked():
    state = classify_phase(x_m=1.0, capture_x_m=2.0)
    assert state.phase == Phase.LOCKED


def test_y_alert_centred():
    label, _ = y_alert(0.05)
    assert "OK" in label


def test_psi_alert_excessive():
    label, _ = psi_alert(15.0)
    assert "ERROR" in label
