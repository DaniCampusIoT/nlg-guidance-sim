from pathlib import Path
import sys
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.estimation.lshape import fit_lshape, LShapeFitResult


def _make_l_points(Y: float, psi: float, n: int = 80) -> np.ndarray:
    """Build a synthetic L-shape point cloud centred at (2.0, Y) with heading psi."""
    arm1_len = 0.35
    arm2_len = 0.20
    t1 = np.linspace(0, arm1_len, n // 2)
    t2 = np.linspace(0, arm2_len, n // 2)
    c, s = math.cos(psi), math.sin(psi)
    pts1 = np.column_stack([2.0 + t1 * c,  Y + t1 * s])
    pts2 = np.column_stack([2.0 - t2 * s,  Y + t2 * c])
    noise = np.random.default_rng(0).normal(0, 0.001, (n, 2))
    return np.vstack([pts1, pts2]) + noise


def test_lshape_Y_accuracy():
    pts = _make_l_points(Y=0.12, psi=math.radians(4.0))
    res = fit_lshape(pts)
    assert abs(res.Y_m - 0.12) < 0.05, f"Y error too large: {res.Y_m:.4f}"


def test_lshape_psi_accuracy():
    psi_true = math.radians(5.0)
    pts = _make_l_points(Y=0.0, psi=psi_true)
    res = fit_lshape(pts)
    assert abs(res.psi_rad - psi_true) < math.radians(3.0), (
        f"psi error too large: {math.degrees(res.psi_rad):.2f} deg"
    )


def test_fallback_centroid_few_points():
    pts = np.array([[2.0, 0.1], [2.1, 0.1]])
    res = fit_lshape(pts)
    assert res.method == "centroid"
    assert res.confidence == 0.0


def test_result_fields():
    pts = _make_l_points(Y=0.0, psi=0.0)
    res = fit_lshape(pts)
    assert isinstance(res, LShapeFitResult)
    assert 0.0 <= res.confidence <= 1.0
    assert res.method in ("lshape", "line", "centroid")
