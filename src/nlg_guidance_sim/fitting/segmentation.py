"""
Stage-1 coarse segmentation helpers.

Implements:
  * Ramer–Douglas–Peucker (RDP) polyline simplification
  * Split-and-Merge corner detection
  * Heuristic corner-candidate extraction

References
----------
Zhang et al., "Efficient L-Shape Fitting for Vehicle Detection Using Laser Scanners"
(Search-Based + corner detection discussion, §III).
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# RDP simplification
# ─────────────────────────────────────────────────────────────────────────────

def _cross2d(u: np.ndarray, v: np.ndarray) -> float:
    """Scalar cross product of two 2-D vectors: u×v = u[0]*v[1] - u[1]*v[0].

    Replaces np.cross(u, v) for 2-D vectors, which raises ValueError in
    NumPy ≥2.0 because that version requires 3-D inputs for np.cross.
    """
    return float(u[0] * v[1] - u[1] * v[0])


def _perp_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point p to line ab."""
    ab = b - a
    norm = float(np.linalg.norm(ab))
    if norm < 1e-12:
        return float(np.linalg.norm(p - a))
    # cross product of (b-a) and (a-p) gives the signed area ×2;
    # dividing by |ab| gives the perpendicular distance.
    return abs(_cross2d(ab, a - p)) / norm


def rdp_simplify(points: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification.

    Parameters
    ----------
    points  : (N, 2) ordered point array
    epsilon : distance threshold [m]

    Returns
    -------
    Simplified (M, 2) array, M <= N.
    """
    if len(points) < 3:
        return points.copy()

    max_dist = 0.0
    idx = 0
    a, b = points[0], points[-1]
    for i in range(1, len(points) - 1):
        d = _perp_distance(points[i], a, b)
        if d > max_dist:
            max_dist = d
            idx = i

    if max_dist > epsilon:
        left = rdp_simplify(points[: idx + 1], epsilon)
        right = rdp_simplify(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([points[0], points[-1]])


# ─────────────────────────────────────────────────────────────────────────────
# Split-and-Merge
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    points: np.ndarray       # (N, 2)
    start_idx: int
    end_idx: int


def split_and_merge(
    points: np.ndarray,
    distance_threshold: float = 0.04,
    min_points: int = 4,
) -> list[Segment]:
    """Iterative Split-and-Merge line segmentation.

    Returns a list of Segment objects. Each segment’s *points* are the
    original LiDAR points that belong to that linear piece.

    Parameters
    ----------
    points             : (N, 2) ordered scan points (hits only)
    distance_threshold : max perpendicular residual to keep a point in segment
    min_points         : minimum points required to keep a segment
    """
    if len(points) < min_points:
        return []

    segments: list[Segment] = []

    def _split(start: int, end: int) -> None:
        if end - start < min_points - 1:
            return
        a = points[start]
        b = points[end]
        max_d = 0.0
        split_i = start
        for i in range(start + 1, end):
            d = _perp_distance(points[i], a, b)
            if d > max_d:
                max_d = d
                split_i = i
        if max_d > distance_threshold:
            _split(start, split_i)
            _split(split_i, end)
        else:
            segments.append(Segment(
                points=points[start: end + 1],
                start_idx=start,
                end_idx=end,
            ))

    _split(0, len(points) - 1)
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Corner candidate extraction
# ─────────────────────────────────────────────────────────────────────────────

def find_corner_candidates(
    hit_points: np.ndarray,
    epsilon: float = 0.02,
    distance_threshold: float = 0.04,
    min_seg_points: int = 4,
) -> list[dict]:
    """Extract L-shape corner candidates from an ordered hit-point cloud.

    Strategy (Zhang et al. §III.B):
    1. Simplify the polyline with RDP.
    2. Run Split-and-Merge on the original hits.
    3. For each adjacent segment pair, compute:
       - intersection point (the candidate corner)
       - arm angles
       - a quality score (sum of squared residuals, lower = better)

    Returns
    -------
    List of dicts sorted by score (ascending = best first)::

        {
            "corner": np.ndarray shape (2,),
            "angle_deg": float,          # interior angle at corner
            "theta0_rad": float,         # initial heading estimate
            "score": float,
            "seg_a": Segment,
            "seg_b": Segment,
        }
    """
    if len(hit_points) < 6:
        return []

    segments = split_and_merge(hit_points, distance_threshold, min_seg_points)
    if len(segments) < 2:
        return []

    candidates: list[dict] = []

    for i in range(len(segments) - 1):
        sa = segments[i]
        sb = segments[i + 1]

        def _fit_line(pts: np.ndarray):
            """Returns (direction_unit, point_on_line, residual_sum)."""
            if len(pts) < 2:
                return None, None, np.inf
            c = pts.mean(axis=0)
            _, _, vt = np.linalg.svd(pts - c)
            direction = vt[0]
            # 2-D scalar cross product for residuals: r_i = (p_i - c) × direction
            diff = pts - c          # (N, 2)
            residuals = diff[:, 0] * direction[1] - diff[:, 1] * direction[0]
            return direction / (np.linalg.norm(direction) + 1e-12), c, float((residuals**2).sum())

        da, ca, ra = _fit_line(sa.points)
        db, cb, rb = _fit_line(sb.points)
        if da is None or db is None:
            continue

        # Intersection of two lines: ca + t*da = cb + s*db
        denom = _cross2d(da, db)
        if abs(denom) < 1e-9:
            continue
        diff = cb - ca
        t = _cross2d(diff, db) / denom
        corner = ca + t * da

        # Interior angle
        cos_a = float(np.clip(np.dot(da, db), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(abs(cos_a)))

        # Initial heading: bisector of the two arm directions
        bisector = da + db
        bn = float(np.linalg.norm(bisector))
        if bn > 1e-9:
            bisector /= bn
        theta0 = math.atan2(float(bisector[1]), float(bisector[0]))

        candidates.append({
            "corner": corner.copy(),
            "angle_deg": angle_deg,
            "theta0_rad": theta0,
            "score": ra + rb,
            "seg_a": sa,
            "seg_b": sb,
        })

    candidates.sort(key=lambda c: c["score"])
    return candidates
