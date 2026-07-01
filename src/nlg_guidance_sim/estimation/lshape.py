"""L-shape fitting over an ordered 2-D point cloud.

Based on the criterion search described in:
  Zhang et al., "Efficient L-shape fitting of laser scanner data
  for vehicle pose estimation", CISRAM 2015.

The algorithm sweeps candidate split indices over the ordered scan,
fitting two line segments (the two arms of the L) and choosing the
split that minimises the total least-squares residual.

Preserving angular order of input points is mandatory — the caller
must pass `ScanResult.ordered_hit_points()` directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class LineSegment(NamedTuple):
    """Infinite line  ax + by + c = 0  (unit normal)."""
    a: float
    b: float
    c: float

    def residuals(self, pts: np.ndarray) -> np.ndarray:
        """Signed distances from pts (N,2) to this line."""
        return self.a * pts[:, 0] + self.b * pts[:, 1] + self.c

    def rms(self, pts: np.ndarray) -> float:
        if len(pts) < 2:
            return np.inf
        return float(np.sqrt(np.mean(self.residuals(pts) ** 2)))


def _fit_line_tls(pts: np.ndarray) -> LineSegment | None:
    """Total Least Squares line fit.  Returns None if degenerate."""
    if len(pts) < 2:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[-1]  # last right singular vector = normal direction
    a, b = float(normal[0]), float(normal[1])
    c = -(a * centroid[0] + b * centroid[1])
    return LineSegment(a, b, c)


@dataclass
class LShapeResult:
    corner_xy: np.ndarray           # estimated corner point (2,)
    line1: LineSegment              # first arm
    line2: LineSegment              # second arm
    split_index: int                # index that separates the two clusters
    rms1: float
    rms2: float
    total_rms: float
    pts1: np.ndarray                # points on arm 1
    pts2: np.ndarray                # points on arm 2

    @property
    def corner_angle_deg(self) -> float:
        """Angle between the two line normals (should be ~90 deg for a true L)."""
        n1 = np.array([self.line1.a, self.line1.b])
        n2 = np.array([self.line2.a, self.line2.b])
        cos_val = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(abs(cos_val))))


class LShapeFitter:
    """Sweep-based L-shape fitter for an ordered 2-D point cloud."""

    def __init__(self, min_pts_per_arm: int = 5) -> None:
        self.min_pts_per_arm = min_pts_per_arm

    def fit(self, pts: np.ndarray) -> LShapeResult | None:
        """Fit an L-shape to the ordered hit-point array (N, 2).

        Returns None when not enough points are available.
        """
        n = len(pts)
        if n < 2 * self.min_pts_per_arm:
            return None

        best: LShapeResult | None = None
        best_rms = np.inf

        for k in range(self.min_pts_per_arm, n - self.min_pts_per_arm + 1):
            p1 = pts[:k]
            p2 = pts[k:]

            l1 = _fit_line_tls(p1)
            l2 = _fit_line_tls(p2)
            if l1 is None or l2 is None:
                continue

            r1 = l1.rms(p1)
            r2 = l2.rms(p2)
            total = (r1 * len(p1) + r2 * len(p2)) / n

            if total < best_rms:
                best_rms = total
                corner = self._intersect(l1, l2)
                if corner is None:
                    continue
                best = LShapeResult(
                    corner_xy=corner,
                    line1=l1,
                    line2=l2,
                    split_index=k,
                    rms1=r1,
                    rms2=r2,
                    total_rms=total,
                    pts1=p1,
                    pts2=p2,
                )

        return best

    @staticmethod
    def _intersect(l1: LineSegment, l2: LineSegment) -> np.ndarray | None:
        """Intersection of two infinite lines  a·x + b·y + c = 0."""
        det = l1.a * l2.b - l2.a * l1.b
        if abs(det) < 1e-9:
            return None
        x = (l1.b * l2.c - l2.b * l1.c) / det
        y = (l2.a * l1.c - l1.a * l2.c) / det
        return np.array([x, y], dtype=float)
