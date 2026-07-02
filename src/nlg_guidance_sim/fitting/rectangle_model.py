"""Rigid rectangle model with fixed W and L (known a-priori from preset).

The optimisation state is x = [xc, yc, theta]^T where:
  xc, yc  — corner point of the L (intersection of the two visible arms)
  theta   — heading angle of the rectangle (rad)

W_real and L_real are kept FIXED (locked to the preset dimensions), which
reduces DOF from 5 to 3 and prevents the 'L-flip' local-minimum problem.

Reference: Vehicle Speed Estimation Using Infrastructure-Mounted LiDAR
           via Rectangle Edge Matching (MDPI 2026, Sec. 3.2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RigidRectangle:
    """Immutable rectangle with fixed side lengths."""
    W: float  # width  (metres)  — e.g. tire_width_m from AircraftPreset
    L: float  # length (metres)  — e.g. tire_length_m from AircraftPreset

    # ── geometry helpers ────────────────────────────────────────────────────

    def corners(self, state: np.ndarray) -> np.ndarray:
        """Return the 4 corners of the rectangle given state [xc, yc, theta].

        The corner (xc, yc) is the origin of the L.
        Arm 1 runs along +heading by length L.
        Arm 2 runs perpendicular (+90°) by width W.
        """
        xc, yc, theta = float(state[0]), float(state[1]), float(state[2])
        c, s = math.cos(theta), math.sin(theta)
        h  = np.array([c,  s])   # heading unit vector
        n  = np.array([-s, c])   # normal (perpendicular) unit vector
        p0 = np.array([xc, yc])
        p1 = p0 + self.L * h
        p2 = p1 + self.W * n
        p3 = p0 + self.W * n
        return np.stack([p0, p1, p2, p3])

    def edges(self, state: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return 4 edges as (start, end) pairs."""
        C = self.corners(state)
        return [(C[i], C[(i + 1) % 4]) for i in range(4)]

    # ── residuals ───────────────────────────────────────────────────────────

    def _point_to_edge_dist(
        self,
        p: np.ndarray,   # (2,)
        a: np.ndarray,   # edge start (2,)
        b: np.ndarray,   # edge end   (2,)
    ) -> float:
        """Signed perpendicular distance from p to the infinite line through a→b."""
        ab = b - a
        length = float(np.linalg.norm(ab))
        if length < 1e-9:
            return float(np.linalg.norm(p - a))
        # perpendicular distance (positive = left side of a→b)
        return float((ab[0] * (a[1] - p[1]) - ab[1] * (a[0] - p[0])) / length)

    def _min_abs_dist(
        self,
        p: np.ndarray,
        state: np.ndarray,
    ) -> tuple[float, int]:
        """Return (min |dist|, edge_index) over all 4 edges."""
        best = (math.inf, 0)
        for i, (a, b) in enumerate(self.edges(state)):
            d = abs(self._point_to_edge_dist(p, a, b))
            if d < best[0]:
                best = (d, i)
        return best

    def residuals(self, state: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Compute per-point residuals (N,) = distance to nearest edge.

        Parameters
        ----------
        state : (3,) array [xc, yc, theta]
        pts   : (N, 2) hit points from LiDAR scan
        """
        res = np.empty(len(pts))
        for i, p in enumerate(pts):
            res[i], _ = self._min_abs_dist(p, state)
        return res

    # ── analytic Jacobian ───────────────────────────────────────────────────

    def jacobian(self, state: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Analytic Jacobian J (N x 3) for state = [xc, yc, theta].

        For the closest edge (a→b) with unit normal n_hat = (-s_edge, c_edge):
          r = n_hat · (a - p)  (signed distance)
          dr/dxc   = n_hat[0] * (partial a / partial xc) — analytically derived
          dr/dyc   = n_hat[1]
          dr/dtheta= d(n_hat)/dtheta · (a - p) + n_hat · d(a)/dtheta

        We use a finite-difference fallback for d/dtheta (one step, exact enough
        for LM which only needs approximate J at each iterate).
        """
        eps_th = 1e-5
        state_plus  = state.copy(); state_plus[2]  += eps_th
        state_minus = state.copy(); state_minus[2] -= eps_th

        r0   = self.residuals(state,       pts)
        rp   = self.residuals(state_plus,  pts)
        rm   = self.residuals(state_minus, pts)

        dr_dtheta = (rp - rm) / (2 * eps_th)

        xc, yc, theta = float(state[0]), float(state[1]), float(state[2])
        c, s = math.cos(theta), math.sin(theta)

        J = np.zeros((len(pts), 3))
        for i, p in enumerate(pts):
            _, eidx = self._min_abs_dist(p, state)
            edges_list = self.edges(state)
            a, b = edges_list[eidx]
            ab   = b - a
            L_ab = float(np.linalg.norm(ab))
            if L_ab < 1e-9:
                continue
            # normal unit vector for this edge
            nx = -ab[1] / L_ab
            ny =  ab[0] / L_ab
            # d(r)/d(xc) = nx * d(edge_a[x])/d(xc): edge 0,3 share xc contribution
            J[i, 0] = nx
            J[i, 1] = ny
            J[i, 2] = dr_dtheta[i]
        return J
