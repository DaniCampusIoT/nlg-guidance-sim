"""Extended Kalman Filter (CVTR model) for NLG pose tracking.

State vector: X = [x, y, theta, v, omega]^T
  x, y    — NLG corner/center position (m)
  theta   — heading angle (rad)
  v       — forward speed (m/s)
  omega   — yaw rate (rad/s)

Measurement: z = [x_meas, y_meas]^T   (from LM fitting output, no theta used
             directly as filter input — same design decision as Cellina et al.
             arXiv 2025, Section III-C, to be robust under non-nominal conditions).

Motion model (discrete, Constant Velocity and Turn Rate — CVTR):
  x_{k+1}   = x_k + v_k * cos(theta_k) * dt
  y_{k+1}   = y_k + v_k * sin(theta_k) * dt
  theta_{k+1}= theta_k + omega_k * dt
  v_{k+1}   = v_k
  omega_{k+1}= omega_k

Reference: Cellina et al. arXiv 2025, eq. (6) — CVTR / Coordinated Turn model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class EKFState:
    x: float     = 0.0   # position X (m)
    y: float     = 0.0   # position Y (m)
    theta: float = 0.0   # heading   (rad)
    v: float     = 0.0   # speed     (m/s)
    omega: float = 0.0   # yaw rate  (rad/s)

    def to_vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v, self.omega])

    @staticmethod
    def from_vec(vec: np.ndarray) -> "EKFState":
        return EKFState(x=vec[0], y=vec[1], theta=vec[2], v=vec[3], omega=vec[4])


@dataclass
class PoseEKF:
    """EKF tracker for a single NLG object."""
    # Process noise covariance (tuning)
    q_pos:   float = 0.005   # position noise std (m)
    q_theta: float = 0.010   # heading noise std  (rad)
    q_v:     float = 0.020   # velocity noise std (m/s)
    q_omega: float = 0.010   # yaw-rate noise std (rad/s)

    # Measurement noise covariance (tuning)
    r_pos: float = 0.030     # position noise std (m) — matches Slamtec S2E ±30 mm

    # Internal state
    _X: np.ndarray = field(default_factory=lambda: np.zeros(5))
    _P: np.ndarray = field(default_factory=lambda: np.eye(5) * 0.1)
    _initialized: bool = False
    _history: list[EKFState] = field(default_factory=list)

    # ── Initialise ──────────────────────────────────────────────────────────

    def initialize(self, x: float, y: float, theta: float = 0.0) -> None:
        self._X = np.array([x, y, theta, 0.0, 0.0])
        self._P = np.diag([0.05, 0.05, 0.10, 0.20, 0.05])
        self._initialized = True
        self._history.clear()
        self._history.append(EKFState.from_vec(self._X))

    # ── CVTR motion model ───────────────────────────────────────────────────

    def _f(self, X: np.ndarray, dt: float) -> np.ndarray:
        x, y, th, v, om = X
        return np.array([
            x  + v * math.cos(th) * dt,
            y  + v * math.sin(th) * dt,
            th + om * dt,
            v,
            om,
        ])

    def _F_jacobian(self, X: np.ndarray, dt: float) -> np.ndarray:
        """Jacobian of the motion model wrt state X."""
        _, _, th, v, _ = X
        c, s = math.cos(th), math.sin(th)
        F = np.eye(5)
        F[0, 2] = -v * s * dt   # dx/dtheta
        F[0, 3] =  c * dt        # dx/dv
        F[1, 2] =  v * c * dt   # dy/dtheta
        F[1, 3] =  s * dt        # dy/dv
        F[2, 4] =  dt            # dtheta/domega
        return F

    def _Q(self) -> np.ndarray:
        return np.diag([
            self.q_pos**2,
            self.q_pos**2,
            self.q_theta**2,
            self.q_v**2,
            self.q_omega**2,
        ])

    def _R(self) -> np.ndarray:
        return np.diag([self.r_pos**2, self.r_pos**2])

    # ── EKF predict / update ────────────────────────────────────────────────

    def predict(self, dt: float = 0.05) -> EKFState:
        """EKF predict step."""
        if not self._initialized:
            return EKFState()
        F = self._F_jacobian(self._X, dt)
        self._X = self._f(self._X, dt)
        self._P = F @ self._P @ F.T + self._Q()
        return EKFState.from_vec(self._X)

    def update(self, x_meas: float, y_meas: float) -> EKFState:
        """EKF update step with position measurement [x, y].

        Heading theta is NOT used as a direct measurement (Cellina et al. 2025
        design choice for robustness to non-nominal states).
        """
        if not self._initialized:
            self.initialize(x_meas, y_meas)
            return EKFState.from_vec(self._X)

        H = np.zeros((2, 5))
        H[0, 0] = 1.0   # z[0] = x
        H[1, 1] = 1.0   # z[1] = y

        z   = np.array([x_meas, y_meas])
        z_hat = H @ self._X
        innov = z - z_hat

        S = H @ self._P @ H.T + self._R()
        K = self._P @ H.T @ np.linalg.inv(S)

        self._X = self._X + K @ innov
        self._P = (np.eye(5) - K @ H) @ self._P

        st = EKFState.from_vec(self._X)
        self._history.append(st)
        return st

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def state(self) -> EKFState:
        return EKFState.from_vec(self._X)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def history(self) -> list[EKFState]:
        return list(self._history)

    def Y_m(self) -> float:
        return float(self._X[1])

    def psi_rad(self) -> float:
        return float(self._X[2])

    def psi_deg(self) -> float:
        return float(math.degrees(self._X[2]))

    def velocity(self) -> float:
        return float(self._X[3])
