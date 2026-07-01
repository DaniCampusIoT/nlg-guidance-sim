from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def rotation_matrix(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def superellipse_points(
    half_length: float,
    half_width: float,
    exponent: float,
    n_points: int = 180,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    ct = np.cos(t)
    st = np.sin(t)

    x = half_length * np.sign(ct) * np.abs(ct) ** (2.0 / exponent)
    y = half_width * np.sign(st) * np.abs(st) ** (2.0 / exponent)
    return np.column_stack([x, y])


@dataclass(frozen=True)
class Tire2DProfile:
    tire_length_m: float = 0.70
    tire_width_m: float = 0.22
    shoulder_exponent: float = 3.6
    n_profile_points: int = 180

    def contour_local(self) -> np.ndarray:
        return superellipse_points(
            half_length=self.tire_length_m / 2.0,
            half_width=self.tire_width_m / 2.0,
            exponent=self.shoulder_exponent,
            n_points=self.n_profile_points,
        )
