from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]])


def superellipse_points(
    half_length: float,
    half_width: float,
    exponent: float,
    n_points: int = 160,
) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    ct = np.cos(t)
    st = np.sin(t)

    x = half_length * np.sign(ct) * np.abs(ct) ** (2.0 / exponent)
    y = half_width * np.sign(st) * np.abs(st) ** (2.0 / exponent)
    return np.column_stack([x, y])


@dataclass
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


@dataclass
class NLGModel:
    track_width_m: float = 0.45
    center_x_m: float = 2.0
    center_y_m: float = 0.12
    psi_rad: float = math.radians(4.0)
    tire_profile: Tire2DProfile = field(default_factory=Tire2DProfile)

    @property
    def left_wheel_center(self) -> np.ndarray:
        return np.array([self.center_x_m, self.center_y_m + self.track_width_m / 2.0])

    @property
    def right_wheel_center(self) -> np.ndarray:
        return np.array([self.center_x_m, self.center_y_m - self.track_width_m / 2.0])

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.left_wheel_center + self.right_wheel_center)

    def heading_vector(self) -> np.ndarray:
        return np.array([math.cos(self.psi_rad), math.sin(self.psi_rad)])

    def axle_vector(self) -> np.ndarray:
        return np.array([-math.sin(self.psi_rad), math.cos(self.psi_rad)])

    def wheel_contour_world(self, center_world: np.ndarray) -> np.ndarray:
        local = self.tire_profile.contour_local()
        rot = rotation_matrix(self.psi_rad)
        return local @ rot.T + center_world

    def left_wheel_contour_world(self) -> np.ndarray:
        return self.wheel_contour_world(self.left_wheel_center)

    def right_wheel_contour_world(self) -> np.ndarray:
        return self.wheel_contour_world(self.right_wheel_center)