from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np

from nlg_guidance_sim.geometry.profiles import Tire2DProfile, rotation_matrix


WheelArrangement = Literal["single", "dual"]


@dataclass
class NLGModel:
    arrangement: WheelArrangement = "dual"
    track_width_m: float = 0.45
    center_x_m: float = 2.0
    center_y_m: float = 0.12
    psi_rad: float = math.radians(4.0)
    tire_profile: Tire2DProfile = field(default_factory=Tire2DProfile)

    def wheel_centers(self) -> list[np.ndarray]:
        center = np.array([self.center_x_m, self.center_y_m], dtype=float)
        if self.arrangement == "single":
            return [center]
        half = self.track_width_m / 2.0
        return [
            np.array([self.center_x_m, self.center_y_m + half], dtype=float),
            np.array([self.center_x_m, self.center_y_m - half], dtype=float),
        ]

    @property
    def midpoint(self) -> np.ndarray:
        centers = self.wheel_centers()
        return np.mean(np.vstack(centers), axis=0)

    def heading_vector(self) -> np.ndarray:
        return np.array([math.cos(self.psi_rad), math.sin(self.psi_rad)], dtype=float)

    def axle_vector(self) -> np.ndarray:
        return np.array([-math.sin(self.psi_rad), math.cos(self.psi_rad)], dtype=float)

    def wheel_contour_world(self, center_world: np.ndarray) -> np.ndarray:
        local = self.tire_profile.contour_local()
        rot = rotation_matrix(self.psi_rad)
        return local @ rot.T + center_world

    def wheel_contours_world(self) -> list[np.ndarray]:
        return [self.wheel_contour_world(center) for center in self.wheel_centers()]

    def wheel_count(self) -> int:
        return len(self.wheel_centers())
