from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from nlg_guidance_sim.geometry.nlg_model import NLGModel


@dataclass
class Scene:
    name: str = "interactive_scene"
    rail_gauge_m: float = 1.20
    rail_length_m: float = 6.0
    rail_start_x_m: float = -0.20
    platform_front_x_m: float = 0.00
    platform_length_m: float = 1.40
    platform_width_m: float = 0.86
    capture_width_m: float = 0.62
    ramp_length_m: float = 0.30
    nlg: NLGModel = field(default_factory=NLGModel)

    @property
    def guide_axis_y_m(self) -> float:
        return 0.0

    @property
    def left_rail_y_m(self) -> float:
        return self.rail_gauge_m / 2.0

    @property
    def right_rail_y_m(self) -> float:
        return -self.rail_gauge_m / 2.0

    @property
    def capture_x_m(self) -> float:
        return self.platform_front_x_m + self.platform_length_m

    @property
    def platform_body_end_x_m(self) -> float:
        return self.capture_x_m - self.ramp_length_m

    def platform_body_outline(self) -> np.ndarray:
        x0 = self.platform_front_x_m
        x1 = self.platform_body_end_x_m
        half = self.platform_width_m / 2.0
        return np.array(
            [
                [x0, -half],
                [x1, -half],
                [x1,  half],
                [x0,  half],
            ],
            dtype=float,
        )

    def ramp_outline(self) -> np.ndarray:
        x0 = self.platform_body_end_x_m
        x1 = self.capture_x_m
        half_body    = self.platform_width_m  / 2.0
        half_capture = self.capture_width_m   / 2.0
        return np.array(
            [
                [x0, -half_body],
                [x1, -half_capture],
                [x1,  half_capture],
                [x0,  half_body],
            ],
            dtype=float,
        )

    def scan_obstacle_polygons(
        self,
        include_platform: bool = False,
    ) -> list[np.ndarray]:
        """Devuelve los polígonos que el LiDAR debe ver.

        Parameters
        ----------
        include_platform:
            Si es True, añade la plataforma (cuerpo + rampa) como obstáculo.
            Por defecto False: el LiDAR está montado en la plataforma y no
            debe detectar su propia estructura.  Actívalo si quieres modelar
            la plataforma como clutter y filtrarlo en etapas posteriores.
        """
        polygons: list[np.ndarray] = []
        if include_platform:
            polygons.append(self.platform_body_outline())
            polygons.append(self.ramp_outline())
        polygons.extend(self.nlg.wheel_contours_world())
        return polygons

    def summary_dict(self) -> dict[str, float | int | str]:
        midpoint = self.nlg.midpoint
        return {
            "scene": self.name,
            "n_wheels": self.nlg.wheel_count(),
            "midpoint_x_m": float(midpoint[0]),
            "midpoint_y_m": float(midpoint[1]),
            "psi_deg": float(math.degrees(self.nlg.psi_rad)),
            "platform_width_m": float(self.platform_width_m),
            "capture_width_m": float(self.capture_width_m),
            "platform_length_m": float(self.platform_length_m),
            "rail_gauge_m": float(self.rail_gauge_m),
        }
