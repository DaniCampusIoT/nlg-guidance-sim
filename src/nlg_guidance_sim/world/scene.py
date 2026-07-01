from __future__ import annotations

from dataclasses import dataclass, field

from nlg_guidance_sim.geometry.nlg_model import NLGModel


@dataclass
class Scene:
    name: str = "nominal_scene"
    rail_gauge_m: float = 1.20
    rail_length_m: float = 5.0
    platform_front_x_m: float = 0.0
    platform_length_m: float = 1.4
    capture_width_m: float = 0.8
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

    def summary(self) -> str:
        return (
            f"Scene(name={self.name}, midpoint=({self.nlg.midpoint[0]:.3f}, {self.nlg.midpoint[1]:.3f}), "
            f"psi_deg={self.nlg.psi_rad * 180.0 / 3.141592653589793:.2f})"
        )
