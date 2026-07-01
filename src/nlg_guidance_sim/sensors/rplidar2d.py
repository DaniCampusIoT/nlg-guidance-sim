from dataclasses import dataclass


@dataclass
class RPLidar2D:
    rate_hz: float = 10.0
    range_max_m: float = 30.0
