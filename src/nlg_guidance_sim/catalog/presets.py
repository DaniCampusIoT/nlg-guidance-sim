from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


WheelArrangement = Literal["single", "dual"]


@dataclass(frozen=True)
class AircraftPreset:
    name: str
    arrangement: WheelArrangement
    tire_length_m: float
    tire_width_m: float
    shoulder_exponent: float
    track_width_m: float
    rail_gauge_m: float
    platform_length_m: float
    platform_width_m: float
    capture_width_m: float
    ramp_length_m: float
    center_x_m: float
    center_y_m: float
    psi_deg: float

    def to_yaml_dict(self) -> dict:
        import dataclasses
        d = dataclasses.asdict(self)
        return d


def load_presets_yaml(path: Path | str | None = None) -> dict[str, AircraftPreset]:
    """Load presets from a YAML file.  Falls back to built-in defaults."""
    if path is None:
        # Try the canonical config location relative to the repo root
        candidate = Path(__file__).resolve().parents[4] / "configs" / "aircraft_presets.yaml"
        if candidate.exists():
            path = candidate

    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        result: dict[str, AircraftPreset] = {}
        for entry in raw.get("presets", []):
            p = AircraftPreset(**entry)
            result[p.name] = p
        return result

    # Built-in fallback
    return _builtin_presets()


def _builtin_presets() -> dict[str, AircraftPreset]:
    return {
        "Ligero / rueda simple": AircraftPreset(
            name="Ligero / rueda simple",
            arrangement="single",
            tire_length_m=0.55,
            tire_width_m=0.16,
            shoulder_exponent=3.0,
            track_width_m=0.00,
            rail_gauge_m=1.10,
            platform_length_m=1.20,
            platform_width_m=0.60,
            capture_width_m=0.42,
            ramp_length_m=0.22,
            center_x_m=1.80,
            center_y_m=0.06,
            psi_deg=3.0,
        ),
        "Regional / rueda simple": AircraftPreset(
            name="Regional / rueda simple",
            arrangement="single",
            tire_length_m=0.62,
            tire_width_m=0.20,
            shoulder_exponent=3.2,
            track_width_m=0.00,
            rail_gauge_m=1.20,
            platform_length_m=1.30,
            platform_width_m=0.70,
            capture_width_m=0.48,
            ramp_length_m=0.26,
            center_x_m=2.00,
            center_y_m=0.08,
            psi_deg=4.0,
        ),
        "Single aisle / doble rueda": AircraftPreset(
            name="Single aisle / doble rueda",
            arrangement="dual",
            tire_length_m=0.70,
            tire_width_m=0.22,
            shoulder_exponent=3.6,
            track_width_m=0.45,
            rail_gauge_m=1.20,
            platform_length_m=1.40,
            platform_width_m=0.86,
            capture_width_m=0.62,
            ramp_length_m=0.30,
            center_x_m=2.00,
            center_y_m=0.12,
            psi_deg=4.0,
        ),
        "Widebody / doble rueda": AircraftPreset(
            name="Widebody / doble rueda",
            arrangement="dual",
            tire_length_m=0.78,
            tire_width_m=0.25,
            shoulder_exponent=4.0,
            track_width_m=0.52,
            rail_gauge_m=1.30,
            platform_length_m=1.55,
            platform_width_m=0.96,
            capture_width_m=0.70,
            ramp_length_m=0.34,
            center_x_m=2.20,
            center_y_m=0.14,
            psi_deg=5.0,
        ),
    }


# Module-level default (can be reloaded at runtime)
PRESETS: dict[str, AircraftPreset] = load_presets_yaml()
