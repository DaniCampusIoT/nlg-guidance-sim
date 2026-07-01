"""Load, validate and save AircraftPresets from/to YAML.

The canonical YAML lives at  configs/presets.yaml  (repo root).
The editor tab in app.py can also work on in-memory YAML strings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nlg_guidance_sim.catalog.presets import AircraftPreset, PRESETS, WheelArrangement

_REQUIRED_KEYS: set[str] = {
    "arrangement",
    "tire_length_m",
    "tire_width_m",
    "shoulder_exponent",
    "track_width_m",
    "rail_gauge_m",
    "platform_length_m",
    "platform_width_m",
    "capture_width_m",
    "ramp_length_m",
    "center_x_m",
    "center_y_m",
    "psi_deg",
}


def _validate_raw(name: str, raw: dict[str, Any]) -> None:
    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Preset '{name}' missing keys: {missing}")
    if raw["arrangement"] not in ("single", "dual"):
        raise ValueError(f"Preset '{name}': arrangement must be 'single' or 'dual'")


def _raw_to_preset(name: str, raw: dict[str, Any]) -> AircraftPreset:
    _validate_raw(name, raw)
    return AircraftPreset(
        name=name,
        arrangement=raw["arrangement"],
        tire_length_m=float(raw["tire_length_m"]),
        tire_width_m=float(raw["tire_width_m"]),
        shoulder_exponent=float(raw["shoulder_exponent"]),
        track_width_m=float(raw["track_width_m"]),
        rail_gauge_m=float(raw["rail_gauge_m"]),
        platform_length_m=float(raw["platform_length_m"]),
        platform_width_m=float(raw["platform_width_m"]),
        capture_width_m=float(raw["capture_width_m"]),
        ramp_length_m=float(raw["ramp_length_m"]),
        center_x_m=float(raw["center_x_m"]),
        center_y_m=float(raw["center_y_m"]),
        psi_deg=float(raw["psi_deg"]),
    )


def load_yaml_str(yaml_text: str) -> dict[str, AircraftPreset]:
    """Parse a YAML string and return a dict of AircraftPreset objects."""
    doc = yaml.safe_load(yaml_text)
    if not isinstance(doc, dict) or "presets" not in doc:
        raise ValueError("YAML must have a top-level 'presets' mapping")
    result: dict[str, AircraftPreset] = {}
    for name, raw in doc["presets"].items():
        result[str(name)] = _raw_to_preset(str(name), raw)
    return result


def load_yaml_file(path: Path) -> dict[str, AircraftPreset]:
    """Load presets from a YAML file on disk."""
    return load_yaml_str(path.read_text(encoding="utf-8"))


def presets_to_yaml_str(presets: dict[str, AircraftPreset]) -> str:
    """Serialise a dict of AircraftPreset back to a YAML string."""
    doc: dict[str, Any] = {"presets": {}}
    for name, p in presets.items():
        doc["presets"][name] = {
            "arrangement": p.arrangement,
            "tire_length_m": round(p.tire_length_m, 4),
            "tire_width_m": round(p.tire_width_m, 4),
            "shoulder_exponent": round(p.shoulder_exponent, 2),
            "track_width_m": round(p.track_width_m, 4),
            "rail_gauge_m": round(p.rail_gauge_m, 4),
            "platform_length_m": round(p.platform_length_m, 4),
            "platform_width_m": round(p.platform_width_m, 4),
            "capture_width_m": round(p.capture_width_m, 4),
            "ramp_length_m": round(p.ramp_length_m, 4),
            "center_x_m": round(p.center_x_m, 4),
            "center_y_m": round(p.center_y_m, 4),
            "psi_deg": round(p.psi_deg, 2),
        }
    return yaml.dump(doc, default_flow_style=False, allow_unicode=True, sort_keys=False)


def merged_presets(yaml_text: str | None = None) -> dict[str, AircraftPreset]:
    """Return built-in PRESETS merged with any extra presets from yaml_text.
    Built-in presets are always present; YAML overrides take precedence by name.
    """
    result = dict(PRESETS)
    if yaml_text:
        try:
            extra = load_yaml_str(yaml_text)
            result.update(extra)
        except Exception:
            pass
    return result
