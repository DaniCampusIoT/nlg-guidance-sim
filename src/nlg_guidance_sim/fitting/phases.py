"""
Approach phase logic (Phase 1 / 2 / 3).

Phase 1 -- Far approach  : X > capture_x - approach_threshold
Phase 2 -- Capture zone  : X in [capture_x - capture_depth, capture_x]
Phase 3 -- Locked / fine : X <= capture_x (ramp region)

Thresholds are configurable; defaults match a typical narrow-body NLG.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import math


class Phase(Enum):
    UNKNOWN = auto()
    FAR = auto()       # Phase 1
    CAPTURE = auto()   # Phase 2
    LOCKED = auto()    # Phase 3


@dataclass
class PhaseState:
    phase: Phase
    label: str
    color: str           # for UI / plotting
    description: str


_PHASE_META: dict[Phase, tuple[str, str, str]] = {
    Phase.UNKNOWN: ("Desconocido", "#aaaaaa", "Sin datos suficientes"),
    Phase.FAR:     ("Fase 1 - Aproximacion lejana", "#c57b2b", "Guiado grueso: X estimado > zona de captura"),
    Phase.CAPTURE: ("Fase 2 - Zona de captura", "#01696f", "LM activo: cuna visible, estimando Y y psi"),
    Phase.LOCKED:  ("Fase 3 - Bloqueado / fino", "#3b6ea8", "Refinamiento milimetrico en rampa"),
}


def classify_phase(
    x_m: float,
    capture_x_m: float,
    far_threshold_m: float = 2.5,
    capture_depth_m: float = 0.60,
) -> PhaseState:
    """Classify the current approach phase.

    Parameters
    ----------
    x_m             : estimated NLG longitudinal position [m]
    capture_x_m     : X coordinate of the capture mouth [m]
    far_threshold_m : distance from capture_x that defines Phase 1 boundary
    capture_depth_m : depth of capture zone (Phase 2 width)
    """
    if x_m > capture_x_m + far_threshold_m:
        p = Phase.FAR
    elif x_m > capture_x_m - capture_depth_m:
        p = Phase.CAPTURE
    else:
        p = Phase.LOCKED

    label, color, desc = _PHASE_META[p]
    return PhaseState(phase=p, label=label, color=color, description=desc)


def y_alert(y_m: float, y_limit_m: float = 0.25) -> tuple[str, str]:
    """Return (status_label, color) for lateral offset alert."""
    if abs(y_m) < y_limit_m * 0.5:
        return "OK Centrado", "#2e7d32"
    if abs(y_m) < y_limit_m:
        return "WARN Desviacion", "#c57b2b"
    return "ERROR Fuera de limite", "#b94e48"


def psi_alert(psi_deg: float, psi_limit_deg: float = 8.0) -> tuple[str, str]:
    """Return (status_label, color) for heading angle alert."""
    if abs(psi_deg) < psi_limit_deg * 0.4:
        return "OK Alineado", "#2e7d32"
    if abs(psi_deg) < psi_limit_deg:
        return "WARN Oblicuidad", "#c57b2b"
    return "ERROR Angulo excesivo", "#b94e48"
