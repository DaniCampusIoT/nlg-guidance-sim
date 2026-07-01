"""Approach phase finite-state machine.

Phase 1 — Acquisition
    NLG is far from the platform; the LiDAR may not yet see the tyres
    clearly.  The system is in open-loop or coarse-guidance mode.
    Transition → Phase 2 when X < phase2_entry_x_m.

Phase 2 — Active guidance
    Both tyres (or the single tyre) are visible.  The L-shape fitter
    delivers Y and psi estimates with sufficient confidence.
    Fine lateral correction is applied.
    Transition → Phase 3 when X < phase3_entry_x_m AND |Y| < y_ok_m
    AND |psi| < psi_ok_deg.

Phase 3 — Capture / engagement
    The NLG is inside the funnel / cuna.  Only short-range sensors
    (proximity, contact) take over; the LiDAR is no longer the primary
    source.  Motion continues at slow speed until full engagement.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ApproachPhase(str, Enum):
    PHASE1 = "1"  # Acquisition
    PHASE2 = "2"  # Active guidance
    PHASE3 = "3"  # Capture / engagement


@dataclass(frozen=True)
class PhaseState:
    phase: ApproachPhase
    label: str
    description: str
    color: str          # hex colour for UI badges
    emoji: str


_STATES: dict[ApproachPhase, PhaseState] = {
    ApproachPhase.PHASE1: PhaseState(
        phase=ApproachPhase.PHASE1,
        label="Fase 1 — Adquisici\u00f3n",
        description=(
            "El NLG est\u00e1 a distancia larga. El LiDAR no ve a\u00fan los neum\u00e1ticos con "
            "claridad suficiente. Guiado en bucle abierto o con sensor largo."
        ),
        color="#3b6ea8",
        emoji="\U0001f7e6",
    ),
    ApproachPhase.PHASE2: PhaseState(
        phase=ApproachPhase.PHASE2,
        label="Fase 2 — Guiado activo",
        description=(
            "Los neum\u00e1ticos son visibles. El fitting L-shape estima Y y \u03c8 con "
            "confianza suficiente. Se aplica correcci\u00f3n lateral fina."
        ),
        color="#c57b2b",
        emoji="\U0001f7e7",
    ),
    ApproachPhase.PHASE3: PhaseState(
        phase=ApproachPhase.PHASE3,
        label="Fase 3 — Captura / enganche",
        description=(
            "El NLG ha entrado en la cuna. Sensores cortos (proximidad/contacto) "
            "toman el relevo. El LiDAR ya no es la fuente primaria."
        ),
        color="#2e7d32",
        emoji="\u2705",
    ),
}


@dataclass(frozen=True)
class PhaseThresholds:
    phase2_entry_x_m: float = 3.50   # X distance at which Phase 2 begins
    phase3_entry_x_m: float = 0.80   # X distance at which Phase 3 begins
    y_ok_m: float = 0.05             # |Y| tolerance to allow Phase 3 entry
    psi_ok_deg: float = 3.0          # |psi| tolerance to allow Phase 3 entry
    confidence_min: float = 0.40     # minimum L-shape confidence for Phase 2


def classify_phase(
    x_m: float,
    Y_m: float = 0.0,
    psi_deg: float = 0.0,
    lidar_confidence: float = 0.0,
    thresholds: PhaseThresholds | None = None,
) -> PhaseState:
    """Return the current PhaseState given the NLG's position and estimation."""
    th = thresholds or PhaseThresholds()

    # Phase 3: inside the capture zone and aligned
    if (
        x_m <= th.phase3_entry_x_m
        and abs(Y_m) <= th.y_ok_m
        and abs(psi_deg) <= th.psi_ok_deg
    ):
        return _STATES[ApproachPhase.PHASE3]

    # Phase 2: close enough AND L-shape confidence is sufficient
    if x_m <= th.phase2_entry_x_m and lidar_confidence >= th.confidence_min:
        return _STATES[ApproachPhase.PHASE2]

    # Default: Phase 1
    return _STATES[ApproachPhase.PHASE1]


def all_phase_states() -> list[PhaseState]:
    return list(_STATES.values())
