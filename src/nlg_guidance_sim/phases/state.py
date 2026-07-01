"""Approach phase finite-state machine for NLG guidance.

Phase definitions (aligned with Bloque C architecture):
  APPROACH   — NLG far from platform; only coarse guidance active
  ALIGN      — NLG within alignment zone; fine lateral + heading correction
  CAPTURE    — NLG inside capture funnel; ramp contact imminent
  HOLD       — NLG seated; guidance complete

Transition logic is purely geometric and driven by the current pose
estimates (X distance, |Y| lateral error, |ψ| heading error).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar


class PhaseState(Enum):
    APPROACH = auto()
    ALIGN    = auto()
    CAPTURE  = auto()
    HOLD     = auto()


# ---------------------------------------------------------------------------
# Threshold bundle — can be overridden per aircraft preset
# ---------------------------------------------------------------------------

@dataclass
class PhaseThresholds:
    # Distance from NLG midpoint to capture mouth [m]
    x_align_m: float   = 1.50   # APPROACH → ALIGN  when X < x_align_m
    x_capture_m: float = 0.40   # ALIGN    → CAPTURE when X < x_capture_m
    x_hold_m: float    = 0.05   # CAPTURE  → HOLD    when X < x_hold_m

    # Lateral error [m] — must be inside window to advance
    y_align_m: float   = 0.30   # max |Y| to enter ALIGN
    y_capture_m: float = 0.12   # max |Y| to enter CAPTURE

    # Heading error [rad]
    psi_align_rad: float   = math.radians(8.0)   # max |ψ| to enter ALIGN
    psi_capture_rad: float = math.radians(3.0)   # max |ψ| to enter CAPTURE


def phase_thresholds(**kwargs) -> PhaseThresholds:
    """Factory with optional overrides."""
    return PhaseThresholds(**kwargs)


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------

@dataclass
class ApproachPhase:
    """Stateful FSM that tracks approach phase transitions.

    Parameters
    ----------
    thresholds:
        Override default transition thresholds.

    Usage
    -----
        fsm = ApproachPhase()
        state = fsm.update(x_to_capture=2.1, Y_m=0.08, psi_rad=0.04)
    """
    thresholds: PhaseThresholds = field(default_factory=PhaseThresholds)
    _state: PhaseState = field(default=PhaseState.APPROACH, init=False, repr=False)
    _history: list[PhaseState] = field(default_factory=list, init=False, repr=False)

    # ---------- class-level display tables (ClassVar — NOT dataclass fields) ----------

    PHASE_LABELS: ClassVar[dict[PhaseState, str]] = {
        PhaseState.APPROACH: "Fase 1 · APPROACH",
        PhaseState.ALIGN:    "Fase 2 · ALIGN",
        PhaseState.CAPTURE:  "Fase 3 · CAPTURE",
        PhaseState.HOLD:     "✅ HOLD",
    }

    PHASE_COLORS: ClassVar[dict[PhaseState, str]] = {
        PhaseState.APPROACH: "#c57b2b",   # orange — caution
        PhaseState.ALIGN:    "#3b6ea8",   # blue — active guidance
        PhaseState.CAPTURE:  "#437a22",   # green — imminent
        PhaseState.HOLD:     "#01696f",   # teal — complete
    }

    PHASE_DESCRIPTIONS: ClassVar[dict[PhaseState, str]] = {
        PhaseState.APPROACH: (
            "El NLG se aproxima. Sólo guiado grueso activo. "
            "Espera a que X, Y y ψ entren en ventana de alineación."
        ),
        PhaseState.ALIGN: (
            "Zona de alineación. Corrección lateral y angular fina activa. "
            "Avanza cuando X < umbral de captura y errores dentro de margen."
        ),
        PhaseState.CAPTURE: (
            "Embocadura alcanzada. Contacto con rampa inminente. "
            "Sensores cortos de validación fina en acción."
        ),
        PhaseState.HOLD: (
            "NLG asentado en plataforma. Guiado completo. "
            "Companion ECU activa freno + bloqueo de raíl."
        ),
    }

    # ---------- public interface ----------

    @property
    def state(self) -> PhaseState:
        return self._state

    def reset(self) -> None:
        self._state = PhaseState.APPROACH
        self._history.clear()

    def update(
        self,
        x_to_capture: float,
        Y_m: float,
        psi_rad: float,
    ) -> PhaseState:
        """Evaluate current pose and advance (or hold) the phase.

        Parameters
        ----------
        x_to_capture : float
            Distance along the approach axis from NLG midpoint to the
            capture mouth of the platform [m].  Always positive.
        Y_m : float
            Lateral offset estimate from the estimator [m].
        psi_rad : float
            Heading error estimate [rad].

        Returns
        -------
        PhaseState
            The (possibly updated) current state.
        """
        t = self.thresholds
        abs_Y = abs(Y_m)
        abs_psi = abs(psi_rad)

        prev = self._state

        if self._state == PhaseState.APPROACH:
            if (
                x_to_capture < t.x_align_m
                and abs_Y < t.y_align_m
                and abs_psi < t.psi_align_rad
            ):
                self._state = PhaseState.ALIGN

        elif self._state == PhaseState.ALIGN:
            if (
                x_to_capture < t.x_capture_m
                and abs_Y < t.y_capture_m
                and abs_psi < t.psi_capture_rad
            ):
                self._state = PhaseState.CAPTURE
            elif x_to_capture >= t.x_align_m or abs_Y >= t.y_align_m:
                self._state = PhaseState.APPROACH   # regress

        elif self._state == PhaseState.CAPTURE:
            if x_to_capture < t.x_hold_m:
                self._state = PhaseState.HOLD
            elif x_to_capture >= t.x_capture_m:
                self._state = PhaseState.ALIGN      # regress

        elif self._state == PhaseState.HOLD:
            pass  # terminal state — only reset() can exit HOLD

        if self._state != prev:
            self._history.append(prev)

        return self._state

    # ---------- display helpers ----------

    def label(self) -> str:
        return self.PHASE_LABELS[self._state]

    def color(self) -> str:
        return self.PHASE_COLORS[self._state]

    def description(self) -> str:
        return self.PHASE_DESCRIPTIONS[self._state]

    def progress(self) -> float:
        """Normalised progress 0..1 through the four phases."""
        return {
            PhaseState.APPROACH: 0.15,
            PhaseState.ALIGN:    0.50,
            PhaseState.CAPTURE:  0.80,
            PhaseState.HOLD:     1.00,
        }[self._state]
