from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.phases.state import ApproachPhase, PhaseState


def test_initial_state():
    fsm = ApproachPhase()
    assert fsm.state == PhaseState.APPROACH


def test_advance_to_align():
    fsm = ApproachPhase()
    state = fsm.update(x_to_capture=1.2, Y_m=0.05, psi_rad=math.radians(3.0))
    assert state == PhaseState.ALIGN


def test_advance_to_capture():
    fsm = ApproachPhase()
    fsm.update(x_to_capture=1.2, Y_m=0.05, psi_rad=math.radians(3.0))
    state = fsm.update(x_to_capture=0.30, Y_m=0.03, psi_rad=math.radians(1.5))
    assert state == PhaseState.CAPTURE


def test_advance_to_hold():
    fsm = ApproachPhase()
    fsm.update(x_to_capture=1.2, Y_m=0.05, psi_rad=math.radians(3.0))
    fsm.update(x_to_capture=0.30, Y_m=0.03, psi_rad=math.radians(1.5))
    state = fsm.update(x_to_capture=0.03, Y_m=0.01, psi_rad=math.radians(0.5))
    assert state == PhaseState.HOLD


def test_regression_to_approach():
    fsm = ApproachPhase()
    fsm.update(x_to_capture=1.2, Y_m=0.05, psi_rad=math.radians(3.0))
    state = fsm.update(x_to_capture=2.0, Y_m=0.50, psi_rad=math.radians(12.0))
    assert state == PhaseState.APPROACH


def test_reset():
    fsm = ApproachPhase()
    fsm.update(x_to_capture=1.2, Y_m=0.05, psi_rad=math.radians(3.0))
    fsm.update(x_to_capture=0.30, Y_m=0.03, psi_rad=math.radians(1.5))
    fsm.reset()
    assert fsm.state == PhaseState.APPROACH


def test_labels_and_colors():
    fsm = ApproachPhase()
    assert "Fase 1" in fsm.label()
    assert fsm.color().startswith("#")
    assert len(fsm.description()) > 10
