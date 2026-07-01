from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.catalog.presets import PRESETS
from nlg_guidance_sim.geometry.nlg_model import NLGModel
from nlg_guidance_sim.world.scene import Scene


def test_presets_exist():
    assert len(PRESETS) >= 3


def test_single_wheel_model():
    model = NLGModel(arrangement="single")
    assert model.wheel_count() == 1


def test_scene_capture_position():
    scene = Scene()
    assert scene.capture_x_m > scene.platform_front_x_m
