from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.catalog.presets import PRESETS, load_presets_yaml
from nlg_guidance_sim.geometry.nlg_model import NLGModel
from nlg_guidance_sim.world.scene import Scene
from nlg_guidance_sim.sensors.rplidar2d import RPLidar2DSim
from nlg_guidance_sim.estimation.lshape import LShapeFitter
from nlg_guidance_sim.estimation.pose import estimate_pose
from nlg_guidance_sim.control.phases import classify_phase, PhaseThresholds


def test_presets_exist():
    assert len(PRESETS) >= 3


def test_single_wheel_model():
    model = NLGModel(arrangement="single")
    assert model.wheel_count() == 1


def test_scene_capture_position():
    scene = Scene()
    assert scene.capture_x_m > scene.platform_front_x_m


def test_lshape_fitter_returns_result():
    """End-to-end: scan a default scene and fit an L-shape."""
    scene = Scene()
    lidar = RPLidar2DSim(num_beams=360, angle_min_deg=-80.0, angle_max_deg=80.0)
    scan = lidar.scan(scene)
    hits = scan.ordered_hit_points()
    assert len(hits) > 0, "LiDAR must see something"
    fitter = LShapeFitter(min_pts_per_arm=5)
    result = fitter.fit(hits)
    assert result is not None, "Fitter must return a result for default scene"
    assert result.total_rms < 0.1


def test_pose_estimate():
    scene = Scene()
    lidar = RPLidar2DSim(num_beams=360, angle_min_deg=-80.0, angle_max_deg=80.0)
    scan = lidar.scan(scene)
    hits = scan.ordered_hit_points()
    fitter = LShapeFitter()
    result = fitter.fit(hits)
    assert result is not None
    pose = estimate_pose(result)
    assert isinstance(pose.Y_m, float)
    assert isinstance(pose.psi_deg, float)
    assert 0.0 <= pose.confidence <= 1.0


def test_phase_classification():
    th = PhaseThresholds()
    # Far away → Phase 1
    p = classify_phase(x_m=5.0)
    assert p.phase.value == "1"
    # Close + confident → Phase 2
    p = classify_phase(x_m=2.0, lidar_confidence=0.8)
    assert p.phase.value == "2"
    # Inside capture zone, aligned → Phase 3
    p = classify_phase(x_m=0.5, Y_m=0.01, psi_deg=1.0, lidar_confidence=0.9)
    assert p.phase.value == "3"


def test_yaml_preset_roundtrip():
    """Built-in presets can be serialised and reloaded."""
    import io
    import yaml
    presets = PRESETS
    data = {"presets": [v.to_yaml_dict() for v in presets.values()]}
    text = yaml.dump(data, allow_unicode=True, sort_keys=False)
    reloaded = load_presets_yaml.__wrapped__ if hasattr(load_presets_yaml, '__wrapped__') else load_presets_yaml
    # parse manually to avoid file-path lookup
    raw = yaml.safe_load(text)
    from nlg_guidance_sim.catalog.presets import AircraftPreset
    result = {e["name"]: AircraftPreset(**e) for e in raw["presets"]}
    assert set(result.keys()) == set(presets.keys())
