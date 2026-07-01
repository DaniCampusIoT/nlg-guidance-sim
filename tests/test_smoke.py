from nlg_guidance_sim.world.scene import Scene
from nlg_guidance_sim.estimation.y_psi_solver import solve_y_psi


def test_smoke_pipeline() -> None:
    scene = Scene()
    result = solve_y_psi(scene)
    assert scene.name == "nominal_scene"
    assert result.confidence == 0.0
