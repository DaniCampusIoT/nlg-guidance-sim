from nlg_guidance_sim.world.scene import Scene
from nlg_guidance_sim.estimation.y_psi_solver import solve_y_psi


def main() -> None:
    scene = Scene()
    result = solve_y_psi(scene)
    print(result)


if __name__ == "__main__":
    main()
