from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.world.scene import Scene
from nlg_guidance_sim.viz.plot_scene import plot_scene


def main() -> None:
    scene = Scene()
    print(scene.summary())
    output_path = plot_scene(scene, save=True, show=False)
    if output_path is not None:
        print(f"Figura guardada en: {output_path}")


if __name__ == "__main__":
    main()
