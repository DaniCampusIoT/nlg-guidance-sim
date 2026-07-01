from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from nlg_guidance_sim.world.scene import Scene


OUTPUT_DIR = Path("data/generated")


def _platform_polygon(scene: Scene) -> Polygon:
    x0 = scene.platform_front_x_m
    x1 = scene.platform_front_x_m + scene.platform_length_m
    half_w = scene.capture_width_m / 2.0
    points = [
        [x0, -half_w],
        [x1, -half_w],
        [x1, half_w],
        [x0, half_w],
    ]
    return Polygon(points, closed=True, fill=False)


def plot_scene(scene: Scene, save: bool = True, show: bool = False) -> Path | None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_aspect("equal")

    ax.plot(
        [0.0, scene.rail_length_m],
        [scene.left_rail_y_m, scene.left_rail_y_m],
        linestyle="--",
        linewidth=2,
        color="gray",
    )
    ax.plot(
        [0.0, scene.rail_length_m],
        [scene.right_rail_y_m, scene.right_rail_y_m],
        linestyle="--",
        linewidth=2,
        color="gray",
    )
    ax.plot(
        [0.0, scene.rail_length_m],
        [scene.guide_axis_y_m, scene.guide_axis_y_m],
        linestyle=":",
        linewidth=1.5,
        color="black",
    )

    platform = _platform_polygon(scene)
    platform.set_edgecolor("dimgray")
    platform.set_linewidth(2.0)
    ax.add_patch(platform)

    left_contour = scene.nlg.left_wheel_contour_world()
    right_contour = scene.nlg.right_wheel_contour_world()

    left_poly = Polygon(left_contour, closed=True, fill=False, edgecolor="tab:blue", linewidth=2.5)
    right_poly = Polygon(right_contour, closed=True, fill=False, edgecolor="tab:blue", linewidth=2.5)
    ax.add_patch(left_poly)
    ax.add_patch(right_poly)

    left_c = scene.nlg.left_wheel_center
    right_c = scene.nlg.right_wheel_center
    mid = scene.nlg.midpoint

    ax.plot([left_c[0], right_c[0]], [left_c[1], right_c[1]], color="tab:green", linewidth=2.0)
    ax.scatter(
        [left_c[0], right_c[0], mid[0]],
        [left_c[1], right_c[1], mid[1]],
        color=["tab:blue", "tab:blue", "tab:red"],
        zorder=5,
    )

    heading = scene.nlg.heading_vector()
    scale = 0.8
    ax.arrow(
        mid[0],
        mid[1],
        heading[0] * scale,
        heading[1] * scale,
        width=0.01,
        head_width=0.08,
        head_length=0.12,
        color="tab:red",
        length_includes_head=True,
    )

    ax.annotate(
        "Eje de guía",
        xy=(scene.rail_length_m * 0.85, scene.guide_axis_y_m + 0.02),
        fontsize=11,
    )
    ax.annotate(
        "Punto medio NLG",
        xy=(mid[0], mid[1]),
        xytext=(mid[0] + 0.15, mid[1] + 0.18),
        arrowprops=dict(arrowstyle="->", lw=1.2),
        fontsize=11,
    )
    ax.annotate(
        f"Y = {mid[1]:.3f} m",
        xy=(mid[0], mid[1] / 2.0),
        xytext=(mid[0] + 0.25, mid[1] / 2.0 + 0.15),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=11,
    )
    ax.annotate(
        f"psi = {math.degrees(scene.nlg.psi_rad):.2f} deg",
        xy=(mid[0] + 0.35, mid[1] + 0.02),
        fontsize=11,
    )

    ax.add_patch(
        Rectangle(
            (scene.capture_x_m - 0.08, -scene.capture_width_m / 2.0),
            0.08,
            scene.capture_width_m,
            fill=False,
            linestyle="-.",
            linewidth=1.8,
            edgecolor="tab:orange",
        )
    )
    ax.annotate(
        "Embocadura/captura",
        xy=(scene.capture_x_m, 0.0),
        xytext=(scene.capture_x_m + 0.15, 0.2),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=11,
    )

    ax.set_title("Escena 2D inicial con perfil paramétrico de neumático")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(-0.2, scene.rail_length_m + 0.8)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.25)

    output_path = OUTPUT_DIR / "scene_2d_nominal.png"
    if save:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path if save else None