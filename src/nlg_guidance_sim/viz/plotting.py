from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from nlg_guidance_sim.sensors.rplidar2d import ScanResult
from nlg_guidance_sim.world.scene import Scene


BG = "#f6f4ef"
SURFACE = "#fbfaf7"
TEXT = "#2a261f"
MUTED = "#76726a"
PRIMARY = "#01696f"
CYAN = "#17a2b8"
GREEN = "#2e7d32"
ORANGE = "#c57b2b"
BLUE = "#3b6ea8"
RED = "#b94e48"
GRID = "#cfc9bf"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color("#c9c3b9")
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.7)


def plot_scene(scene: Scene, scan_result: ScanResult | None = None):
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=140)
    fig.patch.set_facecolor(BG)
    _style_axes(ax)
    ax.set_aspect("equal")

    x0 = scene.rail_start_x_m
    x1 = scene.rail_length_m

    ax.plot([x0, x1], [scene.left_rail_y_m, scene.left_rail_y_m], "--", color="#8f8a81", linewidth=1.8)
    ax.plot([x0, x1], [scene.right_rail_y_m, scene.right_rail_y_m], "--", color="#8f8a81", linewidth=1.8)
    ax.plot([x0, x1], [scene.guide_axis_y_m, scene.guide_axis_y_m], ":", color=MUTED, linewidth=1.4)

    body = Polygon(
        scene.platform_body_outline(),
        closed=True,
        facecolor="#ddd8cf",
        edgecolor="#625d56",
        linewidth=1.8,
    )
    ramp = Polygon(
        scene.ramp_outline(),
        closed=True,
        facecolor="#e8e2d8",
        edgecolor="#625d56",
        linewidth=1.8,
    )
    ax.add_patch(body)
    ax.add_patch(ramp)

    for contour in scene.nlg.wheel_contours_world():
        poly = Polygon(
            contour,
            closed=True,
            facecolor="#dce7ef",
            edgecolor=BLUE,
            linewidth=2.0,
            alpha=0.95,
        )
        ax.add_patch(poly)

    centers = scene.nlg.wheel_centers()
    midpoint = scene.nlg.midpoint
    for c in centers:
        ax.scatter(c[0], c[1], color=BLUE, s=32, zorder=5)

    if len(centers) == 2:
        ax.plot(
            [centers[0][0], centers[1][0]],
            [centers[0][1], centers[1][1]],
            color=GREEN,
            linewidth=2.2,
        )

    ax.scatter(midpoint[0], midpoint[1], color=RED, s=42, zorder=6)
    heading = scene.nlg.heading_vector()
    ax.arrow(
        midpoint[0],
        midpoint[1],
        0.75 * heading[0],
        0.75 * heading[1],
        width=0.008,
        head_width=0.07,
        head_length=0.12,
        color=RED,
        length_includes_head=True,
        zorder=6,
    )

    if scan_result is not None:
        origin = scan_result.origin_xy
        ax.scatter(origin[0], origin[1], color=PRIMARY, s=55, zorder=7)
        ax.annotate(
            "LiDAR 2D",
            xy=(origin[0], origin[1]),
            xytext=(origin[0] - 0.22, origin[1] + 0.18),
            arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.0),
            fontsize=10,
            color=TEXT,
        )

        stride = max(1, len(scan_result.angles_rad) // 60)
        for i in range(0, len(scan_result.angles_rad), stride):
            p = scan_result.points_xy[i]
            color = CYAN if scan_result.hit_mask[i] else "#b9dfe4"
            ax.plot([origin[0], p[0]], [origin[1], p[1]], color=color, alpha=0.15, linewidth=0.9)

        hits = scan_result.ordered_hit_points()
        if len(hits) > 0:
            ax.scatter(hits[:, 0], hits[:, 1], s=10, color=CYAN, alpha=0.9, zorder=6)

    ax.annotate(
        "Eje de gu\u00eda",
        xy=(scene.rail_length_m - 0.7, scene.guide_axis_y_m + 0.03),
        fontsize=10,
        color=TEXT,
    )
    ax.annotate(
        "Embocadura / cuna",
        xy=(scene.capture_x_m, 0.0),
        xytext=(scene.capture_x_m + 0.20, 0.22),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=ORANGE),
        fontsize=10,
        color=TEXT,
    )
    ax.annotate(
        f"Y = {midpoint[1]:.3f} m",
        xy=(midpoint[0], midpoint[1]),
        xytext=(midpoint[0] + 0.25, midpoint[1] + 0.23),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=MUTED),
        fontsize=10,
        color=TEXT,
    )
    ax.annotate(
        f"\u03c8 = {math.degrees(scene.nlg.psi_rad):.2f} deg",
        xy=(midpoint[0] + 0.18, midpoint[1] - 0.14),
        fontsize=10,
        color=TEXT,
    )

    ax.set_title("Escena 2D param\u00e9trica del NLG y plataforma guiada por ra\u00edles", fontsize=13)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(scene.rail_start_x_m - 0.1, scene.rail_length_m + 0.4)
    ax.set_ylim(-1.25, 1.25)
    return fig


def plot_ranges(scan_result: ScanResult):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=140)
    fig.patch.set_facecolor(BG)

    ax0, ax1 = axes
    _style_axes(ax0)
    _style_axes(ax1)

    ax0.set_aspect("equal")
    ax0.scatter(
        scan_result.points_xy[:, 0],
        scan_result.points_xy[:, 1],
        s=10,
        c=np.where(scan_result.hit_mask, CYAN, "#d7d2c8"),
        alpha=0.9,
    )
    ax0.scatter(scan_result.origin_xy[0], scan_result.origin_xy[1], s=50, color=PRIMARY)
    ax0.set_title("Puntos del LiDAR en XY")
    ax0.set_xlabel("X [m]")
    ax0.set_ylabel("Y [m]")

    angles_deg = np.rad2deg(scan_result.angles_rad)
    ax1.plot(angles_deg, scan_result.ranges_m, color=PRIMARY, linewidth=1.6)
    ax1.set_title("Rango por \u00e1ngulo")
    ax1.set_xlabel("\u00c1ngulo [deg]")
    ax1.set_ylabel("Distancia [m]")

    return fig
