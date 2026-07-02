from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from nlg_guidance_sim.sensors.rplidar2d import ScanResult
from nlg_guidance_sim.world.scene import Scene

try:
    from nlg_guidance_sim.estimation.lshape import LShapeFitResult
except ImportError:
    LShapeFitResult = None  # type: ignore

BG      = "#f6f4ef"
SURFACE = "#fbfaf7"
TEXT    = "#2a261f"
MUTED   = "#76726a"
PRIMARY = "#01696f"
CYAN    = "#17a2b8"
GREEN   = "#2e7d32"
ORANGE  = "#c57b2b"
BLUE    = "#3b6ea8"
RED     = "#b94e48"
GRID    = "#cfc9bf"
PURPLE  = "#7a39bb"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color("#c9c3b9")
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.7)


def plot_scene(
    scene: Scene,
    scan_result: ScanResult | None = None,
    est_result=None,      # LShapeFitResult | None
):
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=140)
    fig.patch.set_facecolor(BG)
    _style_axes(ax)
    ax.set_aspect("equal")

    x0 = scene.rail_start_x_m
    x1 = scene.rail_length_m

    ax.plot([x0, x1], [scene.left_rail_y_m]*2,  "--", color="#8f8a81", linewidth=1.8)
    ax.plot([x0, x1], [scene.right_rail_y_m]*2, "--", color="#8f8a81", linewidth=1.8)
    ax.plot([x0, x1], [scene.guide_axis_y_m]*2,  ":", color=MUTED,     linewidth=1.4)

    body = Polygon(scene.platform_body_outline(), closed=True,
                   facecolor="#ddd8cf", edgecolor="#625d56", linewidth=1.8)
    ramp = Polygon(scene.ramp_outline(),          closed=True,
                   facecolor="#e8e2d8", edgecolor="#625d56", linewidth=1.8)
    ax.add_patch(body)
    ax.add_patch(ramp)

    for contour in scene.nlg.wheel_contours_world():
        ax.add_patch(Polygon(contour, closed=True, facecolor="#dce7ef",
                             edgecolor=BLUE, linewidth=2.0, alpha=0.95))

    centers  = scene.nlg.wheel_centers()
    midpoint = scene.nlg.midpoint
    for c in centers:
        ax.scatter(c[0], c[1], color=BLUE, s=32, zorder=5)
    if len(centers) == 2:
        ax.plot([centers[0][0], centers[1][0]],
                [centers[0][1], centers[1][1]], color=GREEN, linewidth=2.2)

    ax.scatter(midpoint[0], midpoint[1], color=RED, s=42, zorder=6)
    heading = scene.nlg.heading_vector()
    ax.arrow(midpoint[0], midpoint[1], 0.75*heading[0], 0.75*heading[1],
             width=0.008, head_width=0.07, head_length=0.12,
             color=RED, length_includes_head=True, zorder=6)

    if scan_result is not None:
        origin = scan_result.origin_xy
        ax.scatter(origin[0], origin[1], color=PRIMARY, s=55, zorder=7)
        ax.annotate("LiDAR 2D", xy=(origin[0], origin[1]),
                    xytext=(origin[0]-0.22, origin[1]+0.18),
                    arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.0),
                    fontsize=10, color=TEXT)
        stride = max(1, len(scan_result.angles_rad) // 60)
        for i in range(0, len(scan_result.angles_rad), stride):
            p = scan_result.points_xy[i]
            color = CYAN if scan_result.hit_mask[i] else "#b9dfe4"
            ax.plot([origin[0], p[0]], [origin[1], p[1]],
                    color=color, alpha=0.15, linewidth=0.9, clip_on=True)
        hits = scan_result.ordered_hit_points()
        if len(hits) > 0:
            ax.scatter(hits[:, 0], hits[:, 1], s=10, color=CYAN, alpha=0.9, zorder=6)

    # ── Overlay L-shape estimation ──
    if est_result is not None:
        corner = est_result.corner_xy
        ax.scatter(corner[0], corner[1], color=PURPLE, s=70, marker="*", zorder=8,
                   label="Esquina L estimada")
        if len(est_result.seg1_pts) > 1:
            ax.plot(est_result.seg1_pts[:, 0], est_result.seg1_pts[:, 1],
                    color=PURPLE, linewidth=1.8, alpha=0.65, label="Brazo 1")
        if len(est_result.seg2_pts) > 1:
            ax.plot(est_result.seg2_pts[:, 0], est_result.seg2_pts[:, 1],
                    color=ORANGE, linewidth=1.8, alpha=0.65, label="Brazo 2")
        ax.annotate(
            f"Ŷ={est_result.Y_m:+.3f} m  ψ̂={est_result.psi_deg:+.2f}°",
            xy=(corner[0], corner[1]),
            xytext=(corner[0]+0.12, corner[1]-0.22),
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.9),
            fontsize=9, color=PURPLE,
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    ax.annotate("Eje de guía",
                xy=(scene.rail_length_m-0.7, scene.guide_axis_y_m+0.03),
                fontsize=10, color=TEXT)
    ax.annotate("Embocadura / cuna", xy=(scene.capture_x_m, 0.0),
                xytext=(scene.capture_x_m+0.20, 0.22),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=ORANGE), fontsize=10, color=TEXT)
    ax.annotate(f"Y = {midpoint[1]:.3f} m", xy=(midpoint[0], midpoint[1]),
                xytext=(midpoint[0]+0.25, midpoint[1]+0.23),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=MUTED), fontsize=10, color=TEXT)
    ax.annotate(f"ψ = {math.degrees(scene.nlg.psi_rad):.2f} deg",
                xy=(midpoint[0]+0.18, midpoint[1]-0.14), fontsize=10, color=TEXT)

    ax.set_title("Escena 2D paramétrica del NLG y plataforma guiada por raíles", fontsize=13)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # ── Límites dinámicos ──────────────────────────────────────────────────────
    # X izquierdo: siempre desde el inicio de los raíles
    x_left = scene.rail_start_x_m - 0.2

    if scan_result is not None:
        lidar_x   = float(scan_result.origin_xy[0])
        # Alcance máximo configurado = valor máximo del array de rangos
        # (los rayos sin impacto se almacenan a max_range_m)
        max_range = float(scan_result.ranges_m.max())
        # El rayo central (0°) apunta en +X; tomamos ese como la frontera derecha del cono
        angle_min = float(scan_result.angles_rad[0])
        angle_max = float(scan_result.angles_rad[-1])
        angle_mid = (angle_min + angle_max) / 2.0
        x_lidar_reach = lidar_x + max_range * math.cos(angle_mid)
        x_right = max(scene.nlg.center_x_m + 1.5, x_lidar_reach + 0.5)
    else:
        x_right = scene.nlg.center_x_m + 1.5

    ax.set_xlim(x_left, x_right)

    # Y proporcional al ancho de la vista para no achatar la escena,
    # con un mínimo de ±1.25 m para siempre ver los raíles
    scene_width = x_right - x_left
    half_y = max(1.25, scene_width * 0.12)
    ax.set_ylim(-half_y, half_y)

    return fig


def plot_ranges(scan_result: ScanResult):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=140)
    fig.patch.set_facecolor(BG)
    ax0, ax1 = axes
    _style_axes(ax0)
    _style_axes(ax1)

    ax0.set_aspect("equal")
    ax0.scatter(scan_result.points_xy[:, 0], scan_result.points_xy[:, 1],
                s=10, c=np.where(scan_result.hit_mask, CYAN, "#d7d2c8"), alpha=0.9)
    ax0.scatter(scan_result.origin_xy[0], scan_result.origin_xy[1], s=50, color=PRIMARY)
    ax0.set_title("Puntos del LiDAR en XY")
    ax0.set_xlabel("X [m]")
    ax0.set_ylabel("Y [m]")

    angles_deg = np.rad2deg(scan_result.angles_rad)
    ax1.plot(angles_deg, scan_result.ranges_m, color=PRIMARY, linewidth=1.6)
    ax1.set_title("Rango por ángulo")
    ax1.set_xlabel("Ángulo [deg]")
    ax1.set_ylabel("Distancia [m]")
    return fig


def plot_estimation(scene: Scene, est):
    """Dedicated figure for the Estimación tab: hit points + L-shape fit."""
    fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=130)
    fig.patch.set_facecolor(BG)
    _style_axes(ax)
    ax.set_aspect("equal")

    # Reference ground truth
    for contour in scene.nlg.wheel_contours_world():
        ax.add_patch(Polygon(contour, closed=True, facecolor="#dce7ef",
                             edgecolor=BLUE, linewidth=1.6, alpha=0.55,
                             label="Rueda (GT)"))

    # Estimated segments
    if len(est.seg1_pts) > 1:
        ax.scatter(est.seg1_pts[:, 0], est.seg1_pts[:, 1],
                   s=22, color=PURPLE, alpha=0.85, label="Brazo 1 (est.)")
        p1 = est.seg1_pts
        ax.plot([p1[0, 0], p1[-1, 0]], [p1[0, 1], p1[-1, 1]],
                color=PURPLE, linewidth=2.2, alpha=0.7)
    if len(est.seg2_pts) > 1:
        ax.scatter(est.seg2_pts[:, 0], est.seg2_pts[:, 1],
                   s=22, color=ORANGE, alpha=0.85, label="Brazo 2 (est.)")
        p2 = est.seg2_pts
        ax.plot([p2[0, 0], p2[-1, 0]], [p2[0, 1], p2[-1, 1]],
                color=ORANGE, linewidth=2.2, alpha=0.7)

    # Estimated corner and heading arrow
    corner = est.corner_xy
    ax.scatter(corner[0], corner[1], color=PURPLE, s=120, marker="*", zorder=8)
    psi_vec = np.array([math.cos(est.psi_rad), math.sin(est.psi_rad)])
    ax.arrow(corner[0], corner[1], 0.5*psi_vec[0], 0.5*psi_vec[1],
             width=0.006, head_width=0.06, head_length=0.10,
             color=PURPLE, length_includes_head=True, zorder=8)

    # Ground truth midpoint
    mp = scene.nlg.midpoint
    ax.scatter(mp[0], mp[1], color=RED, s=60, zorder=7, label="GT midpoint")

    ax.set_title(
        f"Estimación L-shape · Ŷ={est.Y_m:+.4f} m · "
        f"ψ̂={est.psi_deg:+.3f}° · RMSE={est.rmse_m*100:.2f} cm · "
        f"método={est.method}",
        fontsize=11,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    xc = corner[0]
    ax.set_xlim(xc - 1.2, xc + 1.2)
    ax.set_ylim(-0.9, 0.9)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    return fig
