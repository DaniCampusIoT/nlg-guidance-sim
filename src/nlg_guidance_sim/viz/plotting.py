from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from nlg_guidance_sim.estimation.lshape import LShapeResult
from nlg_guidance_sim.estimation.pose import PoseEstimate
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
PURPLE = "#7a39bb"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color("#c9c3b9")
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.7)


def _draw_lshape_overlay(
    ax,
    result: LShapeResult,
    pose: PoseEstimate | None = None,
) -> None:
    """Draw L-shape arms, corner and (optionally) pose annotation."""
    # arm 1
    if len(result.pts1) >= 2:
        ax.plot(
            result.pts1[:, 0], result.pts1[:, 1],
            color=PURPLE, linewidth=2.0, linestyle="--", alpha=0.85,
            label="Brazo 1 (TLS)",
        )
    # arm 2
    if len(result.pts2) >= 2:
        ax.plot(
            result.pts2[:, 0], result.pts2[:, 1],
            color=ORANGE, linewidth=2.0, linestyle="--", alpha=0.85,
            label="Brazo 2 (TLS)",
        )
    # corner
    cx, cy = result.corner_xy
    ax.scatter(cx, cy, color=RED, s=90, zorder=8, marker="x", linewidths=2.5)
    ax.annotate(
        "Corner L",
        xy=(cx, cy),
        xytext=(cx + 0.18, cy + 0.18),
        arrowprops=dict(arrowstyle="->", color=RED, lw=0.9),
        fontsize=9,
        color=RED,
    )

    if pose is not None:
        ax.annotate(
            f"Y_est = {pose.Y_m:+.3f} m\n"
            f"\u03c8_est = {pose.psi_deg:+.2f}\u00b0\n"
            f"conf = {pose.confidence:.2f}",
            xy=(cx, cy),
            xytext=(cx - 0.80, cy - 0.35),
            fontsize=9,
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffdf7",
                      edgecolor="#d4cfc8", linewidth=0.8),
        )


def plot_scene(
    scene: Scene,
    scan_result: ScanResult | None = None,
    lshape_result: LShapeResult | None = None,
    pose_estimate: PoseEstimate | None = None,
):
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=140)
    fig.patch.set_facecolor(BG)
    _style_axes(ax)
    ax.set_aspect("equal")

    x0 = scene.rail_start_x_m
    x1 = scene.rail_length_m

    ax.plot([x0, x1], [scene.left_rail_y_m] * 2, "--", color="#8f8a81", linewidth=1.8)
    ax.plot([x0, x1], [scene.right_rail_y_m] * 2, "--", color="#8f8a81", linewidth=1.8)
    ax.plot([x0, x1], [scene.guide_axis_y_m] * 2, ":", color=MUTED, linewidth=1.4)

    body = Polygon(
        scene.platform_body_outline(), closed=True,
        facecolor="#ddd8cf", edgecolor="#625d56", linewidth=1.8,
    )
    ramp = Polygon(
        scene.ramp_outline(), closed=True,
        facecolor="#e8e2d8", edgecolor="#625d56", linewidth=1.8,
    )
    ax.add_patch(body)
    ax.add_patch(ramp)

    for contour in scene.nlg.wheel_contours_world():
        poly = Polygon(
            contour, closed=True,
            facecolor="#dce7ef", edgecolor=BLUE, linewidth=2.0, alpha=0.95,
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
            color=GREEN, linewidth=2.2,
        )

    ax.scatter(midpoint[0], midpoint[1], color=RED, s=42, zorder=6)
    heading = scene.nlg.heading_vector()
    ax.arrow(
        midpoint[0], midpoint[1],
        0.75 * heading[0], 0.75 * heading[1],
        width=0.008, head_width=0.07, head_length=0.12,
        color=RED, length_includes_head=True, zorder=6,
    )

    if scan_result is not None:
        origin = scan_result.origin_xy
        ax.scatter(origin[0], origin[1], color=PRIMARY, s=55, zorder=7)
        ax.annotate(
            "LiDAR 2D",
            xy=(origin[0], origin[1]),
            xytext=(origin[0] - 0.22, origin[1] + 0.18),
            arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=1.0),
            fontsize=10, color=TEXT,
        )
        stride = max(1, len(scan_result.angles_rad) // 60)
        for i in range(0, len(scan_result.angles_rad), stride):
            p = scan_result.points_xy[i]
            color = CYAN if scan_result.hit_mask[i] else "#b9dfe4"
            ax.plot([origin[0], p[0]], [origin[1], p[1]],
                    color=color, alpha=0.15, linewidth=0.9)
        hits = scan_result.ordered_hit_points()
        if len(hits) > 0:
            ax.scatter(hits[:, 0], hits[:, 1], s=10, color=CYAN, alpha=0.9, zorder=6)

    if lshape_result is not None:
        _draw_lshape_overlay(ax, lshape_result, pose=pose_estimate)
        ax.legend(loc="upper right", fontsize=9,
                  framealpha=0.85, edgecolor="#d4cfc8")

    ax.annotate(
        "Eje de gu\u00eda",
        xy=(scene.rail_length_m - 0.7, scene.guide_axis_y_m + 0.03),
        fontsize=10, color=TEXT,
    )
    ax.annotate(
        "Embocadura / cuna",
        xy=(scene.capture_x_m, 0.0),
        xytext=(scene.capture_x_m + 0.20, 0.22),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=ORANGE),
        fontsize=10, color=TEXT,
    )
    ax.annotate(
        f"Y = {midpoint[1]:.3f} m",
        xy=(midpoint[0], midpoint[1]),
        xytext=(midpoint[0] + 0.25, midpoint[1] + 0.23),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=MUTED),
        fontsize=10, color=TEXT,
    )
    ax.annotate(
        f"\u03c8 = {math.degrees(scene.nlg.psi_rad):.2f} deg",
        xy=(midpoint[0] + 0.18, midpoint[1] - 0.14),
        fontsize=10, color=TEXT,
    )

    ax.set_title(
        "Escena 2D param\u00e9trica del NLG y plataforma guiada por ra\u00edles",
        fontsize=13,
    )
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
        scan_result.points_xy[:, 0], scan_result.points_xy[:, 1],
        s=10,
        c=np.where(scan_result.hit_mask, CYAN, "#d7d2c8"),
        alpha=0.9,
    )
    ax0.scatter(scan_result.origin_xy[0], scan_result.origin_xy[1],
                s=50, color=PRIMARY)
    ax0.set_title("Puntos del LiDAR en XY")
    ax0.set_xlabel("X [m]")
    ax0.set_ylabel("Y [m]")

    angles_deg = np.rad2deg(scan_result.angles_rad)
    ax1.plot(angles_deg, scan_result.ranges_m, color=PRIMARY, linewidth=1.6)
    ax1.set_title("Rango por \u00e1ngulo")
    ax1.set_xlabel("\u00c1ngulo [deg]")
    ax1.set_ylabel("Distancia [m]")
    return fig


def plot_estimation_debug(
    scan_result: ScanResult,
    lshape_result: LShapeResult,
    pose_estimate: PoseEstimate,
):
    """Compact 2-panel debug figure for the estimation pipeline."""
    fig, (ax_xy, ax_rng) = plt.subplots(1, 2, figsize=(13.2, 5.6), dpi=130)
    fig.patch.set_facecolor(BG)
    _style_axes(ax_xy)
    _style_axes(ax_rng)

    # ---- XY panel ----
    ax_xy.set_aspect("equal")
    hits = scan_result.ordered_hit_points()
    ax_xy.scatter(hits[:, 0], hits[:, 1], s=14, color=CYAN, alpha=0.85, label="Hit pts")
    ax_xy.scatter(*scan_result.origin_xy, s=55, color=PRIMARY, zorder=6)

    _draw_lshape_overlay(ax_xy, lshape_result, pose=pose_estimate)

    ax_xy.set_title("Nube 2D + L-shape fit", fontsize=11)
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # ---- Residuals panel ----
    all_pts = np.vstack([lshape_result.pts1, lshape_result.pts2])
    colors_res = ([PURPLE] * len(lshape_result.pts1)
                  + [ORANGE] * len(lshape_result.pts2))
    idx = np.arange(len(all_pts))
    res1 = lshape_result.line1.residuals(lshape_result.pts1)
    res2 = lshape_result.line2.residuals(lshape_result.pts2)
    all_res = np.concatenate([res1, res2])
    ax_rng.bar(idx, all_res * 1000, color=colors_res, alpha=0.8, width=0.9)
    ax_rng.axhline(0, color=MUTED, linewidth=0.8)
    ax_rng.set_title("Residuos TLS por punto [mm]", fontsize=11)
    ax_rng.set_xlabel("\u00cdndice de punto")
    ax_rng.set_ylabel("Residuo [mm]")
    split = len(lshape_result.pts1)
    ax_rng.axvline(split - 0.5, color=RED, linewidth=1.2, linestyle=":")
    ax_rng.annotate(f"split={split}", xy=(split, ax_rng.get_ylim()[0]),
                    xytext=(split + 1, 0), fontsize=8, color=RED)

    fig.suptitle(
        f"L-shape debug  |  RMS tot = {lshape_result.total_rms*1000:.2f} mm  "
        f"| \u00e1ngulo esquina = {lshape_result.corner_angle_deg:.1f}\u00b0",
        fontsize=11, color=TEXT,
    )
    return fig
