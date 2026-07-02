"""Matplotlib helpers for the fitting pipeline visualisation."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import numpy as np

from nlg_guidance_sim.fitting.pipeline import FittingResult
from nlg_guidance_sim.estimation.pose_ekf import EKFState
from nlg_guidance_sim.world.scene import Scene

BG      = "#f6f4ef"
SURFACE = "#fbfaf7"
TEXT    = "#2a261f"
MUTED   = "#76726a"
PRIMARY = "#01696f"
CYAN    = "#17a2b8"
ORANGE  = "#c57b2b"
BLUE    = "#3b6ea8"
RED     = "#b94e48"
GREEN   = "#2e7d32"
GRID    = "#cfc9bf"


def _style_ax(ax) -> None:
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color("#c9c3b9")
    ax.tick_params(colors=MUTED)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.40, linewidth=0.7)


def plot_fitting(
    scene: Scene,
    fit_result: FittingResult,
) -> plt.Figure:
    """Overlay fitted rigid rectangle + RDP corner on the scene 2D view."""
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    ax.set_aspect("equal")

    x0 = scene.rail_start_x_m
    x1 = scene.rail_length_m
    ax.plot([x0, x1], [scene.left_rail_y_m]  * 2, "--", color="#8f8a81", lw=1.6, label="Raíles")
    ax.plot([x0, x1], [scene.right_rail_y_m] * 2, "--", color="#8f8a81", lw=1.6)
    ax.plot([x0, x1], [scene.guide_axis_y_m] * 2, ":",  color=MUTED, lw=1.2)

    # Platform
    from matplotlib.patches import Polygon as MPoly
    for outline, fc in [
        (scene.platform_body_outline(), "#ddd8cf"),
        (scene.ramp_outline(),          "#e8e2d8"),
    ]:
        ax.add_patch(MPoly(outline, closed=True, facecolor=fc,
                           edgecolor="#625d56", linewidth=1.6))

    # True wheel contours (ground-truth)
    for contour in scene.nlg.wheel_contours_world():
        ax.add_patch(MPoly(contour, closed=True, facecolor="#dce7ef",
                           edgecolor=BLUE, linewidth=1.6, alpha=0.60,
                           linestyle="--", label="Contorno real"))

    # LiDAR hit points
    pts = fit_result.pts_used
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=CYAN, alpha=0.85, zorder=5,
               label=f"Puntos LiDAR ({len(pts)})")

    # RDP seed corner
    ci = fit_result.seed_corner_idx
    ax.scatter(pts[ci, 0], pts[ci, 1], s=90, color=ORANGE, zorder=7,
               marker="^", label=f"RDP corner (θ₀={math.degrees(fit_result.seed_theta_rad):.1f}°)")

    # Fitted rectangle
    corners = fit_result.fit.rect_corners()
    rect_closed = np.vstack([corners, corners[0]])  # close polygon
    ax.plot(rect_closed[:, 0], rect_closed[:, 1],
            color=RED, linewidth=2.2, zorder=6, label="Rectángulo ajustado (LM)")
    # Center cross
    cxy = fit_result.fit.center_xy()
    ax.scatter(cxy[0], cxy[1], s=50, color=RED, zorder=7, marker="+")
    ax.annotate(
        f"Y={fit_result.Y_m:+.3f} m\nψ={fit_result.psi_deg:+.2f}°",
        xy=(cxy[0], cxy[1]),
        xytext=(cxy[0] + 0.20, cxy[1] + 0.22),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
        fontsize=9, color=TEXT,
    )

    ax.set_title(
        f"Fitting GN/LM  ·  RMSE={fit_result.rmse_m*100:.2f} cm  "
        f"·  iters={fit_result.fit.n_iter}  "
        f"·  {'✓ converged' if fit_result.converged else '✗ not converged'}",
        fontsize=11,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(scene.rail_start_x_m - 0.1, scene.rail_length_m + 0.4)
    ax.set_ylim(-1.25, 1.25)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)
    ax.legend(unique_handles, unique_labels, fontsize=8.5,
              loc="upper right", framealpha=0.85)
    return fig


def plot_ekf_history(
    history: list[EKFState],
    true_Y: float,
    true_psi_deg: float,
) -> plt.Figure:
    """Plot EKF-smoothed Y and ψ over time steps."""
    if not history:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=110)
        ax.text(0.5, 0.5, "Sin historial EKF aún",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    steps = list(range(len(history)))
    Y_vals   = [s.y    for s in history]
    psi_vals = [math.degrees(s.theta) for s in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=120)
    fig.patch.set_facecolor(BG)
    _style_ax(ax1)
    _style_ax(ax2)

    # Y track
    ax1.plot(steps, Y_vals, color=PRIMARY, linewidth=2.0, label="EKF Y")
    ax1.axhline(true_Y, color=RED, linewidth=1.4, linestyle="--", label=f"Y real={true_Y:+.3f} m")
    ax1.fill_between(steps, Y_vals, true_Y, alpha=0.12, color=PRIMARY)
    ax1.set_title("Trayectoria EKF — Error lateral Y", fontsize=10)
    ax1.set_xlabel("Paso de tiempo")
    ax1.set_ylabel("Y [m]")
    ax1.legend(fontsize=8.5)

    # ψ track
    ax2.plot(steps, psi_vals, color=ORANGE, linewidth=2.0, label="EKF ψ")
    ax2.axhline(true_psi_deg, color=RED, linewidth=1.4, linestyle="--",
                label=f"ψ real={true_psi_deg:+.2f}°")
    ax2.fill_between(steps, psi_vals, true_psi_deg, alpha=0.12, color=ORANGE)
    ax2.set_title("Trayectoria EKF — Oblicuidad ψ", fontsize=10)
    ax2.set_xlabel("Paso de tiempo")
    ax2.set_ylabel("ψ [deg]")
    ax2.legend(fontsize=8.5)

    fig.suptitle(
        "EKF CVTR  (Cellina et al. arXiv 2025)  —  Historial de estimación",
        fontsize=11, color=TEXT,
    )
    fig.tight_layout()
    return fig
