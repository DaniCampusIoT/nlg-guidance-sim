"""Matplotlib helpers for the fitting pipeline visualisation.

Adapted to PipelineResult (two-stage: coarse LShapeResult + refined RefinedPose).

Fix (2026-07-02)
----------------
plot_fitting() previously contained the dead / broken line::

    pts = fit_result.coarse.n_points

coarse.n_points is an *int* (count of points), not a point array.
The actual hit-point cloud is now stored in PipelineResult.hit_points_array
and drawn as a coloured scatter in the scene overlay.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import numpy as np

from nlg_guidance_sim.fitting.pipeline import PipelineResult
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


def _rect_corners(
    xc: float, yc: float, theta: float, half_l: float, half_w: float
) -> np.ndarray:
    """Return (4, 2) corner array for a rectangle centred at (xc, yc) rotated by theta."""
    c, s = math.cos(theta), math.sin(theta)
    offsets = np.array([
        [ half_l,  half_w],
        [-half_l,  half_w],
        [-half_l, -half_w],
        [ half_l, -half_w],
    ])
    rot = np.array([[c, -s], [s, c]])
    return offsets @ rot.T + np.array([xc, yc])


def plot_fitting(
    scene: Scene,
    fit_result: PipelineResult,
) -> plt.Figure:
    """Overlay fitted rigid rectangle + coarse L-shape seed on the 2D scene.

    The LiDAR hit-point cloud is drawn from ``fit_result.hit_points_array``
    (an (N, 2) ndarray stored by ``run_pipeline``).
    Previously this function incorrectly read ``fit_result.coarse.n_points``
    (an int) as though it were the point array — fixed here.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=130)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    ax.set_aspect("equal")

    # ── Rails + guide axis ──────────────────────────────────────────────────
    x0 = scene.rail_start_x_m
    x1 = scene.rail_length_m
    ax.plot([x0, x1], [scene.left_rail_y_m]  * 2, "--", color="#8f8a81", lw=1.6, label="Raíles")
    ax.plot([x0, x1], [scene.right_rail_y_m] * 2, "--", color="#8f8a81", lw=1.6)
    ax.plot([x0, x1], [scene.guide_axis_y_m] * 2, ":",  color=MUTED,     lw=1.2)

    # ── Platform ────────────────────────────────────────────────────────────
    from matplotlib.patches import Polygon as MPoly
    for outline, fc in [
        (scene.platform_body_outline(), "#ddd8cf"),
        (scene.ramp_outline(),          "#e8e2d8"),
    ]:
        ax.add_patch(MPoly(outline, closed=True, facecolor=fc,
                           edgecolor="#625d56", linewidth=1.6))

    # ── Ground-truth wheel contours (dashed) ────────────────────────────────
    for contour in scene.nlg.wheel_contours_world():
        ax.add_patch(MPoly(contour, closed=True, facecolor="#dce7ef",
                           edgecolor=BLUE, linewidth=1.6, alpha=0.60,
                           linestyle="--", label="Contorno real"))

    # ── LiDAR hit-point cloud ───────────────────────────────────────────────
    # hit_points_array is an (N, 2) ndarray stored in PipelineResult by
    # run_pipeline().  coarse.n_points is an *int* (count only) — do NOT use
    # it as a point array.
    hit_pts = fit_result.hit_points_array
    if hit_pts is not None and len(hit_pts) > 0:
        ax.scatter(
            hit_pts[:, 0], hit_pts[:, 1],
            s=9, color=CYAN, alpha=0.75, zorder=5,
            label=f"Nube LiDAR ({len(hit_pts)} pts)",
        )

    # ── Stage-1: coarse L-shape centroid ───────────────────────────────────
    cx0, cy0 = fit_result.coarse.xc, fit_result.coarse.yc
    ax.scatter(
        cx0, cy0, s=70, color=ORANGE, zorder=7, marker="s",
        label=f"L-shape coarse  (costo={fit_result.coarse.cost:.4f})",
    )

    # ── Stage-0: top-3 corner candidates from segmentation ─────────────────
    for cand in fit_result.corner_candidates[:3]:
        cpt = cand.get("corner", None)
        if cpt is not None:
            ax.scatter(cpt[0], cpt[1], s=45, color=ORANGE,
                       alpha=0.55, zorder=6, marker="^")

    # ── Stage-2: LM-refined rectangle ──────────────────────────────────────
    ref = fit_result.refined
    half_l = ref.length_m / 2.0
    half_w = ref.width_m  / 2.0
    corners = _rect_corners(ref.xc, ref.yc, ref.theta_rad, half_l, half_w)
    rect_closed = np.vstack([corners, corners[0]])
    ax.plot(
        rect_closed[:, 0], rect_closed[:, 1],
        color=RED, linewidth=2.4, zorder=6, label="Rectángulo LM",
    )
    ax.scatter(ref.xc, ref.yc, s=55, color=RED, zorder=7, marker="+")

    ax.annotate(
        f"Y={fit_result.Y_m:+.3f} m\nψ={fit_result.psi_deg:+.2f}°",
        xy=(ref.xc, ref.yc),
        xytext=(ref.xc + 0.20, ref.yc + 0.22),
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
        fontsize=9, color=TEXT,
    )

    converged_str = "✓ converged" if ref.converged else "✗ not converged"
    ax.set_title(
        f"Fitting LM  ·  RMSE={ref.rmse*100:.2f} cm  "
        f"·  iters={ref.n_iter}  ·  {converged_str}  "
        f"·  {fit_result.n_points_used} pts  ·  {fit_result.elapsed_ms:.1f} ms",
        fontsize=10,
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(scene.rail_start_x_m - 0.1, scene.rail_length_m + 0.4)
    ax.set_ylim(-1.25, 1.25)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lbl in zip(handles, labels):
        if lbl not in seen:
            uh.append(h); ul.append(lbl); seen.add(lbl)
    ax.legend(uh, ul, fontsize=8.5, loc="upper right", framealpha=0.85)
    return fig


def plot_ekf_history(
    history: list[EKFState],
    true_Y: float,
    true_psi_deg: float,
) -> plt.Figure:
    """Plot EKF-smoothed Y and ψ over time steps."""
    if not history:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=110)
        fig.patch.set_facecolor(BG)
        _style_ax(ax)
        ax.text(
            0.5, 0.5,
            "Sin historial EKF aún — mueve el tren para acumular pasos",
            ha="center", va="center",
            transform=ax.transAxes, color=MUTED, fontsize=10,
        )
        return fig

    steps    = list(range(len(history)))
    Y_vals   = [s.y for s in history]
    psi_vals = [math.degrees(s.theta) for s in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), dpi=120)
    fig.patch.set_facecolor(BG)
    _style_ax(ax1)
    _style_ax(ax2)

    ax1.plot(steps, Y_vals, color=PRIMARY, linewidth=2.0, label="EKF Y")
    ax1.axhline(true_Y, color=RED, linewidth=1.4, linestyle="--",
                label=f"Y real={true_Y:+.3f} m")
    ax1.fill_between(steps, Y_vals, true_Y, alpha=0.12, color=PRIMARY)
    ax1.set_title("EKF — Error lateral Y", fontsize=10)
    ax1.set_xlabel("Paso de tiempo")
    ax1.set_ylabel("Y [m]")
    ax1.legend(fontsize=8.5)

    ax2.plot(steps, psi_vals, color=ORANGE, linewidth=2.0, label="EKF ψ")
    ax2.axhline(true_psi_deg, color=RED, linewidth=1.4, linestyle="--",
                label=f"ψ real={true_psi_deg:+.2f}°")
    ax2.fill_between(steps, psi_vals, true_psi_deg, alpha=0.12, color=ORANGE)
    ax2.set_title("EKF — Oblicuidad ψ", fontsize=10)
    ax2.set_xlabel("Paso de tiempo")
    ax2.set_ylabel("ψ [deg]")
    ax2.legend(fontsize=8.5)

    fig.suptitle(
        "EKF CVTR  (Cellina et al. arXiv 2025)  —  Historial de estimación",
        fontsize=11, color=TEXT,
    )
    fig.tight_layout()
    return fig
