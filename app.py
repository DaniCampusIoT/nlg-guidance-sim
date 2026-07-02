from __future__ import annotations

import math
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.catalog.presets import PRESETS
from nlg_guidance_sim.catalog.yaml_loader import load_yaml_str, merged_presets
from nlg_guidance_sim.estimation.lshape import fit_lshape
from nlg_guidance_sim.estimation.pose_ekf import PoseEKF
from nlg_guidance_sim.fitting.pipeline import run_fitting_pipeline
from nlg_guidance_sim.geometry.nlg_model import NLGModel
from nlg_guidance_sim.geometry.profiles import Tire2DProfile
from nlg_guidance_sim.phases.state import ApproachPhase, PhaseState
from nlg_guidance_sim.sensors.rplidar2d import RPLidar2DSim
from nlg_guidance_sim.viz.plotting import plot_ranges, plot_scene, plot_estimation
from nlg_guidance_sim.viz.fitting_plots import plot_fitting, plot_ekf_history
from nlg_guidance_sim.world.scene import Scene


# ─── CSS ──────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top:1.2rem; padding-bottom:1.2rem; max-width:1500px; }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(1,105,111,0.06), transparent 24%),
                linear-gradient(180deg, #f7f6f2 0%, #f3f0ec 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f7f3 0%, #efebe5 100%);
            border-right: 1px solid rgba(40,37,29,0.08);
        }
        .metric-card {
            padding:0.9rem 1rem; border-radius:16px;
            background:rgba(255,255,255,0.70);
            border:1px solid rgba(40,37,29,0.08);
            box-shadow:0 8px 24px rgba(30,30,30,0.05);
        }
        .phase-banner {
            padding:1.1rem 1.4rem; border-radius:14px;
            font-weight:600; font-size:1.15rem; color:#fff; margin-bottom:0.8rem;
        }
        .small-note { color:#66625a; font-size:0.92rem; }
        .prog-bar-outer { background:#e0ddd6; border-radius:999px; height:10px; width:100%; }
        .prog-bar-inner { height:10px; border-radius:999px; transition:width 0.4s ease; }
        .fit-chip {
            display:inline-block; padding:0.25rem 0.7rem; border-radius:999px;
            font-size:0.82rem; font-weight:600; margin-right:0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _phase_banner(phase: ApproachPhase) -> None:
    pct = int(phase.progress() * 100)
    st.markdown(
        f'<div class="phase-banner" style="background:{phase.color()}">'
        f'{phase.label()} &nbsp;·&nbsp; {pct}% completado</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="prog-bar-outer"><div class="prog-bar-inner" '
        f'style="width:{pct}%; background:{phase.color()}"></div></div>',
        unsafe_allow_html=True,
    )
    st.caption(phase.description())


# ─── sidebar ──────────────────────────────────────────────────────────────────

def build_scene_and_sensor(
    extra_presets: dict | None = None,
) -> tuple[Scene, RPLidar2DSim | None]:
    st.sidebar.title("NLG Guidance Sim")
    st.sidebar.caption("Escena · LiDAR · Fitting GN/LM · EKF")

    all_presets = dict(PRESETS)
    if extra_presets:
        all_presets.update(extra_presets)

    preset_name = st.sidebar.selectbox("Preset de aeronave", list(all_presets.keys()))
    preset      = all_presets[preset_name]

    st.sidebar.subheader("Configuración del NLG")
    arrangement   = st.sidebar.radio(
        "Ruedas del tren delantero", ["single", "dual"],
        index=0 if preset.arrangement == "single" else 1,
        format_func=lambda v: "Una rueda" if v == "single" else "Dos ruedas",
    )
    tire_length_m     = st.sidebar.slider("Longitud neumático [m]", 0.40, 1.10, float(preset.tire_length_m), 0.01)
    tire_width_m      = st.sidebar.slider("Anchura neumático [m]",  0.10, 0.40, float(preset.tire_width_m),  0.01)
    shoulder_exponent = st.sidebar.slider("Perfil de hombro",       2.0,  6.0,  float(preset.shoulder_exponent), 0.1)
    track_width_m     = st.sidebar.slider(
        "Separación ruedas [m]", 0.00, 0.90,
        float(0.0 if arrangement == "single" else preset.track_width_m),
        0.01, disabled=(arrangement == "single"),
    )

    st.sidebar.subheader("Plataforma")
    rail_gauge_m      = st.sidebar.slider("Separación raíles [m]",  0.80, 2.00, float(preset.rail_gauge_m),      0.01)
    platform_length_m = st.sidebar.slider("Longitud plataforma [m]",0.80, 3.50, float(preset.platform_length_m), 0.01)
    platform_width_m  = st.sidebar.slider("Ancho plataforma [m]",   0.45, 1.60, float(preset.platform_width_m),  0.01)
    capture_width_m   = st.sidebar.slider(
        "Ancho embocadura [m]", 0.30, float(platform_width_m),
        float(min(preset.capture_width_m, platform_width_m)), 0.01,
    )
    ramp_length_m     = st.sidebar.slider("Longitud rampa [m]",     0.10, 1.00, float(preset.ramp_length_m), 0.01)

    st.sidebar.subheader("Pose del tren")
    center_x_m = st.sidebar.slider("Posición X [m]",   0.60, 6.00,  float(preset.center_x_m), 0.01)
    center_y_m = st.sidebar.slider("Error lateral Y [m]", -0.60, 0.60, float(preset.center_y_m), 0.01)
    psi_deg    = st.sidebar.slider("Oblicuidad ψ [deg]",  -20.0, 20.0, float(preset.psi_deg),    0.1)

    tire_profile = Tire2DProfile(
        tire_length_m=tire_length_m,
        tire_width_m=tire_width_m,
        shoulder_exponent=shoulder_exponent,
        n_profile_points=220,
    )
    nlg = NLGModel(
        arrangement=arrangement,
        track_width_m=track_width_m,
        center_x_m=center_x_m,
        center_y_m=center_y_m,
        psi_rad=math.radians(psi_deg),
        tire_profile=tire_profile,
    )
    scene = Scene(
        name="interactive_scene",
        rail_gauge_m=rail_gauge_m,
        rail_length_m=max(6.0, center_x_m + 1.0),
        rail_start_x_m=-0.20,
        platform_front_x_m=0.00,
        platform_length_m=platform_length_m,
        platform_width_m=platform_width_m,
        capture_width_m=capture_width_m,
        ramp_length_m=ramp_length_m,
        nlg=nlg,
    )

    # ── LiDAR — Slamtec S2E ──────────────────────────────────────────────────
    st.sidebar.subheader("LiDAR 2D  ·  Slamtec S2E")
    enable_lidar = st.sidebar.toggle("Activar LiDAR", value=True)
    if not enable_lidar:
        return scene, None

    origin_x_m    = st.sidebar.slider("LiDAR X [m]", -0.10, float(scene.capture_x_m + 0.20),
                                       float(scene.capture_x_m - 0.12), 0.01)
    origin_y_m    = st.sidebar.slider("LiDAR Y [m]", -0.50, 0.50, 0.00, 0.01)
    angle_min_deg = st.sidebar.slider("FoV mín [deg]", -140, 0, -60, 1)
    angle_max_deg = st.sidebar.slider("FoV máx [deg]", 0, 140,  60, 1)
    num_beams     = st.sidebar.slider("Haces (S2E: 3200 @ 10 Hz)", 60, 3200, 3200, 30)
    max_range_m   = st.sidebar.slider("Alcance [m]", 0.05, 30.00, 30.00, 0.05)
    noise_std_m   = st.sidebar.slider("Ruido gaussiano [m]", 0.000, 0.050, 0.030, 0.001)

    lidar = RPLidar2DSim(
        origin_x_m=origin_x_m,
        origin_y_m=origin_y_m,
        angle_min_deg=angle_min_deg,
        angle_max_deg=angle_max_deg,
        num_beams=num_beams,
        max_range_m=max_range_m,
        range_noise_std_m=noise_std_m,
    )
    return scene, lidar


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NLG Guidance Sim",
        page_icon="🛞",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # ── Session state ─────────────────────────────────────────────────────────
    if "yaml_text"           not in st.session_state:
        yaml_path = ROOT / "configs" / "presets.yaml"
        st.session_state["yaml_text"] = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
    if "yaml_extra_presets"  not in st.session_state:
        st.session_state["yaml_extra_presets"] = {}
    if "yaml_error"          not in st.session_state:
        st.session_state["yaml_error"] = ""
    if "fsm"                 not in st.session_state:
        st.session_state["fsm"] = ApproachPhase()
    if "pose_ekf"            not in st.session_state:
        st.session_state["pose_ekf"] = PoseEKF()

    st.title("Simulador interactivo del tren delantero")
    st.caption(
        "Geometría paramétrica · LiDAR 2D sintético · "
        "L-shape (Zhang 2015) · Fitting GN/LM con prior rígido · "
        "EKF CVTR (Cellina et al. 2025) · Fases de aproximación · YAML presets"
    )

    # ── Build scene + scan ───────────────────────────────────────────────────
    scene, lidar = build_scene_and_sensor(
        extra_presets=st.session_state["yaml_extra_presets"]
    )
    scan     = lidar.scan(scene) if lidar is not None else None
    hit_pts  = scan.ordered_hit_points() if scan is not None else None

    # ── Stage 1: L-shape (Search-Based, Zhang 2015) ──────────────────────────
    est = None
    if hit_pts is not None and len(hit_pts) >= 4:
        est = fit_lshape(hit_pts)

    # ── Stage 2: GN/LM with rigid prior (MDPI 2026 + Cellina 2025) ──────────
    fit_res = None
    preset_for_fit = list(PRESETS.values())[0]   # fallback
    if hit_pts is not None and len(hit_pts) >= 6:
        # W and L come from the active scene's NLG profile — the "known prior"
        W_prior = scene.nlg.tire_profile.tire_width_m
        L_prior = scene.nlg.tire_profile.tire_length_m
        fit_res = run_fitting_pipeline(hit_pts, W_real=W_prior, L_real=L_prior)

    # ── EKF CVTR update ──────────────────────────────────────────────────────
    ekf: PoseEKF = st.session_state["pose_ekf"]
    if fit_res is not None:
        cxy = fit_res.fit.center_xy()
        if not ekf.initialized:
            ekf.initialize(cxy[0], cxy[1], fit_res.psi_rad)
        else:
            ekf.predict(dt=0.05)
            ekf.update(cxy[0], cxy[1])

    # ── Phase FSM ────────────────────────────────────────────────────────────
    fsm: ApproachPhase = st.session_state["fsm"]
    x_to_capture = max(0.0, scene.capture_x_m - scene.nlg.center_x_m)
    Y_est   = fit_res.Y_m if fit_res is not None else (
                  est.Y_m if est is not None else scene.nlg.center_y_m)
    psi_est = fit_res.psi_rad if fit_res is not None else (
                  est.psi_rad if est is not None else scene.nlg.psi_rad)
    fsm.update(x_to_capture=x_to_capture, Y_m=Y_est, psi_rad=psi_est)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_scene, tab_lidar, tab_est, tab_fit, tab_phases, tab_yaml, tab_diag = st.tabs(
        ["Escena", "LiDAR 2D", "Estimación Y/ψ", "Fitting GN/LM", "Fases", "Editor YAML", "Diagnóstico"]
    )

    # ── Tab: Escena ──────────────────────────────────────────────────────────
    with tab_scene:
        fig = plot_scene(scene, scan_result=scan, est_result=est)
        st.pyplot(fig, use_container_width=True)

    # ── Tab: LiDAR ───────────────────────────────────────────────────────────
    with tab_lidar:
        if scan is None:
            st.info("Activa el LiDAR en la barra lateral para visualizar la nube ordenada.")
        else:
            fig = plot_ranges(scan)
            st.pyplot(fig, use_container_width=True)

    # ── Tab: Estimación L-shape (Zhang 2015) ─────────────────────────────────
    with tab_est:
        if est is None:
            st.warning("No hay suficientes puntos de impacto. Activa el LiDAR y acerca el tren.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Y estimada", f"{est.Y_m:+.4f} m",
                          delta=f"ref {scene.nlg.center_y_m:+.4f} m")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("ψ estimada", f"{est.psi_deg:+.3f} deg",
                          delta=f"ref {math.degrees(scene.nlg.psi_rad):+.3f} deg")
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RMSE fit", f"{est.rmse_m*100:.2f} cm")
                st.markdown("</div>", unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confianza", f"{est.confidence:.2f}", delta=est.method)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")
            fig_est = plot_estimation(scene, est)
            st.pyplot(fig_est, use_container_width=True)
            with st.expander("ℹ️ Método L-shape (CISRAM 2015)"):
                st.markdown(
                    "Barre **N_THETA = 180** orientaciones candidatas en [0, π). "
                    "El criterio de coste es la suma de residuos cuadráticos "
                    "perpendiculares a las dos rectas ajustadas (Zhang et al. 2015). "
                    "Fallbacks: `line` si <2 segmentos válidos, `centroid` si <8 puntos."
                )

    # ── Tab: Fitting GN/LM ───────────────────────────────────────────────────
    with tab_fit:
        st.subheader("Pipeline: RDP seed → Gauss-Newton / Levenberg-Marquardt")
        st.caption(
            "W y L del neumático se fijan desde el preset (prior rígido). "
            "Solo se optimizan (xc, yc, θ). "
            "Referencias: MDPI 2026 (Rectangle Edge Matching) · Cellina et al. arXiv 2025."
        )

        if fit_res is None:
            st.warning("No hay suficientes puntos LiDAR (mín. 6). Activa el LiDAR y acerca el tren.")
        else:
            # ── Metrics row ──
            m1, m2, m3, m4, m5 = st.columns(5)
            lm_chip_color = "#2e7d32" if fit_res.converged else "#b94e48"
            lm_label      = "✓ LM converged" if fit_res.converged else "✗ LM not converged"

            with m1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Y (LM)", f"{fit_res.Y_m:+.4f} m",
                          delta=f"err {(fit_res.Y_m - scene.nlg.center_y_m)*1000:+.1f} mm")
                st.markdown("</div>", unsafe_allow_html=True)
            with m2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("ψ (LM)", f"{fit_res.psi_deg:+.3f}°",
                          delta=f"err {fit_res.psi_deg - math.degrees(scene.nlg.psi_rad):+.3f}°")
                st.markdown("</div>", unsafe_allow_html=True)
            with m3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RMSE", f"{fit_res.rmse_m*100:.2f} cm")
                st.markdown("</div>", unsafe_allow_html=True)
            with m4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Iteraciones LM", str(fit_res.fit.n_iter))
                st.markdown("</div>", unsafe_allow_html=True)
            with m5:
                st.markdown(
                    f'<div class="metric-card"><span class="fit-chip" '
                    f'style="background:{lm_chip_color};color:#fff">{lm_label}</span>\''
                    f'<br><span class="small-note">θ₀ RDP: {math.degrees(fit_res.seed_theta_rad):+.2f}°</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # ── Two-column layout: scene overlay + EKF history ──
            col_l, col_r = st.columns([1.1, 0.9])

            with col_l:
                st.markdown("**Escena con rectángulo ajustado**")
                fig_fit = plot_fitting(scene, fit_res)
                st.pyplot(fig_fit, use_container_width=True)

            with col_r:
                st.markdown("**EKF CVTR — Historial Y / ψ**")
                ekf_history = ekf.history
                fig_ekf = plot_ekf_history(
                    ekf_history,
                    true_Y=float(scene.nlg.center_y_m),
                    true_psi_deg=float(math.degrees(scene.nlg.psi_rad)),
                )
                st.pyplot(fig_ekf, use_container_width=True)

                # EKF current state
                if ekf.initialized:
                    st.markdown("**Estado EKF actual**")
                    eks = ekf.state
                    st.markdown(
                        f"| Variable | EKF | Real |\n"
                        f"|---|---|---|\n"
                        f"| Y [m] | `{eks.y:+.4f}` | `{scene.nlg.center_y_m:+.4f}` |\n"
                        f"| ψ [deg] | `{math.degrees(eks.theta):+.3f}` | `{math.degrees(scene.nlg.psi_rad):+.3f}` |\n"
                        f"| v [m/s] | `{eks.v:.3f}` | — |\n"
                        f"| ω [rad/s] | `{eks.omega:.4f}` | — |\n"
                    )

                if st.button("🔄 Reiniciar EKF", key="reset_ekf"):
                    st.session_state["pose_ekf"] = PoseEKF()
                    st.rerun()

            # ── Expander: method details ──
            with st.expander("ℹ️ Pipeline de dos etapas — detalle matemático"):
                st.markdown(
                    r"""
**Etapa 1 — Semilla RDP** (Ramer-Douglas-Peucker)

Sobre la nube ordenada por ángulo $\{p_i\}$, se busca el índice
$k^* = \arg\max_k\, d_\perp(p_k, \overline{p_0 p_N})$.

Ese punto es el vértice de la L. El brazo dominante define $\theta_0$.

---

**Etapa 2 — Levenberg-Marquardt con prior rígido** (MDPI 2026)

Estado: $\mathbf{x} = [x_c,\; y_c,\; \theta]^T$. Las dimensiones $W$ y $L$ son **constantes** del preset.

Residuo por punto:
$$r_i(\mathbf{x}) = \min_j\, d_\perp\!\left(p_i,\; \text{borde}_j(\mathbf{x}, W, L)\right)$$

Actualización LM:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - \bigl(\mathbf{J}^T\mathbf{J} + \lambda\,\mathrm{diag}(\mathbf{J}^T\mathbf{J})\bigr)^{-1}\mathbf{J}^T\mathbf{r}(\mathbf{x}_k)$$

---

**Etapa 3 — EKF CVTR** (Cellina et al. arXiv 2025, eq. 6)

$\mathbf{X} = [x, y, \theta, v, \omega]^T$. Medición: sólo $[x_c, y_c]$ (sin $\theta$ directo).

Modelo de movimiento:
$$x_{k+1} = x_k + v_k \cos\theta_k\,\Delta t, \quad y_{k+1} = y_k + v_k \sin\theta_k\,\Delta t$$
"""
                )

    # ── Tab: Fases ───────────────────────────────────────────────────────────
    with tab_phases:
        _phase_banner(fsm)
        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Parámetros de transición")
            abs_Y   = abs(Y_est)
            abs_psi = abs(math.degrees(psi_est))
            t = fsm.thresholds
            for label, value, note in [
                ("X → captura", f"{x_to_capture:.3f} m",
                 f"ALIGN={t.x_align_m:.2f} m · CAPTURE={t.x_capture_m:.2f} m"),
                ("|Y|",  f"{abs_Y:.4f} m",
                 f"ALIGN={t.y_align_m:.2f} m · CAPTURE={t.y_capture_m:.2f} m"),
                ("|ψ|",  f"{abs_psi:.3f}°",
                 f"ALIGN={math.degrees(t.psi_align_rad):.1f}° · CAPTURE={math.degrees(t.psi_capture_rad):.1f}°"),
            ]:
                st.markdown(f"**{label}**: `{value}` — *{note}*")
        with col_r:
            st.subheader("Diagrama de fases")
            for ph in [PhaseState.APPROACH, PhaseState.ALIGN,
                       PhaseState.CAPTURE, PhaseState.HOLD]:
                active = (ph == fsm.state)
                border = f"3px solid {fsm.PHASE_COLORS[ph]}" if active else "1px solid #ddd"
                bg     = f"{fsm.PHASE_COLORS[ph]}18"         if active else "#fafaf8"
                badge  = " ◄ ACTIVA" if active else ""
                st.markdown(
                    f'<div style="border:{border};background:{bg};border-radius:12px;'
                    f'padding:0.7rem 1rem;margin-bottom:0.5rem">'
                    f'<b style="color:{fsm.PHASE_COLORS[ph]}">{fsm.PHASE_LABELS[ph]}{badge}</b><br>'
                    f'<span style="font-size:0.88rem;color:#66625a">{fsm.PHASE_DESCRIPTIONS[ph]}</span></div>',
                    unsafe_allow_html=True,
                )
        if st.button("🔄 Reiniciar FSM"):
            fsm.reset()
            st.rerun()

    # ── Tab: Editor YAML ─────────────────────────────────────────────────────
    with tab_yaml:
        st.subheader("Editor de presets YAML")
        st.caption("Añade nuevos tipos de aeronave sin tocar código Python.")
        col_edit, col_help = st.columns([3, 1])
        with col_help:
            st.markdown(
                """
**Estructura mínima**
```yaml
presets:
  "Mi aeronave":
    arrangement: single
    tire_length_m: 0.65
    tire_width_m: 0.20
    shoulder_exponent: 3.5
    track_width_m: 0.00
    rail_gauge_m: 1.15
    platform_length_m: 1.30
    platform_width_m: 0.72
    capture_width_m: 0.50
    ramp_length_m: 0.25
    center_x_m: 2.00
    center_y_m: 0.00
    psi_deg: 0.0
```"""
            )
        with col_edit:
            edited = st.text_area(
                "YAML", value=st.session_state["yaml_text"], height=460,
                label_visibility="collapsed",
            )
        b1, b2 = st.columns([1, 4])
        with b1:
            if st.button("✅ Aplicar presets", type="primary"):
                try:
                    new_p = load_yaml_str(edited)
                    st.session_state.update(yaml_text=edited, yaml_extra_presets=new_p, yaml_error="")
                    st.success(f"{len(new_p)} preset(s): {list(new_p.keys())}")
                    st.rerun()
                except Exception as exc:
                    st.session_state["yaml_error"] = str(exc)
        with b2:
            if st.button("↩️ Restaurar"):
                yaml_path = ROOT / "configs" / "presets.yaml"
                if yaml_path.exists():
                    st.session_state.update(
                        yaml_text=yaml_path.read_text(encoding="utf-8"),
                        yaml_extra_presets={}, yaml_error="",
                    )
                    st.rerun()
        if st.session_state["yaml_error"]:
            st.error(f"Error: {st.session_state['yaml_error']}")

    # ── Tab: Diagnóstico ─────────────────────────────────────────────────────
    with tab_diag:
        c1, c2, c3, c4 = st.columns(4)
        summary = scene.summary_dict()
        for col, label, key, fmt in zip(
            [c1, c2, c3, c4],
            ["Midpoint X", "Error lateral Y", "Oblicuidad ψ", "Ruedas"],
            ["midpoint_x_m", "midpoint_y_m", "psi_deg", "n_wheels"],
            ["{:.3f} m", "{:+.3f} m", "{:.2f} deg", "{}"],
        ):
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(label, fmt.format(summary[key]))
                st.markdown("</div>", unsafe_allow_html=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Resumen de escena")
            st.json(summary)
        with right:
            st.subheader("Estado del sistema")
            _conf   = f"{est.confidence:.2f}" if est is not None else "n/a"
            _method = est.method              if est is not None else "n/a"
            _fit_rmse = f"{fit_res.rmse_m*100:.2f} cm" if fit_res is not None else "n/a"
            _fit_iter = str(fit_res.fit.n_iter) if fit_res is not None else "n/a"
            st.markdown(
                f"""| Variable | Valor |
|---|---|
| Fase | **{fsm.label()}** |
| X → captura | `{x_to_capture:.3f} m` |
| Y estimada | `{Y_est:+.4f} m` |
| ψ estimada | `{math.degrees(psi_est):+.3f}°` |
| Método L-shape | `{_method}` |
| Confianza L-shape | `{_conf}` |
| RMSE GN/LM | `{_fit_rmse}` |
| Iters LM | `{_fit_iter}` |
| EKF init | `{ekf.initialized}` |
"""
            )


if __name__ == "__main__":
    main()
