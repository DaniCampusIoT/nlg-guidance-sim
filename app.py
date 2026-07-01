from __future__ import annotations

import math
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.catalog.presets import PRESETS
from nlg_guidance_sim.catalog.yaml_loader import (
    load_yaml_str,
    presets_to_yaml_str,
    merged_presets,
)
from nlg_guidance_sim.estimation.lshape import fit_lshape
from nlg_guidance_sim.geometry.nlg_model import NLGModel
from nlg_guidance_sim.geometry.profiles import Tire2DProfile
from nlg_guidance_sim.phases.state import ApproachPhase, PhaseState
from nlg_guidance_sim.sensors.rplidar2d import RPLidar2DSim
from nlg_guidance_sim.viz.plotting import plot_ranges, plot_scene, plot_estimation
from nlg_guidance_sim.world.scene import Scene


# ─── helpers ──────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            max-width: 1500px;
        }
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
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,0.70);
            border: 1px solid rgba(40,37,29,0.08);
            box-shadow: 0 8px 24px rgba(30, 30, 30, 0.05);
        }
        .phase-banner {
            padding: 1.1rem 1.4rem;
            border-radius: 14px;
            font-weight: 600;
            font-size: 1.15rem;
            color: #fff;
            margin-bottom: 0.8rem;
        }
        .small-note { color: #66625a; font-size: 0.92rem; }
        .prog-bar-outer {
            background: #e0ddd6;
            border-radius: 999px;
            height: 10px;
            width: 100%;
        }
        .prog-bar-inner {
            height: 10px;
            border-radius: 999px;
            transition: width 0.4s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _phase_banner(phase: ApproachPhase) -> None:
    pct = int(phase.progress() * 100)
    st.markdown(
        f'<div class="phase-banner" style="background:{phase.color()}">'
        f'{phase.label()} &nbsp;·&nbsp; {pct}% completado'
        f"</div>",
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
    st.sidebar.caption("Escena geométrica + LiDAR 2D sintético")

    all_presets = dict(PRESETS)
    if extra_presets:
        all_presets.update(extra_presets)

    preset_name = st.sidebar.selectbox("Preset de aeronave", list(all_presets.keys()))
    preset = all_presets[preset_name]

    st.sidebar.subheader("Configuración del NLG")
    arrangement = st.sidebar.radio(
        "Ruedas del tren delantero",
        options=["single", "dual"],
        index=0 if preset.arrangement == "single" else 1,
        format_func=lambda v: "Una rueda" if v == "single" else "Dos ruedas",
    )
    tire_length_m = st.sidebar.slider("Longitud aparente del neumático [m]", 0.40, 1.10,
                                       float(preset.tire_length_m), 0.01)
    tire_width_m  = st.sidebar.slider("Anchura aparente del neumático [m]", 0.10, 0.40,
                                       float(preset.tire_width_m), 0.01)
    shoulder_exponent = st.sidebar.slider("Perfil de hombro", 2.0, 6.0,
                                           float(preset.shoulder_exponent), 0.1)
    track_width_m = st.sidebar.slider(
        "Separación entre ruedas [m]", 0.00, 0.90,
        float(0.0 if arrangement == "single" else preset.track_width_m),
        0.01, disabled=arrangement == "single",
    )

    st.sidebar.subheader("Plataforma")
    rail_gauge_m      = st.sidebar.slider("Separación entre raíles [m]",   0.80, 2.00, float(preset.rail_gauge_m),      0.01)
    platform_length_m = st.sidebar.slider("Longitud de plataforma [m]",    0.80, 3.50, float(preset.platform_length_m), 0.01)
    platform_width_m  = st.sidebar.slider("Ancho de plataforma [m]",       0.45, 1.60, float(preset.platform_width_m),  0.01)
    capture_width_m   = st.sidebar.slider(
        "Ancho de embocadura/cuna [m]", 0.30, float(platform_width_m),
        float(min(preset.capture_width_m, platform_width_m)), 0.01,
    )
    ramp_length_m = st.sidebar.slider("Longitud de rampa [m]", 0.10, 1.00, float(preset.ramp_length_m), 0.01)

    st.sidebar.subheader("Pose del tren")
    center_x_m = st.sidebar.slider("Posición longitudinal X [m]", 0.60, 6.00, float(preset.center_x_m), 0.01)
    center_y_m = st.sidebar.slider("Error lateral Y [m]",         -0.60, 0.60, float(preset.center_y_m), 0.01)
    psi_deg    = st.sidebar.slider("Oblicuidad ψ [deg]",          -20.0, 20.0, float(preset.psi_deg),    0.1)

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

    st.sidebar.subheader("LiDAR 2D")
    enable_lidar = st.sidebar.toggle("Activar LiDAR", value=True)
    if not enable_lidar:
        return scene, None

    origin_x_m    = st.sidebar.slider("LiDAR X [m]", -0.10, float(scene.capture_x_m + 0.20),
                                       float(scene.capture_x_m - 0.12), 0.01)
    origin_y_m    = st.sidebar.slider("LiDAR Y [m]", -0.50, 0.50, 0.00, 0.01)
    angle_min_deg = st.sidebar.slider("FoV mínimo [deg]", -140, 0, -60, 1)
    angle_max_deg = st.sidebar.slider("FoV máximo [deg]", 0, 140, 60, 1)
    num_beams     = st.sidebar.slider("Número de haces", 60, 1440, 360, 30)
    max_range_m   = st.sidebar.slider("Alcance máximo [m]", 0.50, 10.00, 5.00, 0.10)
    noise_std_m   = st.sidebar.slider("Ruido gaussiano [m]", 0.0, 0.02, 0.002, 0.001)

    lidar = RPLidar2DSim(
        origin_x_m=origin_x_m, origin_y_m=origin_y_m,
        angle_min_deg=angle_min_deg, angle_max_deg=angle_max_deg,
        num_beams=num_beams, max_range_m=max_range_m,
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

    # ── YAML editor state ──
    if "yaml_text" not in st.session_state:
        yaml_path = ROOT / "configs" / "presets.yaml"
        st.session_state["yaml_text"] = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
    if "yaml_extra_presets" not in st.session_state:
        st.session_state["yaml_extra_presets"] = {}
    if "yaml_error" not in st.session_state:
        st.session_state["yaml_error"] = ""

    # ── Phase FSM state ──
    if "fsm" not in st.session_state:
        st.session_state["fsm"] = ApproachPhase()

    st.title("Simulador interactivo del tren delantero")
    st.caption(
        "Geometría paramétrica · LiDAR 2D sintético · Estimación Y/ψ · "
        "Fases de aproximación · Editor de presets YAML"
    )

    scene, lidar = build_scene_and_sensor(
        extra_presets=st.session_state["yaml_extra_presets"]
    )
    scan = lidar.scan(scene) if lidar is not None else None
    hit_pts = scan.ordered_hit_points() if scan is not None else None

    # ── Estimación ──
    est = None
    if hit_pts is not None and len(hit_pts) >= 4:
        est = fit_lshape(hit_pts)

    # ── Phase FSM update ──
    fsm: ApproachPhase = st.session_state["fsm"]
    x_to_capture = max(0.0, scene.capture_x_m - scene.nlg.center_x_m)
    Y_est   = est.Y_m      if est is not None else scene.nlg.center_y_m
    psi_est = est.psi_rad  if est is not None else scene.nlg.psi_rad
    fsm.update(x_to_capture=x_to_capture, Y_m=Y_est, psi_rad=psi_est)

    # ── Tabs ──
    tab_scene, tab_lidar, tab_est, tab_phases, tab_yaml, tab_diag = st.tabs(
        ["Escena", "LiDAR 2D", "Estimación Y/ψ", "Fases", "Editor YAML", "Diagnóstico"]
    )

    # ── Tab: Escena ──
    with tab_scene:
        fig = plot_scene(scene, scan_result=scan, est_result=est)
        st.pyplot(fig, use_container_width=True)

    # ── Tab: LiDAR ──
    with tab_lidar:
        if scan is None:
            st.info("Activa el LiDAR en la barra lateral para visualizar la nube ordenada.")
        else:
            fig = plot_ranges(scan)
            st.pyplot(fig, use_container_width=True)

    # ── Tab: Estimación ──
    with tab_est:
        if est is None:
            st.warning("No hay suficientes puntos de impacto para estimar. Activa el LiDAR y acerca el tren.")
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
                st.metric("Confianza", f"{est.confidence:.2f}",
                          delta=est.method)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")
            fig_est = plot_estimation(scene, est)
            st.pyplot(fig_est, use_container_width=True)

            with st.expander("ℹ️ Método L-shape (CISRAM 2015)"):
                st.markdown(
                    """
                    El estimador barre **N_THETA = 180** orientaciones candidatas en [0, π)
                    y para cada θ divide la proyección ordenada de los puntos de impacto en
                    dos brazos. El criterio de coste es la suma de residuos cuadráticos
                    perpendiculares a las dos rectas ajustadas (método de Zhang et al. 2015).

                    El ángulo óptimo define la orientación del NLG; la intersección de las
                    dos rectas da la esquina del L, cuya coordenada Y es el estimador de
                    offset lateral. ψ se extrae del brazo más perpendicular al eje X.

                    **Fallbacks:**
                    - `line` — menos de dos segmentos válidos: ajuste de recta única.
                    - `centroid` — menos de 8 puntos: centroide de los puntos de impacto.
                    """
                )

    # ── Tab: Fases ──
    with tab_phases:
        _phase_banner(fsm)
        st.markdown("---")

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.subheader("Parámetros de transición actuales")
            summary = scene.summary_dict()
            x_to_capture_ui = max(0.0, scene.capture_x_m - scene.nlg.center_x_m)
            abs_Y   = abs(Y_est)
            abs_psi = abs(math.degrees(psi_est))

            t = fsm.thresholds
            rows = [
                ("X → captura", f"{x_to_capture_ui:.3f} m",
                 f"umbral ALIGN={t.x_align_m:.2f} m · CAPTURE={t.x_capture_m:.2f} m"),
                ("|Y| estimada", f"{abs_Y:.4f} m",
                 f"umbral ALIGN={t.y_align_m:.2f} m · CAPTURE={t.y_capture_m:.2f} m"),
                ("|ψ| estimada", f"{abs_psi:.3f} deg",
                 f"umbral ALIGN={math.degrees(t.psi_align_rad):.1f}° · CAPTURE={math.degrees(t.psi_capture_rad):.1f}°"),
            ]
            for label, value, note in rows:
                st.markdown(f"**{label}**: `{value}` — *{note}*")

        with col_r:
            st.subheader("Diagrama de fases")
            phase_order = [PhaseState.APPROACH, PhaseState.ALIGN,
                           PhaseState.CAPTURE, PhaseState.HOLD]
            for ph in phase_order:
                active = (ph == fsm.state)
                border = f"3px solid {fsm.PHASE_COLORS[ph]}" if active else "1px solid #ddd"
                bg     = f"{fsm.PHASE_COLORS[ph]}18" if active else "#fafaf8"
                label  = fsm.PHASE_LABELS[ph]
                desc   = fsm.PHASE_DESCRIPTIONS[ph]
                badge  = " ◀ ACTIVA" if active else ""
                st.markdown(
                    f'<div style="border:{border};background:{bg};border-radius:12px;'
                    f'padding:0.7rem 1rem;margin-bottom:0.5rem">'
                    f'<b style="color:{fsm.PHASE_COLORS[ph]}">{label}{badge}</b><br>'
                    f'<span style="font-size:0.88rem;color:#66625a">{desc}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")
        if st.button("🔄 Reiniciar FSM", type="secondary"):
            fsm.reset()
            st.rerun()

    # ── Tab: Editor YAML ──
    with tab_yaml:
        st.subheader("Editor de presets YAML")
        st.caption(
            "Edita el YAML directamente para añadir nuevos tipos de aeronave sin tocar código Python. "
            "Los presets cargados se fusionan con los incorporados y aparecen en el selector de la barra lateral."
        )

        col_edit, col_help = st.columns([3, 1])
        with col_help:
            st.markdown(
                """
                **Estructura mínima**
                ```yaml
                presets:
                  "Mi aeronave":
                    arrangement: single  # o dual
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
                ```
                """
            )

        with col_edit:
            edited = st.text_area(
                "Contenido YAML",
                value=st.session_state["yaml_text"],
                height=460,
                label_visibility="collapsed",
            )

        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            if st.button("✅ Aplicar presets", type="primary"):
                try:
                    new_presets = load_yaml_str(edited)
                    st.session_state["yaml_text"] = edited
                    st.session_state["yaml_extra_presets"] = new_presets
                    st.session_state["yaml_error"] = ""
                    st.success(f"{len(new_presets)} preset(s) cargados: {list(new_presets.keys())}")
                    st.rerun()
                except Exception as exc:
                    st.session_state["yaml_error"] = str(exc)

        with btn_col2:
            if st.button("↩️ Restaurar por defecto"):
                yaml_path = ROOT / "configs" / "presets.yaml"
                if yaml_path.exists():
                    st.session_state["yaml_text"] = yaml_path.read_text(encoding="utf-8")
                    st.session_state["yaml_extra_presets"] = {}
                    st.session_state["yaml_error"] = ""
                    st.rerun()

        if st.session_state["yaml_error"]:
            st.error(f"Error de validación: {st.session_state['yaml_error']}")

        if st.session_state["yaml_extra_presets"]:
            st.info(
                f"Presets activos desde YAML: "
                f"{list(st.session_state['yaml_extra_presets'].keys())}"
            )

    # ── Tab: Diagnóstico ──
    with tab_diag:
        c1, c2, c3, c4 = st.columns(4)
        summary = scene.summary_dict()
        with c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Midpoint X", f"{summary['midpoint_x_m']:.3f} m")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Error lateral Y", f"{summary['midpoint_y_m']:.3f} m")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Oblicuidad ψ", f"{summary['psi_deg']:.2f} deg")
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Ruedas", str(summary["n_wheels"]))
            st.markdown("</div>", unsafe_allow_html=True)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Resumen de escena")
            st.json(summary)
        with right:
            st.subheader("Estado del sistema")
            st.markdown(
                f"""
                | Variable | Valor |
                |---|---|
                | Fase actual | **{fsm.label()}** |
                | X → captura | `{x_to_capture:.3f} m` |
                | Y estimada | `{Y_est:+.4f} m` |
                | ψ estimada | `{math.degrees(psi_est):+.3f} deg` |
                | Método est. | `{est.method if est else 'n/a'}` |
                | Confianza | `{est.confidence:.2f if est else 'n/a'}` |
                """
            )


if __name__ == "__main__":
    main()
