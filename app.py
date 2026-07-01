from __future__ import annotations

import io
import math
from pathlib import Path
import sys

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlg_guidance_sim.catalog.presets import AircraftPreset, load_presets_yaml
from nlg_guidance_sim.control.phases import (
    PhaseThresholds, classify_phase, all_phase_states,
)
from nlg_guidance_sim.estimation.lshape import LShapeFitter
from nlg_guidance_sim.estimation.pose import estimate_pose
from nlg_guidance_sim.geometry.nlg_model import NLGModel
from nlg_guidance_sim.geometry.profiles import Tire2DProfile
from nlg_guidance_sim.sensors.rplidar2d import RPLidar2DSim
from nlg_guidance_sim.viz.plotting import (
    plot_estimation_debug, plot_ranges, plot_scene,
)
from nlg_guidance_sim.world.scene import Scene


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top:1.2rem; padding-bottom:1.2rem; max-width:1500px; }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(1,105,111,0.06), transparent 24%),
                linear-gradient(180deg,#f7f6f2 0%,#f3f0ec 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg,#f8f7f3 0%,#efebe5 100%);
            border-right: 1px solid rgba(40,37,29,0.08);
        }
        .metric-card {
            padding:.9rem 1rem; border-radius:16px;
            background:rgba(255,255,255,0.70);
            border:1px solid rgba(40,37,29,0.08);
            box-shadow:0 8px 24px rgba(30,30,30,0.05);
        }
        .phase-badge {
            display:inline-block; padding:.35rem .85rem;
            border-radius:999px; font-weight:600;
            font-size:.95rem; letter-spacing:.01em;
        }
        .small-note { color:#66625a; font-size:.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Preset loader (with YAML upload / live editor)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_default_yaml_text() -> str:
    p = ROOT / "configs" / "aircraft_presets.yaml"
    if p.exists():
        return p.read_text(encoding="utf-8")
    # generate from built-in
    from nlg_guidance_sim.catalog.presets import _builtin_presets
    presets_list = [v.to_yaml_dict() for v in _builtin_presets().values()]
    return yaml.dump({"presets": presets_list}, allow_unicode=True, sort_keys=False)


def sidebar_preset_editor() -> dict[str, AircraftPreset]:
    """Sidebar section for YAML-based preset management."""
    with st.sidebar.expander("\U0001f6e0\ufe0f Editor de presets (YAML)", expanded=False):
        uploaded = st.file_uploader(
            "Subir aircraft_presets.yaml",
            type=["yaml", "yml"],
            help="Sube un archivo YAML con la misma estructura que configs/aircraft_presets.yaml",
        )
        default_txt = _load_default_yaml_text()
        yaml_text = st.text_area(
            "YAML en vivo (edita directamente)",
            value=(
                uploaded.read().decode("utf-8") if uploaded is not None
                else default_txt
            ),
            height=260,
            help="Modifica aquí para añadir aeronaves sin tocar el código.",
        )
        download_btn = st.download_button(
            "\u2b07\ufe0f Descargar YAML actual",
            data=yaml_text,
            file_name="aircraft_presets.yaml",
            mime="text/yaml",
        )
        try:
            raw = yaml.safe_load(yaml_text)
            loaded: dict[str, AircraftPreset] = {}
            for entry in raw.get("presets", []):
                p_obj = AircraftPreset(**entry)
                loaded[p_obj.name] = p_obj
            if loaded:
                st.success(f"\u2714\ufe0f {len(loaded)} presets cargados")
                return loaded
        except Exception as exc:
            st.error(f"YAML no v\u00e1lido: {exc}")
    # fallback
    return load_presets_yaml()


# ---------------------------------------------------------------------------
# Build scene + sensor from sidebar
# ---------------------------------------------------------------------------

def build_scene_and_sensor(
    presets: dict[str, AircraftPreset],
) -> tuple[Scene, RPLidar2DSim | None]:
    st.sidebar.title("NLG Guidance Sim")
    st.sidebar.caption("Escena geom\u00e9trica + LiDAR 2D sint\u00e9tico")

    preset_name = st.sidebar.selectbox("Preset de aeronave", list(presets.keys()))
    preset = presets[preset_name]

    st.sidebar.subheader("Configuraci\u00f3n del NLG")
    arrangement = st.sidebar.radio(
        "Ruedas del tren delantero",
        options=["single", "dual"],
        index=0 if preset.arrangement == "single" else 1,
        format_func=lambda v: "Una rueda" if v == "single" else "Dos ruedas",
    )
    tire_length_m = st.sidebar.slider("Longitud aparente del neum\u00e1tico [m]",
        0.40, 1.10, float(preset.tire_length_m), 0.01)
    tire_width_m = st.sidebar.slider("Anchura aparente del neum\u00e1tico [m]",
        0.10, 0.40, float(preset.tire_width_m), 0.01)
    shoulder_exponent = st.sidebar.slider("Perfil de hombro",
        2.0, 6.0, float(preset.shoulder_exponent), 0.1)
    track_width_m = st.sidebar.slider("Separaci\u00f3n entre ruedas [m]",
        0.00, 0.90,
        float(0.0 if arrangement == "single" else preset.track_width_m),
        0.01, disabled=arrangement == "single")

    st.sidebar.subheader("Plataforma")
    rail_gauge_m = st.sidebar.slider("Separaci\u00f3n entre ra\u00edles [m]",
        0.80, 2.00, float(preset.rail_gauge_m), 0.01)
    platform_length_m = st.sidebar.slider("Longitud de plataforma [m]",
        0.80, 3.50, float(preset.platform_length_m), 0.01)
    platform_width_m = st.sidebar.slider("Ancho de plataforma [m]",
        0.45, 1.60, float(preset.platform_width_m), 0.01)
    capture_width_m = st.sidebar.slider("Ancho de embocadura/cuna [m]",
        0.30, float(platform_width_m),
        float(min(preset.capture_width_m, platform_width_m)), 0.01)
    ramp_length_m = st.sidebar.slider("Longitud de rampa [m]",
        0.10, 1.00, float(preset.ramp_length_m), 0.01)

    st.sidebar.subheader("Pose del tren")
    center_x_m = st.sidebar.slider("Posici\u00f3n longitudinal X [m]",
        0.60, 6.00, float(preset.center_x_m), 0.01)
    center_y_m = st.sidebar.slider("Error lateral Y [m]",
        -0.60, 0.60, float(preset.center_y_m), 0.01)
    psi_deg = st.sidebar.slider("Oblicuidad \u03c8 [deg]",
        -20.0, 20.0, float(preset.psi_deg), 0.1)

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

    origin_x_m = st.sidebar.slider("LiDAR X [m]",
        -0.10, float(scene.capture_x_m + 0.20),
        float(scene.capture_x_m - 0.12), 0.01)
    origin_y_m = st.sidebar.slider("LiDAR Y [m]", -0.50, 0.50, 0.00, 0.01)
    angle_min_deg = st.sidebar.slider("FoV m\u00ednimo [deg]", -140, 0, -60, 1)
    angle_max_deg = st.sidebar.slider("FoV m\u00e1ximo [deg]", 0, 140, 60, 1)
    num_beams = st.sidebar.slider("N\u00famero de haces", 60, 1440, 360, 30)
    max_range_m = st.sidebar.slider("Alcance m\u00e1ximo [m]", 0.50, 10.00, 5.00, 0.10)
    noise_std_m = st.sidebar.slider("Ruido gaussiano [m]", 0.0, 0.02, 0.002, 0.001)

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


# ---------------------------------------------------------------------------
# Phase banner
# ---------------------------------------------------------------------------

def render_phase_banner(phase_state, pose_est=None) -> None:
    """Top-of-page colour banner showing current approach phase."""
    col_badge, col_desc, col_prog = st.columns([2, 5, 3])
    with col_badge:
        st.markdown(
            f'<span class="phase-badge" '
            f'style="background:{phase_state.color}20;color:{phase_state.color};'
            f'border:1.5px solid {phase_state.color}80;">'
            f'{phase_state.emoji} {phase_state.label}</span>',
            unsafe_allow_html=True,
        )
    with col_desc:
        st.caption(phase_state.description)
    with col_prog:
        phases = all_phase_states()
        phase_nums = [p.phase.value for p in phases]
        current_idx = phase_nums.index(phase_state.phase.value)
        cols = st.columns(len(phases))
        for i, (ph, col) in enumerate(zip(phases, cols)):
            active = i == current_idx
            done = i < current_idx
            marker = "\u2705" if done else (ph.emoji if active else "\u26aa")
            col.markdown(
                f'<div style="text-align:center;font-size:.78rem;'
                f'color:{ph.color if (active or done) else "#aaa9a5"};">'
                f'{marker}<br><b>F{ph.phase.value}</b></div>',
                unsafe_allow_html=True,
            )
    st.divider()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="NLG Guidance Sim",
        page_icon="\U0001f6de",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # --- preset editor (sidebar, collapsible) ---
    presets = sidebar_preset_editor()

    # --- build scene and sensor ---
    scene, lidar = build_scene_and_sensor(presets)
    scan = lidar.scan(scene) if lidar is not None else None

    # --- estimation pipeline ---
    lshape_result = None
    pose_est = None
    confidence = 0.0

    if scan is not None:
        hits = scan.ordered_hit_points()
        if len(hits) >= 10:
            fitter = LShapeFitter(min_pts_per_arm=5)
            lshape_result = fitter.fit(hits)
            if lshape_result is not None:
                pose_est = estimate_pose(lshape_result,
                                         guide_axis_y=scene.guide_axis_y_m)
                confidence = pose_est.confidence

    # --- phase classification ---
    midpoint = scene.nlg.midpoint
    phase_state = classify_phase(
        x_m=float(midpoint[0]),
        Y_m=float(pose_est.Y_m) if pose_est else float(midpoint[1]),
        psi_deg=float(pose_est.psi_deg) if pose_est else math.degrees(scene.nlg.psi_rad),
        lidar_confidence=confidence,
    )

    # --- header ---
    st.title("Simulador interactivo del tren delantero")
    st.caption(
        "Geometr\u00eda param\u00e9trica del NLG \u00b7 plataforma guiada por ra\u00edles \u00b7 "
        "LiDAR 2D sint\u00e9tico \u00b7 estimaci\u00f3n Y/\u03c8 \u00b7 m\u00e1quina de fases"
    )
    render_phase_banner(phase_state, pose_est=pose_est)

    # --- tabs ---
    tab_scene, tab_lidar, tab_est, tab_phases, tab_diag = st.tabs(
        ["Escena", "LiDAR 2D", "Estimaci\u00f3n Y/\u03c8", "Fases", "Diagn\u00f3stico"]
    )

    with tab_scene:
        fig = plot_scene(
            scene, scan_result=scan,
            lshape_result=lshape_result,
            pose_estimate=pose_est,
        )
        st.pyplot(fig, use_container_width=True)

    with tab_lidar:
        if scan is None:
            st.info("Activa el LiDAR en la barra lateral para visualizar la nube ordenada.")
        else:
            fig = plot_ranges(scan)
            st.pyplot(fig, use_container_width=True)

    with tab_est:
        if pose_est is None:
            st.warning(
                "No hay suficientes puntos de impacto para el fitting L-shape. "
                "Ajusta el FoV o la posici\u00f3n del LiDAR para ver los neum\u00e1ticos."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Y estimada", f"{pose_est.Y_m:+.4f} m",
                          delta=f"{pose_est.Y_m - midpoint[1]:+.4f} m vs real")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("\u03c8 estimado", f"{pose_est.psi_deg:+.3f}\u00b0",
                          delta=f"{pose_est.psi_deg - math.degrees(scene.nlg.psi_rad):+.3f}\u00b0 vs real")
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confianza L-shape", f"{pose_est.confidence:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RMS fitting", f"{lshape_result.total_rms*1000:.2f} mm")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            if lshape_result is not None:
                fig_dbg = plot_estimation_debug(scan, lshape_result, pose_est)
                st.pyplot(fig_dbg, use_container_width=True)

    with tab_phases:
        st.subheader("M\u00e1quina de fases de aproximaci\u00f3n")
        for ph in all_phase_states():
            is_active = ph.phase == phase_state.phase
            border = f"3px solid {ph.color}" if is_active else f"1px solid {ph.color}40"
            bg = f"{ph.color}14" if is_active else "#faf9f6"
            st.markdown(
                f'<div style="border:{border};border-radius:12px;padding:.8rem 1.2rem;'
                f'margin-bottom:.6rem;background:{bg};">'
                f'<b style="color:{ph.color};">{ph.emoji} {ph.label}</b>'
                f'{" &nbsp;\u25c4 <i>ACTIVA</i>" if is_active else ""}'
                f'<br><span style="font-size:.9rem;color:#5a574f;">{ph.description}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.subheader("Umbrales de transici\u00f3n")
        th = PhaseThresholds()
        col_a, col_b = st.columns(2)
        col_a.markdown(
            f"| Par\u00e1metro | Valor |\n|---|---|\n"
            f"| Inicio Fase 2 (X \u2264) | {th.phase2_entry_x_m} m |\n"
            f"| Inicio Fase 3 (X \u2264) | {th.phase3_entry_x_m} m |\n"
            f"| Tolerancia Y | \u00b1{th.y_ok_m} m |\n"
            f"| Tolerancia \u03c8 | \u00b1{th.psi_ok_deg}\u00b0 |\n"
            f"| Confianza m\u00edn. LiDAR | {th.confidence_min} |"
        )
        col_b.markdown(
            """
            **L\u00f3gica de transici\u00f3n**

            - **F1 \u2192 F2**: X \u2264 3.5 m **Y** confianza L-shape \u2265 0.40
            - **F2 \u2192 F3**: X \u2264 0.8 m **Y** |Y| \u2264 0.05 m **Y** |\u03c8| \u2264 3\u00b0
            - Si los criterios no se cumplen, el sistema permanece en la fase anterior.
            """
        )

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
            st.metric("Oblicuidad \u03c8", f"{summary['psi_deg']:.2f} deg")
            st.markdown("</div>", unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Ruedas", str(summary["n_wheels"]))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Resumen de escena")
            st.json(summary)
        with right:
            st.subheader("Pipeline de estimaci\u00f3n")
            if pose_est:
                st.json({
                    "Y_est_m": round(pose_est.Y_m, 6),
                    "psi_est_deg": round(pose_est.psi_deg, 4),
                    "confidence": round(pose_est.confidence, 4),
                    "rms_total_mm": round(lshape_result.total_rms * 1000, 3),
                    "corner_angle_deg": round(lshape_result.corner_angle_deg, 2),
                    "split_index": lshape_result.split_index,
                    "n_pts_arm1": len(lshape_result.pts1),
                    "n_pts_arm2": len(lshape_result.pts2),
                })
            else:
                st.info("Sin estimaci\u00f3n disponible (pocos puntos de impacto).")


if __name__ == "__main__":
    main()
