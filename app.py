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
from nlg_guidance_sim.geometry.nlg_model import NLGModel
from nlg_guidance_sim.geometry.profiles import Tire2DProfile
from nlg_guidance_sim.sensors.rplidar2d import RPLidar2DSim
from nlg_guidance_sim.viz.plotting import plot_ranges, plot_scene
from nlg_guidance_sim.world.scene import Scene


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
        .small-note {
            color: #66625a;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_scene_and_sensor() -> tuple[Scene, RPLidar2DSim | None]:
    st.sidebar.title("NLG Guidance Sim")
    st.sidebar.caption("Escena geom\u00e9trica + LiDAR 2D sint\u00e9tico")

    preset_name = st.sidebar.selectbox("Preset de aeronave", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    st.sidebar.subheader("Configuraci\u00f3n del NLG")
    arrangement = st.sidebar.radio(
        "Ruedas del tren delantero",
        options=["single", "dual"],
        index=0 if preset.arrangement == "single" else 1,
        format_func=lambda v: "Una rueda" if v == "single" else "Dos ruedas",
    )

    tire_length_m = st.sidebar.slider(
        "Longitud aparente del neum\u00e1tico [m]",
        min_value=0.40,
        max_value=1.10,
        value=float(preset.tire_length_m),
        step=0.01,
    )
    tire_width_m = st.sidebar.slider(
        "Anchura aparente del neum\u00e1tico [m]",
        min_value=0.10,
        max_value=0.40,
        value=float(preset.tire_width_m),
        step=0.01,
    )
    shoulder_exponent = st.sidebar.slider(
        "Perfil de hombro",
        min_value=2.0,
        max_value=6.0,
        value=float(preset.shoulder_exponent),
        step=0.1,
    )
    track_width_m = st.sidebar.slider(
        "Separaci\u00f3n entre ruedas [m]",
        min_value=0.00,
        max_value=0.90,
        value=float(0.0 if arrangement == "single" else preset.track_width_m),
        step=0.01,
        disabled=arrangement == "single",
    )

    st.sidebar.subheader("Plataforma")
    rail_gauge_m = st.sidebar.slider(
        "Separaci\u00f3n entre ra\u00edles [m]",
        min_value=0.80,
        max_value=2.00,
        value=float(preset.rail_gauge_m),
        step=0.01,
    )
    platform_length_m = st.sidebar.slider(
        "Longitud de plataforma [m]",
        min_value=0.80,
        max_value=3.50,
        value=float(preset.platform_length_m),
        step=0.01,
    )
    platform_width_m = st.sidebar.slider(
        "Ancho de plataforma [m]",
        min_value=0.45,
        max_value=1.60,
        value=float(preset.platform_width_m),
        step=0.01,
    )
    capture_width_m = st.sidebar.slider(
        "Ancho de embocadura/cuna [m]",
        min_value=0.30,
        max_value=float(platform_width_m),
        value=float(min(preset.capture_width_m, platform_width_m)),
        step=0.01,
    )
    ramp_length_m = st.sidebar.slider(
        "Longitud de rampa [m]",
        min_value=0.10,
        max_value=1.00,
        value=float(preset.ramp_length_m),
        step=0.01,
    )

    st.sidebar.subheader("Pose del tren")
    center_x_m = st.sidebar.slider(
        "Posici\u00f3n longitudinal X [m]",
        min_value=0.60,
        max_value=6.00,
        value=float(preset.center_x_m),
        step=0.01,
    )
    center_y_m = st.sidebar.slider(
        "Error lateral Y [m]",
        min_value=-0.60,
        max_value=0.60,
        value=float(preset.center_y_m),
        step=0.01,
    )
    psi_deg = st.sidebar.slider(
        "Oblicuidad \u03c8 [deg]",
        min_value=-20.0,
        max_value=20.0,
        value=float(preset.psi_deg),
        step=0.1,
    )

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

    origin_x_m = st.sidebar.slider(
        "LiDAR X [m]",
        min_value=-0.10,
        max_value=float(scene.capture_x_m + 0.20),
        value=float(scene.capture_x_m - 0.12),
        step=0.01,
    )
    origin_y_m = st.sidebar.slider(
        "LiDAR Y [m]",
        min_value=-0.50,
        max_value=0.50,
        value=0.00,
        step=0.01,
    )
    angle_min_deg = st.sidebar.slider(
        "FoV m\u00ednimo [deg]",
        min_value=-140,
        max_value=0,
        value=-60,
        step=1,
    )
    angle_max_deg = st.sidebar.slider(
        "FoV m\u00e1ximo [deg]",
        min_value=0,
        max_value=140,
        value=60,
        step=1,
    )
    num_beams = st.sidebar.slider(
        "N\u00famero de haces",
        min_value=60,
        max_value=1440,
        value=360,
        step=30,
    )
    max_range_m = st.sidebar.slider(
        "Alcance m\u00e1ximo [m]",
        min_value=0.50,
        max_value=10.00,
        value=5.00,
        step=0.10,
    )
    noise_std_m = st.sidebar.slider(
        "Ruido gaussiano [m]",
        min_value=0.0,
        max_value=0.02,
        value=0.002,
        step=0.001,
    )

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


def main() -> None:
    st.set_page_config(
        page_title="NLG Guidance Sim",
        page_icon="\U0001f6de",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    st.title("Simulador interactivo del tren delantero")
    st.caption(
        "Geometr\u00eda param\u00e9trica del NLG, plataforma guiada por ra\u00edles y LiDAR 2D sint\u00e9tico "
        "preparado para futuras etapas de fitting y c\u00e1lculo de Y / \u03c8."
    )

    scene, lidar = build_scene_and_sensor()
    scan = lidar.scan(scene) if lidar is not None else None

    tab_scene, tab_lidar, tab_diag = st.tabs(
        ["Escena", "LiDAR 2D", "Diagn\u00f3stico"]
    )

    with tab_scene:
        fig = plot_scene(scene, scan_result=scan)
        st.pyplot(fig, use_container_width=True)

    with tab_lidar:
        if scan is None:
            st.info("Activa el LiDAR en la barra lateral para visualizar la nube ordenada.")
        else:
            fig = plot_ranges(scan)
            st.pyplot(fig, use_container_width=True)

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
            st.subheader("Notas")
            st.markdown(
                """
                - El modelo de rueda es param\u00e9trico y no circular.
                - La plataforma se puede ensanchar o estrechar.
                - Se puede simular una o dos ruedas.
                - El LiDAR devuelve puntos ordenados por \u00e1ngulo.
                - La base est\u00e1 lista para a\u00f1adir fitting de Y y \u03c8.
                """
            )


if __name__ == "__main__":
    main()
