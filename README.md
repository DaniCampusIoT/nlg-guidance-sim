# NLG Guidance Sim

Simulador interactivo 2D para estudiar la aproximación del nose landing gear (NLG)
hacia una plataforma guiada por raíles, con geometría paramétrica, LiDAR 2D sintético,
estimación de Y/ψ por fitting L-shape y máquina de fases de aproximación.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
pip install -e .
```

## Ejecución

```bash
streamlit run app.py
```

## Qué incluye (v0.2)

| Módulo | Funcionalidad |
|---|---|
| `catalog/presets.py` | Presets de aeronave — cargados desde YAML o built-in |
| `geometry/` | Perfil 2D de neumático (superelipse) + modelo single/dual wheel |
| `world/scene.py` | Plataforma, cuna, raíles |
| `sensors/rplidar2d.py` | LiDAR 2D sintético por ray casting con ruido gaussiano |
| `estimation/lshape.py` | Fitting L-shape TLS sobre nube 2D ordenada (CISRAM 2015) |
| `estimation/pose.py` | Estimación de Y y ψ a partir del resultado L-shape |
| `control/phases.py` | Máquina de fases Fase 1/2/3 con umbrales configurables |
| `viz/plotting.py` | Escena, rangos LiDAR, debug de estimación |

## Estructura de directorios

```
nlg-guidance-sim/
├─ app.py
├─ requirements.txt
├─ pyproject.toml
├─ configs/
│  └─ aircraft_presets.yaml      ← edita aquí para nuevas aeronaves
├─ .streamlit/config.toml
└─ src/nlg_guidance_sim/
   ├─ catalog/
   ├─ geometry/
   ├─ world/
   ├─ sensors/
   ├─ estimation/     ← lshape.py + pose.py
   ├─ control/        ← phases.py (FSM)
   └─ viz/
```

## Editor de presets YAML

Abre el expander **🛠️ Editor de presets** en la barra lateral para:
- Editar el YAML directamente en la UI
- Subir un archivo `aircraft_presets.yaml` propio
- Descargar el YAML actual para compartirlo

Estructura de un preset:
```yaml
presets:
  - name: "Mi aeronave"
    arrangement: dual          # single | dual
    tire_length_m: 0.72
    tire_width_m: 0.23
    shoulder_exponent: 3.8
    track_width_m: 0.48
    rail_gauge_m: 1.25
    platform_length_m: 1.45
    platform_width_m: 0.90
    capture_width_m: 0.65
    ramp_length_m: 0.32
    center_x_m: 2.10
    center_y_m: 0.10
    psi_deg: 3.5
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```
