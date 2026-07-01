# NLG Guidance Sim

Simulador interactivo 2D para estudiar la aproximación del nose landing gear (NLG)
hacia una plataforma guiada por raíles, con geometría paramétrica, LiDAR 2D sintético,
estimación de Y/ψ por L-shape fitting y máquina de estados de fases de aproximación.

## Instalación

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Ejecución

```bash
streamlit run app.py
```

## Estructura de capas

```
nlg-guidance-sim/
├── app.py                          ← Streamlit UI (6 tabs)
├── configs/presets.yaml            ← Editor YAML de presets
└── src/nlg_guidance_sim/
    ├── catalog/
    │   ├── presets.py              ← Presets Python (built-in)
    │   └── yaml_loader.py          ← Carga/valida/serializa YAML
    ├── geometry/
    │   ├── profiles.py             ← Superelipse paramétrica
    │   └── nlg_model.py            ← Modelo single/dual wheel
    ├── world/scene.py              ← Plataforma + cuna + raíles
    ├── sensors/rplidar2d.py        ← Ray casting con ruido gaussiano
    ├── estimation/
    │   └── lshape.py               ← L-shape fitting (CISRAM 2015)
    ├── phases/
    │   └── state.py                ← FSM Fases 1/2/3/HOLD
    └── viz/plotting.py             ← Matplotlib + overlay estimación
tests/
    ├── test_smoke.py
    ├── test_estimation.py
    └── test_phases.py
```

## Qué incluye v0.2

- **Estimación Y/ψ** — L-shape fitting sobre nube 2D ordenada (método CISRAM 2015).
  Fallbacks automáticos a ajuste de recta y centroide.
- **Editor YAML** — añade presets de aeronave sin tocar código Python.
  Los presets se cargan en caliente y aparecen en la barra lateral.
- **Fases de aproximación** — FSM con 4 estados: APPROACH → ALIGN → CAPTURE → HOLD.
  Transiciones basadas en X, |Y| y |ψ|; regresión automática si el NLG se desvía.
  Reset manual desde la UI.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Próximos pasos sugeridos

- Integración con datos reales de RPLidar A1/A2
- Filtro de Kalman sobre estimaciones Y/ψ
- Exportación de escenarios a JSON para reproducibilidad
- Sensores cortos de validación fina (Fase 3)
