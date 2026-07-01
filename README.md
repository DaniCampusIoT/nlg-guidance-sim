# NLG Guidance Sim

Simulador interactivo 2D para estudiar la aproximación del nose landing gear (NLG)
hacia una plataforma guiada por raíles, con geometría paramétrica y LiDAR 2D sintético.

## Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Ejecución

```bash
streamlit run app.py
```

## Qué incluye

- Presets de NLG (ligero, regional, single aisle, widebody)
- Configuración single / dual wheel
- Perfil 2D paramétrico de neumático (superelipse)
- Plataforma y cuna parametrizables
- LiDAR 2D sintético por ray casting
- Visualización de escena y nube ordenada

## Estructura

```
nlg-guidance-sim/
├─ app.py
├─ pyproject.toml
├─ requirements.txt
├─ .gitignore
├─ README.md
├─ .streamlit/
│  └─ config.toml
├─ src/
│  └─ nlg_guidance_sim/
│     ├─ __init__.py
│     ├─ catalog/
│     │  ├─ __init__.py
│     │  └─ presets.py
│     ├─ geometry/
│     │  ├─ __init__.py
│     │  ├─ profiles.py
│     │  └─ nlg_model.py
│     ├─ world/
│     │  ├─ __init__.py
│     │  └─ scene.py
│     ├─ sensors/
│     │  ├─ __init__.py
│     │  └─ rplidar2d.py
│     └─ viz/
│        ├─ __init__.py
│        └─ plotting.py
└─ tests/
   └─ test_smoke.py
```

## Próximos pasos sugeridos

- Ajuste geométrico de centros de rueda
- Estimación de Y y psi
- Estados Fase 1 / 2 / 3
- Sensores cortos de validación fina
- Exportación de escenarios
