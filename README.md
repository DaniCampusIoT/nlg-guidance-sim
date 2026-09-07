# NLG Guidance Sim 🛞

Simulador interactivo 2D para estudiar la aproximación del **Nose Landing Gear (NLG)**
hacia una plataforma guiada por raíles. Integra geometría paramétrica de neumáticos,
LiDAR 2D sintético (modelo Slamtec S2E), estimación de pose en dos etapas
(L-shape coarse + Levenberg-Marquardt), filtro de Kalman extendido con modelo cinemático
CVTR y una máquina de estados de fases de aproximación.

---

## Índice

1. [Requisitos](#1-requisitos)
2. [Instalación paso a paso](#2-instalación-paso-a-paso)
3. [Ejecución](#3-ejecución)
4. [Interfaz — 7 pestañas](#4-interfaz--7-pestañas)
5. [Arquitectura de módulos](#5-arquitectura-de-módulos)
6. [Pipeline de estimación](#6-pipeline-de-estimación)
7. [Presets de aeronave](#7-presets-de-aeronave)
8. [Tests](#8-tests)
9. [Parámetros configurables](#9-parámetros-configurables)
10. [Próximos pasos](#10-próximos-pasos)

---

## 1. Requisitos

| Componente | Versión mínima |
|---|---|
| Python | 3.10 |
| pip | 23+ |
| Sistema operativo | Windows 10 / Ubuntu 22.04 / macOS 12 |

Dependencias declaradas en `requirements.txt` y `pyproject.toml`:

```
streamlit
matplotlib
numpy
scipy
pyyaml
```

---

## 2. Instalación paso a paso

```bash
# 1. Clonar el repositorio
git clone https://github.com/DaniCampusIoT/nlg-guidance-sim.git
cd nlg-guidance-sim

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
#    Windows
.venv\Scripts\activate
#    Linux / macOS
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Instalar el paquete en modo editable
pip install -e .
```

El paso 5 registra `nlg_guidance_sim` como paquete importable sin necesidad
de manipular `PYTHONPATH` manualmente.

---

## 3. Ejecución

```bash
streamlit run app.py
```

Streamlit abrirá automáticamente el navegador en `http://localhost:8501`.
Si el puerto está ocupado, usa:

```bash
streamlit run app.py --server.port 8502
```

Para exponer la aplicación en red local:

```bash
streamlit run app.py --server.address 0.0.0.0
```

---

## 4. Interfaz — 7 pestañas

La barra lateral contiene **todos los controles paramétricos**. Las pestañas
muestran los resultados en tiempo real.

### Barra lateral

| Sección | Parámetros |
|---|---|
| **Preset de aeronave** | Selector de aeronave (built-in + YAML) |
| **Configuración del NLG** | Ruedas (single/dual), longitud y anchura del neumático, exponente de hombro, separación de ruedas |
| **Plataforma** | Separación de raíles, longitud/ancho de plataforma, ancho de embocadura, longitud de rampa |
| **Pose del tren** | Posición X [m], error lateral Y [m], oblicuidad ψ [deg] |
| **LiDAR 2D · Slamtec S2E** | Activar/desactivar, posición XY del sensor, FoV mín/máx, número de haces (hasta 3 200), alcance [m], ruido gaussiano [m] |

### Pestaña `Escena`

Vista 2D cenital de la plataforma + raíles + neumático(s) + nube LiDAR.
Superpone el rectángulo estimado por L-shape fitting si hay puntos suficientes.

### Pestaña `LiDAR 2D`

Gráfico polar con los rangos medidos por cada haz. Permite visualizar la nube
ordenada tal como la vería un RPLidar físico.

### Pestaña `Estimación Y/ψ`

Métricas del estimador L-shape (Zhang CISRAM 2015):

- **Y estimada** — desplazamiento lateral en metros, con delta respecto al valor real.
- **ψ estimada** — oblicuidad en grados, con delta respecto al valor real.
- **RMSE fit** — error cuadrático medio del ajuste de rectas [cm].
- **Confianza** — indicador numérico + método usado (`lshape`, `line` o `centroid`).

Incluye expansor con detalle del algoritmo de búsqueda angular.

### Pestaña `🔧 Fitting`

Pipeline completo L-shape → Levenberg-Marquardt → EKF:

- **Semáforo de fase** — indicador visual de los 4 estados (APPROACH / ALIGN / CAPTURE / HOLD).
- **Métricas LM** — Y, ψ, RMSE [cm], iteraciones, tiempo [ms], estado de convergencia y Δcosto.
- **Prior rígido** — W y L del preset fijados como constantes; solo se optimizan (xc, yc, θ).
- **Escena con rectángulo LM ajustado** — superposición del resultado refinado sobre la escena.
- **EKF CVTR — Historial Y / ψ** — evolución temporal de las estimaciones filtradas.
- **Estado EKF actual** — tabla con Y, ψ, velocidad v y velocidad angular ω del filtro.
- Botón **Reiniciar EKF** para limpiar el historial.

### Pestaña `🚦 Fases`

Máquina de estados de la aproximación:

| Estado | Color | Condición de entrada |
|---|---|---|
| `APPROACH` | Naranja | Estado inicial; X > umbral ALIGN |
| `ALIGN` | Azul | X ≤ umbral ALIGN y \|Y\| o \|ψ\| dentro de banda |
| `CAPTURE` | Verde | X ≤ umbral CAPTURE y \|Y\| y \|ψ\| dentro de banda estrecha |
| `HOLD` | Verde azulado | NLG centrado y estable en embocadura |

Muestra parámetros de transición en tiempo real (X, \|Y\|, \|ψ\|) y un diagrama
de fases con la activa resaltada. Botón **Reiniciar FSM** disponible.

### Pestaña `Editor YAML`

Editor de texto para añadir nuevos presets de aeronave sin modificar código Python.

**Estructura mínima de un preset:**

```yaml
presets:
  "Mi aeronave":
    arrangement: single          # "single" o "dual"
    tire_length_m: 0.65
    tire_width_m: 0.20
    shoulder_exponent: 3.5
    track_width_m: 0.00          # 0.0 si arrangement=single
    rail_gauge_m: 1.15
    platform_length_m: 1.30
    platform_width_m: 0.72
    capture_width_m: 0.50
    ramp_length_m: 0.25
    center_x_m: 2.00
    center_y_m: 0.00
    psi_deg: 0.0
```

Pulsa **Aplicar presets** para cargarlos en caliente. Los presets aparecen
inmediatamente en el selector de la barra lateral sin reiniciar la aplicación.

### Pestaña `Diagnóstico`

Métricas de resumen de la escena (midpoint X, error lateral Y, ψ, número
de ruedas) y tabla de estado completo del sistema: fase, estimaciones,
RMSE, iteraciones LM, tiempo de pipeline y estado del EKF.

---

## 5. Arquitectura de módulos

```
nlg-guidance-sim/
├── app.py                          ← Entrypoint Streamlit (7 tabs + sidebar)
├── configs/
│   └── presets.yaml                ← Presets de aeronave editables en caliente
├── data/
│   └── generated/                  ← Datos de escenarios generados
├── scripts/                        ← Scripts de utilidad y batch
├── tests/
│   ├── test_smoke.py               ← Prueba de importación y construcción básica
│   ├── test_estimation.py          ← Validación numérica del estimador L-shape y LM
│   └── test_phases.py              ← Validación de transiciones de la FSM
└── src/nlg_guidance_sim/
    ├── catalog/
    │   ├── presets.py              ← Presets Python built-in (dict PRESETS)
    │   └── yaml_loader.py          ← Carga, valida y serializa YAML de presets
    ├── geometry/
    │   ├── profiles.py             ← Superelipse paramétrica del perfil 2D del neumático
    │   └── nlg_model.py            ← Modelo NLG single/dual wheel con pose (x, y, ψ)
    ├── world/
    │   └── scene.py                ← Plataforma + cuna + raíles + NLG integrado
    ├── sensors/
    │   └── rplidar2d.py            ← Ray casting con ruido gaussiano (modelo Slamtec S2E)
    ├── estimation/
    │   ├── lshape.py               ← L-shape fitting coarse (Zhang CISRAM 2015)
    │   └── pose_ekf.py             ← EKF con modelo cinemático CVTR (Cellina 2025)
    ├── fitting/
    │   └── pipeline.py             ← Pipeline L-shape → LM; devuelve PipelineResult
    ├── phases/
    │   └── state.py                ← FSM ApproachPhase (APPROACH→ALIGN→CAPTURE→HOLD)
    ├── control/                    ← (en desarrollo) control de plataforma
    └── viz/
        ├── plotting.py             ← plot_scene / plot_ranges / plot_estimation
        └── fitting_plots.py        ← plot_fitting / plot_ekf_history
```

### Descripción de módulos clave

**`geometry/profiles.py` — `Tire2DProfile`**
Genera el contorno 2D del neumático mediante una superelipse:
`(x/a)^n + (y/b)^n = 1`. El exponente de hombro (`shoulder_exponent`)
controla la transición entre la zona plana de la banda de rodadura y el
flanco lateral.

**`geometry/nlg_model.py` — `NLGModel`**
Ensambla uno o dos perfiles `Tire2DProfile` con la pose (x, y, ψ) del tren.
Para configuración dual, aplica la separación `track_width_m` respecto al
eje central.

**`world/scene.py` — `Scene`**
Define la geometría completa de la plataforma: raíles, cuna de captura
y rampa. Expone `summary_dict()` con todas las variables dimensionales.

**`sensors/rplidar2d.py` — `RPLidar2DSim`**
Ray casting angular sobre la escena. Cada haz se lanza desde la posición del
sensor y registra el primer impacto con el contorno del NLG o la plataforma.
El ruido gaussiano se aplica sobre la distancia medida. El método
`scan.ordered_hit_points()` devuelve los puntos ordenados por ángulo, listos
para el estimador.

**`estimation/lshape.py` — `fit_lshape`**
Búsqueda L-shape: barre N_THETA orientaciones en [0°, 90°) y minimiza la
suma de distancias cuadráticas perpendiculares a los dos lados del rectángulo.
Fallback automático a ajuste de recta (`line`) si hay menos de 2 segmentos
válidos, o a centroide (`centroid`) si hay menos de 8 puntos.

**`fitting/pipeline.py` — `run_pipeline`**
```python
result: PipelineResult = run_pipeline(
    hit_points,   # np.ndarray (N, 2) — puntos ordenados del LiDAR
    width_m,      # prior rígido: anchura del neumático
    length_m,     # prior rígido: longitud del neumático
)
# result.Y_m          — desplazamiento lateral estimado
# result.psi_deg      — oblicuidad estimada [deg]
# result.refined      — RefinedPose: xc, yc, theta_rad, rmse, n_iter, converged, cost_final
# result.coarse       — LShapeResult del paso 1
# result.elapsed_ms   — tiempo total del pipeline
# result.n_points_used
```

**`estimation/pose_ekf.py` — `PoseEKF`**
EKF con estado `[x, y, θ, v, ω]` y modelo cinemático CVTR
(Constant Velocity + Turn Rate). Métodos principales:
- `ekf.initialize(xc, yc, theta_rad)` — inicializa el filtro con la primera estimación LM.
- `ekf.predict(dt)` — propagación cinemática.
- `ekf.update(xc, yc)` — corrección con medición LM.
- `ekf.state` — `EKFState(x, y, theta, v, omega)`.
- `ekf.history` — lista de estados para visualización temporal.

**`phases/state.py` — `ApproachPhase`**
FSM con 4 estados. Las transiciones dependen de:
- `x_to_capture` — distancia restante hasta la embocadura [m].
- `Y_m` — error lateral estimado [m].
- `psi_rad` — oblicuidad estimada [rad].

Métodos de consulta: `phase.label()`, `phase.color()`,
`phase.description()`, `phase.progress()`.

---

## 6. Pipeline de estimación

```
LiDAR 2D sintético
      │  scan.ordered_hit_points()
      ▼
┌─────────────────────────────────────────┐
│  Etapa 1 — L-shape Search               │
│  Zhang et al., CISRAM 2015              │
│  Búsqueda angular θ ∈ [0°, 90°)        │
│  Criterio: suma residuos ⊥ al rect.     │
│  Fallbacks: line → centroid             │
└───────────────┬─────────────────────────┘
                │ LShapeResult (seed xc, yc, θ)
                ▼
┌─────────────────────────────────────────┐
│  Etapa 2 — Levenberg-Marquardt          │
│  Prior rígido: W, L del preset          │
│  Estado libre: [xc, yc, θ]             │
│  Residuo: dist. ⊥ al lado más cercano   │
│  scipy least_squares, method='lm'       │
└───────────────┬─────────────────────────┘
                │ RefinedPose
                ▼
┌─────────────────────────────────────────┐
│  Etapa 3 — EKF CVTR                     │
│  Cellina et al., arXiv 2025             │
│  Estado: [x, y, θ, v, ω]               │
│  Medición: [xc, yc] de LM              │
└───────────────┬─────────────────────────┘
                │ EKFState
                ▼
        ApproachPhase FSM
```

---

## 7. Presets de aeronave

Los presets built-in están definidos en `src/nlg_guidance_sim/catalog/presets.py`.
Para añadir nuevos presets sin tocar Python, edita `configs/presets.yaml`
o usa el **Editor YAML** en la pestaña correspondiente de la UI.

El archivo `configs/presets.yaml` se carga automáticamente al iniciar la
aplicación. Los presets YAML tienen precedencia sobre los built-in si
comparten el mismo nombre.

---

## 8. Tests

```bash
# Instalar pytest (si no está instalado)
pip install pytest

# Ejecutar todos los tests
pytest tests/ -v

# Ejecutar un fichero concreto
pytest tests/test_estimation.py -v

# Ejecutar con cobertura (requiere pytest-cov)
pip install pytest-cov
pytest tests/ --cov=src/nlg_guidance_sim --cov-report=term-missing
```

| Fichero | Qué valida |
|---|---|
| `test_smoke.py` | Importación de todos los módulos y construcción básica de `Scene` + `RPLidar2DSim` |
| `test_estimation.py` | Error numérico del estimador L-shape y del pipeline LM sobre escenas sintéticas conocidas |
| `test_phases.py` | Transiciones correctas de la FSM ante secuencias de (X, Y, ψ) controladas |

---

## 9. Parámetros configurables

### LiDAR sintético (`rplidar2d.py`)

| Parámetro | Descripción | Rango típico |
|---|---|---|
| `origin_x_m` | Posición X del sensor en la escena | −0.1 … captura+0.2 m |
| `origin_y_m` | Posición Y del sensor | −0.5 … 0.5 m |
| `angle_min_deg` | Inicio del campo de visión | −140 … 0° |
| `angle_max_deg` | Fin del campo de visión | 0 … 140° |
| `num_beams` | Número de haces (S2E: 3 200 @ 10 Hz) | 60 … 3 200 |
| `max_range_m` | Alcance máximo | 0.05 … 30 m |
| `range_noise_std_m` | Desviación estándar del ruido gaussiano | 0 … 0.05 m |

### Preset de aeronave

| Campo | Descripción |
|---|---|
| `arrangement` | `"single"` (rueda única) o `"dual"` (par de ruedas) |
| `tire_length_m` | Longitud del neumático [m] |
| `tire_width_m` | Anchura del neumático [m] |
| `shoulder_exponent` | Exponente de la superelipse (hombro del neumático) |
| `track_width_m` | Separación entre ruedas en configuración dual [m] |
| `rail_gauge_m` | Separación entre raíles de la plataforma [m] |
| `platform_length_m` | Longitud de la plataforma [m] |
| `platform_width_m` | Anchura de la plataforma [m] |
| `capture_width_m` | Anchura de la embocadura de captura [m] |
| `ramp_length_m` | Longitud de la rampa de entrada [m] |
| `center_x_m` | Posición X inicial del NLG [m] |
| `center_y_m` | Error lateral inicial Y [m] |
| `psi_deg` | Oblicuidad inicial ψ [°] |

---

## 10. Próximos pasos

- Integración con datos reales de RPLidar A1/A2 vía puerto serie.
- Exportación de escenarios a JSON para reproducibilidad y benchmarking.
- Sensores de corto alcance para validación fina en Fase 3 (A121, BGT60TR13C).
- Controlador de plataforma en `src/nlg_guidance_sim/control/`.
- Interfaz de replay sobre datos grabados.

---

## Referencias

- Zhang, X. et al. (2015). *L-shape fitting for vehicle pose estimation*. CISRAM 2015.
- Cellina, F. et al. (2025). *EKF CVTR for mobile target tracking*. arXiv 2025.
- MDPI (2026). *Levenberg-Marquardt pose refinement for LiDAR-based NLG detection*.

---

## Licencia

Proyecto académico/experimental. Sin licencia comercial asignada.
