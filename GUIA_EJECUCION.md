# Guía de ejecución y verificación

Pasos para ejecutar el proyecto (scripts, notebooks, Docker) y comprobar que funciona.
Todos los comandos se corren desde la raíz del repositorio.

## 0. Prerrequisitos

| Vía | Necesitas |
| --- | --- |
| Local | Python 3.12+ y [uv](https://docs.astral.sh/uv/) |
| Docker | Docker y Docker Compose |

Comprobar:

```bash
python3 --version   # >= 3.12
uv --version
docker --version && docker compose version
```

El dataset ya está versionado en `data/raw/diabetic_data.csv`. Si falta, se descarga
con `uv run python -m src.data` (necesita internet).

---

## 1. Instalación local

```bash
uv sync
```

**Verificación:** termina sin errores y existe la carpeta `.venv/`.

```bash
uv run python -c "import sklearn, pandas, streamlit; print('ok', sklearn.__version__)"
```

---

## 2. Tests y calidad de código (rápido, ~segundos)

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

**Esperado:**
- `pytest`: 3 tests pasan (`test_target_binario`, `test_limpieza_y_split_sin_fuga`,
  `test_pipeline_probabilidades`).
- `ruff`: sin errores.

---

## 3. Pipeline completo con `train.py`

```bash
uv run python train.py
```

Ejecuta 8 etapas y debe imprimir `1/8 ...` hasta `8/8 ...` y terminar con `Listo.`.
Puede tardar varios minutos (GridSearch con SVM/SVR sobre ~20.000 filas).

**Salidas que deben existir al terminar:**

| Ruta | Contenido |
| --- | --- |
| `data/processed/train.csv`, `test.csv` | Split 80/20 por paciente |
| `artifacts/metrics.csv` | Métricas de clasificación en test |
| `artifacts/metrics_regression.csv` | Métricas de regresión (MSE, MAE, RMSE) |
| `artifacts/cv_results.csv` | F1 por fold de validación cruzada |
| `artifacts/importancia.csv` | Importancia de variables |
| `artifacts/winner.json` | Clasificador y regresor ganadores |
| `artifacts/model.joblib` | Modelo final reentrenado |
| `artifacts/models/*.joblib` | Un modelo por algoritmo |
| `artifacts/muestra_app.csv` | Muestra de 2.000 filas para la app |
| `reports/figures/*.png` | Gráficas |

Comprobación rápida:

```bash
ls artifacts artifacts/models reports/figures
cat artifacts/winner.json
```

**Señales de que está bien:**
- En la etapa 2 no salta el `assert` de fuga (ningún paciente en train y test a la vez).
- `winner.json` tiene las claves `winner`, `recall`, `f1` y `winner_regression`.
- `artifacts/models/` contiene 5 clasificadores y 3 regresores.

> `artifacts/` está en `.gitignore`: se genera localmente y no se sube al repo.

---

## 4. Notebooks

### 4.1 Con JupyterLab local

```bash
uv run jupyter lab
```

Ejecuta **en orden** (Kernel → Restart & Run All en cada uno):

| # | Notebook | Qué verificar |
| --- | --- | --- |
| 1 | `notebooks/01_eda.ipynb` | Carga el dataset y muestra el análisis exploratorio sin errores. |
| 2 | `notebooks/02_preprocesamiento.ipynb` | Limpieza, features, split sin fuga, escalado y selección. Genera `data/processed/`. |
| 3 | `notebooks/03_modelado.ipynb` | Entrena clasificadores y regresores. Escribe `artifacts/metrics*.csv`, `cv_results.csv` y `artifacts/models/`. |
| 4 | `notebooks/04_evaluacion.ipynb` | Métricas, gráficas y elección del mejor modelo. |

El notebook 03 tarda varios minutos por los GridSearch.

### 4.2 Ejecución sin interfaz (recomendado para verificar)

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_eda.ipynb \
  notebooks/02_preprocesamiento.ipynb \
  notebooks/03_modelado.ipynb \
  notebooks/04_evaluacion.ipynb
```

**Esperado:** los 4 notebooks terminan sin `CellExecutionError`.
Ojo: `--inplace` sobrescribe las salidas guardadas; si no quieres cambiar los
archivos, usa `--output-dir /ruta/temporal` en lugar de `--inplace`.

### 4.3 Comprobaciones tras ejecutarlos

- No aparece el warning `sklearn.utils.parallel.delayed should be used with ...Parallel`
  en las celdas de `tune_model` (notebook 03).
- Los valores de `artifacts/metrics.csv` son coherentes con los de `train.py`
  (usan la misma semilla, `SEED`).

---

## 5. App Streamlit

Requiere haber corrido antes `train.py` (o los notebooks 03 y 04), porque lee `artifacts/` y
`reports/figures/`.

```bash
uv run streamlit run app.py
```

Abrir <http://localhost:8501>.

**Esperado:** carga la app con 3 pestañas, sin errores en pantalla ni en la terminal.

---

## 6. Docker

### 6.1 JupyterLab en contenedor

```bash
docker compose up --build
```

Abrir <http://localhost:8888> (sin token; solo para uso local) y ejecutar los notebooks en orden,
igual que en la sección 4.

**Verificación:**
- La construcción termina sin errores (`uv sync --frozen` instala las dependencias).
- JupyterLab abre y los notebooks corren con el kernel por defecto.
- Los archivos generados aparecen también en tu carpeta local, porque `.` está montado en `/app`.

### 6.2 App Streamlit en contenedor

Genera antes `artifacts/` y `reports/` (sección 3 o notebooks), porque se montan en modo lectura.

```bash
docker compose --profile app up app
```

Abrir <http://localhost:8501>.

### 6.3 Ejecutar scripts y tests dentro del contenedor

```bash
docker compose run --rm jupyter pytest
docker compose run --rm jupyter python train.py
```

### 6.4 Detener y limpiar

```bash
docker compose down
```

---

## 7. Lista de verificación final

- [ ] `uv sync` sin errores
- [ ] `pytest` en verde (3 tests)
- [ ] `ruff check` y `ruff format --check` sin errores
- [ ] `train.py` llega a `Listo.` y genera `artifacts/` y `reports/figures/`
- [ ] Los 4 notebooks se ejecutan de principio a fin sin errores
- [ ] No hay warnings de `sklearn.utils.parallel.delayed`
- [ ] La app Streamlit abre en el puerto 8501
- [ ] `docker compose up --build` levanta JupyterLab en el puerto 8888
- [ ] `docker compose --profile app up app` levanta la app

## 8. Problemas frecuentes

| Síntoma | Causa y solución |
| --- | --- |
| `FileNotFoundError: data/raw/diabetic_data.csv` | Corre `uv run python -m src.data` para descargarlo. |
| La app dice que faltan archivos en `artifacts/` | Aún no has corrido `train.py` ni los notebooks 03 y 04. |
| Rutas rotas en un notebook | Ábrelo desde la raíz del repo o desde `notebooks/`; la celda 1 ya ajusta el directorio. |
| `train.py` / notebook 03 muy lento | Es normal con SVM/SVR. Se submuestrean ~20.000 filas para limitarlo. |
| Puerto 8888 o 8501 ocupado | Cambia el mapeo en `docker-compose.yml` (por ejemplo `"8889:8888"`). |
| `uv: command not found` | Instala uv: <https://docs.astral.sh/uv/getting-started/installation/> |
