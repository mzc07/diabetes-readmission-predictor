# Diabetes Readmission Predictor

<!-- [Equipo] Banner 800x300: guardar como docs/assets/banner.png -->
<p align="center"><img src="docs/assets/banner.png" width="800" height="300" alt="Banner del proyecto"></p>

## 👥 Autores
Nombre Apellido 1, Nombre Apellido 2, Nombre Apellido 3

## 🎯 Objetivo
Predecir si un paciente diabético será readmitido en menos de 30 días tras su alta hospitalaria.

## 📊 Dataset
**Diabetes 130-US Hospitals for Years 1999-2008** (UCI Machine Learning Repository): más de 100.000 encuentros hospitalarios de pacientes diabéticos en 130 hospitales de EE. UU., con variables demográficas, clínicas y de medicación.

🔗 Descarga: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## 🤖 Modelos
Clasificación: Gaussian Naive Bayes, Decision Tree, Random Forest, SVM, Red Neuronal (MLP).
Regresión (target ordinal NO=0, >30=1, <30=2): Decision Tree Regressor, Random Forest Regressor, SVR.
Métricas: accuracy, precision, F1 (más recall y ROC-AUC), MSE, MAE, RMSE.
Criterio de selección: F1 de la clase `<30` en validación cruzada (desempate por recall); en regresión, menor RMSE.

## 🔗 Enlaces
- 💻 Código: https://github.com/USUARIO/diabetes-readmission-predictor/tree/main/notebooks
- 🎥 Video: https://youtu.be/XXXXXXXX
- 📦 Repositorio: https://github.com/USUARIO/diabetes-readmission-predictor

## 🖼 Imagen del proyecto
<!-- [Equipo] Imagen 800x300 con título del proyecto e información de los estudiantes: docs/assets/portada.png -->
<p align="center"><img src="docs/assets/portada.png" width="800" height="300" alt="Imagen del proyecto"></p>

## 🎬 Video
<!-- [Equipo] Máximo 5 minutos. Entregar el archivo y alojarlo también en YouTube -->
[Ver en YouTube](https://youtu.be/XXXXXXXX) · Archivo: `video/presentacion.mp4`

## 📑 Diapositivas
[Descargar diapositivas](docs/assets/diapositivas.pdf)

---

## 🚀 Inicio Rápido

### Prerrequisitos
- Docker y Docker Compose (recomendado), **o** Python 3.12+ y [uv](https://docs.astral.sh/uv/)

### Con Docker
```bash
git clone https://github.com/USUARIO/diabetes-readmission-predictor
cd diabetes-readmission-predictor
docker compose up --build
# Abrir http://localhost:8888 y ejecutar los notebooks en orden
# (en 01_eda.ipynb se descarga el dataset automáticamente)
docker compose --profile app up app   # opcional: app Streamlit en http://localhost:8501
```

### Con uv
```bash
uv sync
uv run python -m src.data          # descarga el dataset a data/raw
uv run jupyter lab
uv run python train.py             # alternativa: corre todo el pipeline y genera artifacts/
uv run streamlit run app.py        # app con los resultados (requiere haber corrido train.py)
```

## 📓 Notebooks
1. `01_eda.ipynb`: análisis exploratorio
2. `02_preprocesamiento.ipynb`: limpieza, features, split sin fuga, escalado y selección
3. `03_modelado.ipynb`: clasificación, regresión y red neuronal
4. `04_evaluacion_seleccion.ipynb`: métricas, gráficas y selección del mejor modelo

## 🗂 Estructura
`data/raw/` (no versionado) · `data/processed/` · `notebooks/` · `src/` (data, features, models, evaluation) · `tests/` · `artifacts/` · `reports/figures/` · `docs/assets/` · `video/`

## 🧪 Tests y calidad
```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pre-commit install   # opcional: ruff antes de cada commit
```

> ⚠️ Proyecto educativo: no sirve para decisiones clínicas reales.

## 📄 Licencia
MIT
