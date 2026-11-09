# Diabetes Readmission Predictor

<!-- [Equipo] Banner 800x300: guardar como docs/assets/banner.png -->
<p align="center"><img src="docs/assets/banner.png" width="800" height="300" alt="Banner del proyecto"></p>

## Autores
- Ivanna Álvarez - 2251805
- Martín Ariza - 2251516
- Juliana Rueda - 2251801

## Objetivo del Proyecto
Predecir si un paciente diabético será readmitido en menos de 30 días tras su alta hospitalaria.

## Dataset
**Diabetes 130-US Hospitals for Years 1999-2008** (UCI Machine Learning Repository): más de 100.000 encuentros hospitalarios de pacientes diabéticos en 130 hospitales de EE. UU., con variables demográficas, clínicas y de medicación.

 Descarga: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## Modelos Utilizados
#### Clasificación: 
- Gaussian Naive Bayes (GNB)
- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM)

#### Regresión: 
- Decision Tree Regressor
- Random Forest Regressor
- SVR
  
#### Métricas:
- recall
- precision
- F1
- MSE
- MAE
- RMSE
  
#### Criterio de selección:
- Clasificación: F1 de la clase `<30` (desempate por recall)
- Regresión: menor RMSE

## Enlaces Destacados
- Código: (https://github.com/mzc07/diabetes-readmission-predictor/tree/develop/notebooks)
- Video: INSERTAR LINK DE YUTU
- Repositorio: [https://github.com/USUARIO/diabetes-readmission-predictor](https://github.com/mzc07/diabetes-readmission-predictor/tree/develop)
- Diapositivas: INSERTAR LINK DE CANVA OSIOSI

---

## Inicio Rápido

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

## Acceso a los Notebooks
1. `01_eda.ipynb`: análisis exploratorio
2. `02_preprocesamiento.ipynb`: limpieza, features, split sin fuga, escalado y selección
3. `03_modelado.ipynb`: clasificación, regresión
4. `04_evaluacion.ipynb`: métricas, gráficas y selección del mejor modelo

## Estructura
`data/raw/` (no versionado) · `data/processed/` · `notebooks/` · `src/` (data, features, models, evaluation) · `tests/` · `artifacts/` · `reports/figures/` · `docs/assets/` · `video/`

## Tests y calidad
```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pre-commit install   # opcional: ruff antes de cada commit
```
## Uso del proyecto
> Proyecto educativo: no sirve para decisiones clínicas reales.

## Licencia
MIT
