"""App Streamlit con 3 pestañas. Lee solo artifacts/ y reports/figures/ (no el CSV grande)."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ART, FIG = Path("artifacts"), Path("reports/figures")

st.set_page_config(page_title="Reingreso en diabéticos", layout="wide")
st.title("Reingreso hospitalario <30 días en pacientes diabéticos")
st.caption("Proyecto educativo: no sirve para decisiones clínicas reales.")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ART / name)


@st.cache_data
def load_winner() -> dict:
    return json.loads((ART / "winner.json").read_text())


def show(name: str, caption: str | None = None) -> None:
    path = FIG / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Falta {name}. Corre `python train.py`.")


tab_datos, tab_modelos, tab_riesgo = st.tabs(["Datos", "Modelos", "Pacientes en riesgo"])

with tab_datos:
    c1, c2 = st.columns(2)
    with c1:
        show("eda_target.png")
        show("eda_edad.png")
    with c2:
        show("eda_visitas.png")
        show("eda_diagnostico.png")

with tab_modelos:
    w = load_winner()
    st.success(f"Modelo ganador (mayor F1 medio en validación cruzada): **{w['winner']}**")
    st.dataframe(load_csv("metrics.csv"), hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        show("cv_boxplot.png")
    with c2:
        show("cv_lineas.png")
    show("matrices_confusion.png")
    st.subheader("Umbral de decisión por modelo")
    names = ["GaussianNB", "RandomForest", "DecisionTree", "SVM", "MLP"]
    for pair in (names[:2], names[2:4], names[4:]):
        cols = st.columns(2)
        for col, n in zip(cols, pair, strict=True):
            with col:
                show(f"umbral_{n}.png")
    st.subheader("Comparación de los clasificadores")
    c1, c2 = st.columns(2)
    with c1:
        show("comparativa_umbral_f1.png")
        show("comparativa_roc.png")
    with c2:
        show("comparativa_pr.png")
        show("comparativa_metricas.png")
    st.subheader("Red neuronal (MLP): curva de pérdida")
    show("mlp_loss_curve.png")
    st.subheader("Regresión (target ordinal NO=0, >30=1, <30=2)")
    st.success(f"Regresor ganador (menor RMSE): **{w.get('winner_regression', '-')}**")
    st.dataframe(load_csv("metrics_regression.csv"), hide_index=True)
    show("comparativa_regresion.png")

with tab_riesgo:
    muestra = load_csv("muestra_app.csv")
    c1, c2, c3 = st.columns(3)
    top_n = c1.slider("Top N pacientes", 5, 100, 20)
    edad = c2.slider("Edad (punto medio)", 5, 95, (5, 95))
    grupos = c3.multiselect(
        "Grupo de diagnóstico", sorted(muestra["diag_group"].unique()),
        default=sorted(muestra["diag_group"].unique()),
    )  # fmt: skip

    f = muestra[muestra["age_num"].between(*edad) & muestra["diag_group"].isin(grupos)]
    f = f.sort_values("risk", ascending=False).head(top_n)
    cols = ["id", "risk", "real", "age_num", "diag_group", "time_in_hospital", "visitas_previas"]
    st.dataframe(f[cols], hide_index=True)
