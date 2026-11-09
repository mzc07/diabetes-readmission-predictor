"""Features nuevas, preprocesamiento (escalado + one-hot) y selección de las top 20."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Las 23 columnas de medicamentos (No / Steady / Up / Down)
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]  # fmt: skip

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_num", "visitas_previas", "n_meds_activos", "n_meds_cambiados",
]  # fmt: skip

CATEGORICAL_COLS = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "diag_group", "max_glu_serum",
    "A1Cresult", "insulin", "metformin", "change", "diabetesMed",
]  # fmt: skip

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def age_to_number(age: str) -> float:
    """'[70-80)' -> 75 (punto medio del rango)."""
    low, high = age.strip("[)").split("-")
    return (int(low) + int(high)) / 2


def diag_group(code) -> str:
    """Agrupa el código ICD-9 de diag_1 en 9 grupos."""
    if pd.isna(code) or str(code).startswith(("V", "E")):
        return "Otros"
    n = float(code)
    if 250 <= n < 251:
        return "Diabetes"
    if 390 <= n <= 459 or n == 785:
        return "Circulatorio"
    if 460 <= n <= 519 or n == 786:
        return "Respiratorio"
    if 520 <= n <= 579 or n == 787:
        return "Digestivo"
    if 580 <= n <= 629 or n == 788:
        return "Genitourinario"
    if 140 <= n <= 239:
        return "Neoplasias"
    if 710 <= n <= 739:
        return "Musculoesqueletico"
    if 800 <= n <= 999:
        return "Lesiones"
    return "Otros"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las columnas de FEATURE_COLS (incluye las 3 features nuevas)."""
    out = pd.DataFrame(index=df.index)
    out["age_num"] = df["age"].map(age_to_number)
    out["diag_group"] = df["diag_1"].map(diag_group)
    out["visitas_previas"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    out["n_meds_activos"] = (df[MED_COLS] != "No").sum(axis=1)
    out["n_meds_cambiados"] = df[MED_COLS].isin(["Up", "Down"]).sum(axis=1)

    for col in NUMERIC_COLS:
        if col not in out:
            out[col] = df[col]
    for col in CATEGORICAL_COLS:
        if col not in out:
            out[col] = df[col]

    # Nulo = "no medido" (es información, no un error)
    out["max_glu_serum"] = out["max_glu_serum"].fillna("No medido")
    out["A1Cresult"] = out["A1Cresult"].fillna("No medido")
    for col in CATEGORICAL_COLS:
        out[col] = out[col].fillna("Desconocido").astype(str)
    return out[FEATURE_COLS]


def make_preprocessor() -> ColumnTransformer:
    """StandardScaler para numéricas + OneHot para categóricas.

    max_categories=11 deja las 10 categorías más frecuentes de cada columna y agrupa
    el resto como "infrequent" (equivale a "top 10 + Otros"). Se calcula solo con train.
    """
    onehot = OneHotEncoder(
        handle_unknown="infrequent_if_exist", max_categories=11, sparse_output=False
    )
    return ColumnTransformer(
        [("num", StandardScaler(), NUMERIC_COLS), ("cat", onehot, CATEGORICAL_COLS)]
    )


def make_selector(k: int = 20) -> SelectFromModel:
    """Se queda con las k columnas más importantes según un Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=42
    )
    return SelectFromModel(rf, max_features=k, threshold=-np.inf)


def importance_table(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Importancia de cada columna (ya transformada) y su importancia acumulada.

    Sirve para el gráfico que justifica quedarse con las top 20.
    """
    pre = make_preprocessor()
    Xt = pre.fit_transform(X_train)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=42
    ).fit(Xt, y_train)
    table = pd.DataFrame(
        {"feature": pre.get_feature_names_out(), "importance": rf.feature_importances_}
    )
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table["acumulada"] = table["importance"].cumsum() / table["importance"].sum()
    return table
