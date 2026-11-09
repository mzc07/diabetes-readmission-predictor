"""Los tests clave. No necesitan los CSV reales: usan datos pequeños inventados."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.data import (
    DECEASED_IDS,
    build_target,
    build_target_reg,
    clean_data,
    split_train_test,
)
from src.features import CATEGORICAL_COLS, NUMERIC_COLS, make_preprocessor
from src.models import build_pipeline


def test_target_binario():
    out = build_target(pd.Series(["<30", ">30", "NO", "<30"]))
    assert out.tolist() == [1, 0, 0, 1]
    assert out.isna().sum() == 0
    # target de regresión (ADR-02): NO=0, >30=1, <30=2
    assert build_target_reg(pd.Series(["<30", ">30", "NO"])).tolist() == [2, 1, 0]


def test_limpieza_y_split_sin_fuga():
    n = 400
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": rng.integers(0, 120, n),  # varios encuentros por paciente
            "discharge_disposition_id": rng.choice([1, 1, 1, 11], n),  # 11 = falleció
            "readmitted": rng.choice(["NO", ">30", "<30"], n, p=[0.5, 0.35, 0.15]),
        }
    )
    out = clean_data(raw)
    assert not out["discharge_disposition_id"].isin(DECEASED_IDS).any()
    assert len(out) < n  # se quitaron los fallecidos

    tr, te = split_train_test(out["target"], out["patient_nbr"])
    # un mismo paciente nunca está en train y test a la vez
    assert set(out["patient_nbr"].iloc[tr]).isdisjoint(out["patient_nbr"].iloc[te])


def _toy_X(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(5, 3, (n, len(NUMERIC_COLS))), columns=NUMERIC_COLS)
    for col in CATEGORICAL_COLS:
        X[col] = rng.choice(["a", "b", "c"], n)
    return X


def test_pipeline_probabilidades():
    X = _toy_X()
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(X)))

    # las columnas numéricas escaladas quedan con media ~0
    Xt = make_preprocessor().fit_transform(X)
    assert np.allclose(Xt[:, : len(NUMERIC_COLS)].mean(axis=0), 0, atol=1e-6)

    # el pipeline completo entrena y da probabilidades entre 0 y 1
    model = build_pipeline(DecisionTreeClassifier(max_depth=3)).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
