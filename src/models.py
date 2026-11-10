"""Clasificadores (4 clásicos + MLP), regresores (3), búsqueda de hiperparámetros y CV."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.features import make_preprocessor, make_selector

SEED = 42

# Cuadrículas pequeñas (HU-09). Las claves llevan el prefijo del paso del Pipeline ("clf__").
PARAM_GRIDS = {
    "DecisionTree": {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [20, 50]},
    "RandomForest": {"clf__n_estimators": [100, 200], "clf__min_samples_leaf": [10, 20, 40]},
    "SVM": {"clf__C": [0.5, 1.0, 2.0]},
    "MLP": {"clf__alpha": [1e-4, 1e-2], "clf__hidden_layer_sizes": [(64, 32), (32, 16)]},
    "DecisionTreeRegressor": {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [20, 50]},
    "RandomForestRegressor": {
        "clf__n_estimators": [100, 200],
        "clf__min_samples_leaf": [10, 20, 40],
    },
    "SVR": {"clf__C": [0.5, 1.0, 2.0]},
}


class BalancedMLP(MLPClassifier):
    """MLPClassifier no tiene class_weight: duplica al azar los casos positivos hasta
    igualar a los negativos solo al entrenar (en cada fit, o sea solo con datos de train)."""

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        extra = rng.choice(pos, size=max(len(neg) - len(pos), 0), replace=True)
        idx = rng.permutation(np.concatenate([np.arange(len(y)), extra]))
        return super().fit(np.asarray(X)[idx], y[idx])


def build_pipeline(est) -> Pipeline:
    """preprocesar (escalar) -> seleccionar top 20 -> modelo. Todo se ajusta solo con train."""
    return Pipeline([("prep", make_preprocessor()), ("select", make_selector(20)), ("clf", est)])


def get_models() -> dict[str, Pipeline]:
    """4 clasificadores clásicos + red neuronal (MLP).

    GaussianNB no soporta class_weight; el MLP balancea con BalancedMLP (sobremuestreo).
    """
    return {
        "GaussianNB": build_pipeline(GaussianNB()),
        "RandomForest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=SEED,
            )  # fmt: skip
        ),
        "DecisionTree": build_pipeline(
            DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=SEED
            )
        ),
        "SVM": build_pipeline(SVC(kernel="rbf", class_weight="balanced", random_state=SEED)),
        "MLP": build_pipeline(
            BalancedMLP(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                validation_fraction=0.1,
                max_iter=300,
                random_state=SEED,
            )
        ),
    }


def get_regressors() -> dict[str, Pipeline]:
    """3 regresores sobre el target ordinal (NO=0, >30=1, <30=2)."""
    return {
        "DecisionTreeRegressor": build_pipeline(
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=50, random_state=SEED)
        ),
        "RandomForestRegressor": build_pipeline(
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=20, random_state=SEED
            )
        ),
        "SVR": build_pipeline(SVR(kernel="rbf")),
    }


def tune_model(name: str, model: Pipeline, X, y, groups, regression: bool = False) -> Pipeline:
    """GridSearchCV pequeño (3 folds agrupados por paciente). Devuelve el mejor Pipeline ya
    ajustado. Clasificación: F1 de la clase '<30'. Regresión: RMSE."""
    if regression:
        cv, scoring = GroupKFold(n_splits=3), "neg_root_mean_squared_error"
    else:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        scoring = "f1"
    search = GridSearchCV(model, PARAM_GRIDS[name], cv=cv, scoring=scoring, n_jobs=-1)
    search.fit(X, y, groups=groups)
    print(f"  {name}: {search.best_params_} ({scoring} = {search.best_score_:.3f})")
    return search.best_estimator_


def cross_validate_models(models: dict, X, y, groups, n_splits: int = 5) -> pd.DataFrame:
    """F1 de la clase '<30' en cada fold (folds agrupados por paciente).
    Devuelve tabla larga: model, fold, f1."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="f1", n_jobs=-1)
        rows += [{"model": name, "fold": i + 1, "f1": s} for i, s in enumerate(scores)]
        print(f"  {name}: F1 medio = {scores.mean():.3f} (±{scores.std():.3f})")
    return pd.DataFrame(rows)
