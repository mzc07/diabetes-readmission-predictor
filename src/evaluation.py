"""Métricas y gráficos."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

FIG_DIR = Path("reports/figures")


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=120)
    plt.close(fig)


# ---------- Puntaje de riesgo y métricas ----------


def risk_score(model: Pipeline, X) -> np.ndarray:
    """Probabilidad de reingreso. SVM no da probabilidad (probability=False por velocidad):
    se usa decision_function pasada por una sigmoide para dejarla entre 0 y 1."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(X)))


def compute_metrics(models: dict, X_test, y_test) -> pd.DataFrame:
    """Accuracy, precision, recall, F1 (clase '<30') y AUC de cada modelo en test."""
    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "auc": roc_auc_score(y_test, risk_score(model, X_test)),
            }
        )
    return pd.DataFrame(rows).round(4)


def compute_regression_metrics(models: dict, X_test, y_test) -> pd.DataFrame:
    """MSE, MAE y RMSE de cada regresor en test."""
    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        rows.append(
            {
                "model": name,
                "mse": mse,
                "mae": mean_absolute_error(y_test, pred),
                "rmse": float(np.sqrt(mse)),
            }
        )
    return pd.DataFrame(rows).round(4)


# ---------- Gráficos ----------


def plot_cv_boxplot(cv_results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=cv_results, x="model", y="f1", ax=ax)
    ax.set(title="F1 (clase <30) por modelo, 5 folds", xlabel="", ylabel="F1")
    _save(fig, "cv_boxplot.png")


def plot_cv_lines(cv_results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=cv_results, x="fold", y="f1", hue="model", marker="o", ax=ax)
    ax.set(title="F1 por fold", xticks=sorted(cv_results["fold"].unique()))
    _save(fig, "cv_lineas.png")


def plot_confusion_matrices(models: dict, X_test, y_test) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8))
    for ax, (name, model) in zip(np.atleast_1d(axes), models.items(), strict=True):
        ConfusionMatrixDisplay.from_predictions(
            y_test, model.predict(X_test), ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
    _save(fig, "matrices_confusion.png")


def _threshold_curves(y_test, scores):
    prec, rec, thr = precision_recall_curve(y_test, scores)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return thr, prec, rec, f1


def plot_threshold_curves(models: dict, X_test, y_test) -> dict:
    """Una figura por modelo (umbral_<modelo>.png) con precision, recall y F1 vs. umbral.
    Devuelve los puntajes de cada modelo para reutilizarlos."""
    all_scores = {}
    for name, model in models.items():
        scores = risk_score(model, X_test)
        all_scores[name] = scores
        thr, prec, rec, f1 = _threshold_curves(y_test, scores)
        best = thr[np.argmax(f1)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(thr, prec, label="precision")
        ax.plot(thr, rec, label="recall")
        ax.plot(thr, f1, label="F1")
        ax.axvline(best, color="gray", ls="--", label=f"mejor F1 (umbral {best:.2f})")
        ax.set(title=f"{name}: métricas vs. umbral", xlabel="umbral", ylabel="valor")
        ax.legend()
        _save(fig, f"umbral_{name}.png")
    return all_scores


def plot_comparison(all_scores: dict, y_test) -> None:
    """Gráficos comparativos: F1 vs. umbral, curvas PR y ROC de todos los modelos."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, scores in all_scores.items():
        thr, _, _, f1 = _threshold_curves(y_test, scores)
        ax.plot(thr, f1, label=name)
    ax.set(title="F1 vs. umbral (todos los modelos)", xlabel="umbral", ylabel="F1")
    ax.legend()
    _save(fig, "comparativa_umbral_f1.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, scores in all_scores.items():
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ax.plot(rec, prec, label=name)
    ax.set(title="Curvas precision-recall", xlabel="recall", ylabel="precision")
    ax.legend()
    _save(fig, "comparativa_pr.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, scores in all_scores.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, scores):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":")
    ax.set(title="Curvas ROC", xlabel="tasa de falsos positivos", ylabel="recall")
    ax.legend()
    _save(fig, "comparativa_roc.png")


def plot_metrics_bars(metrics: pd.DataFrame) -> None:
    long = metrics.melt(id_vars="model", var_name="métrica", value_name="valor")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=long, x="métrica", y="valor", hue="model", ax=ax)
    ax.set(title="Métricas de clasificación en test", xlabel="")
    _save(fig, "comparativa_metricas.png")


def plot_regression_bars(metrics: pd.DataFrame) -> None:
    long = metrics.melt(id_vars="model", var_name="métrica", value_name="valor")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=long, x="métrica", y="valor", hue="model", ax=ax)
    ax.set(title="MSE / MAE / RMSE de los regresores en test", xlabel="")
    _save(fig, "comparativa_regresion.png")


def plot_loss_curve(model: Pipeline, name: str = "MLP") -> None:
    """Curva de pérdida de entrenamiento (y de validación interna) del MLP."""
    clf = model.steps[-1][1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(clf.loss_curve_, label="pérdida (train)")
    if getattr(clf, "validation_scores_", None) is not None:
        ax2 = ax.twinx()
        ax2.plot(clf.validation_scores_, color="C1", label="accuracy (validación)")
        ax2.set_ylabel("accuracy de validación")
        ax2.legend(loc="center right")
    ax.set(title=f"{name}: curva de pérdida", xlabel="época", ylabel="pérdida")
    ax.legend(loc="upper right")
    _save(fig, "mlp_loss_curve.png")


def plot_importance(table: pd.DataFrame, k: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(table) + 1), table["acumulada"])
    ax.axvline(k, color="red", ls="--", label=f"top {k}")
    ax.set(title="Importancia acumulada", xlabel="nº de columnas", ylabel="acumulada")
    ax.legend()
    _save(fig, "importancia_acumulada.png")


def plot_top_features(table: pd.DataFrame, k: int = 20) -> None:
    top = table.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set(title=f"Top {k} columnas más importantes (Random Forest)", xlabel="importancia")
    _save(fig, "top_features.png")


def make_eda_figures(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    """Tres figuras básicas para la pestaña 'Datos' de la app."""
    fig, ax = plt.subplots(figsize=(5, 4))
    y.map({0: "No reingresa <30", 1: "Reingresa <30"}).value_counts().plot.bar(ax=ax)
    ax.set(title=f"Distribución del target ({y.mean():.1%} positivos)", xlabel="")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, "eda_target.png")

    for col, fname, title in [
        ("age_num", "eda_edad.png", "Tasa de reingreso <30 por edad"),
        ("visitas_previas", "eda_visitas.png", "Tasa de reingreso <30 por visitas previas"),
    ]:
        x = X[col].clip(upper=8) if col == "visitas_previas" else X[col]
        fig, ax = plt.subplots(figsize=(7, 4))
        y.groupby(x).mean().plot.bar(ax=ax)
        ax.set(title=title, xlabel=col, ylabel="tasa")
        _save(fig, fname)

    fig, ax = plt.subplots(figsize=(7, 4))
    y.groupby(X["diag_group"]).mean().sort_values().plot.barh(ax=ax)
    ax.set(title="Tasa de reingreso <30 por grupo de diagnóstico", xlabel="tasa", ylabel="")
    _save(fig, "eda_diagnostico.png")
