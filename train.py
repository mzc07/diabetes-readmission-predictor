"""Corre todo el pipeline y escribe data/processed/, artifacts/ y reports/figures/.

    uv run python train.py

Los notebooks 03 y 04 usan estas mismas funciones de src/.
"""

import json
from pathlib import Path

import joblib

from src import evaluation as ev
from src.data import (
    PROCESSED_DIR,
    clean_data,
    download_data,
    group_subsample,
    load_data,
    split_train_test,
)
from src.features import add_features, importance_table
from src.models import SEED, cross_validate_models, get_models, get_regressors, tune_model

ART = Path("artifacts")
(ART / "models").mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("1/8 Cargando y limpiando...")
download_data()
df = clean_data(load_data())
X, y, y_reg, groups = add_features(df), df["target"], df["target_reg"], df["patient_nbr"]
print(f"  {len(df):,} encuentros, {groups.nunique():,} pacientes, {y.mean():.1%} <30 días")
ev.make_eda_figures(df, X, y)

print("2/8 Split 80/20 estratificado y agrupado por paciente...")
tr, te = split_train_test(y, groups, seed=SEED)
X_train, X_test = X.iloc[tr], X.iloc[te]
y_train, y_test = y.iloc[tr], y.iloc[te]
yr_train, yr_test = y_reg.iloc[tr], y_reg.iloc[te]
g_train = groups.iloc[tr]
assert set(groups.iloc[tr]).isdisjoint(groups.iloc[te]), "fuga: un paciente está en train y test"
X_train.assign(target=y_train, target_reg=yr_train, patient_nbr=g_train).to_csv(
    PROCESSED_DIR / "train.csv", index=False
)
X_test.assign(target=y_test, target_reg=yr_test, patient_nbr=groups.iloc[te]).to_csv(
    PROCESSED_DIR / "test.csv", index=False
)
imp = importance_table(X_train, y_train)
ev.plot_importance(imp)
ev.plot_top_features(imp)
imp.to_csv(ART / "importancia.csv", index=False)

print("3/8 Submuestra de ~20,000 filas de train (SVM y SVR son lentos)...")
sub = group_subsample(y_train, g_train, 20_000, seed=SEED)
X_sub, y_sub, yr_sub, g_sub = (
    X_train.iloc[sub],
    y_train.iloc[sub],
    yr_train.iloc[sub],
    g_train.iloc[sub],
)
print(f"  {len(X_sub):,} filas, {y_sub.mean():.1%} positivos")

print("4/8 GridSearchCV de clasificadores...")
models = {n: (tune_model(n, m, X_sub, y_sub, g_sub) if n != "GaussianNB" else m)
          for n, m in get_models().items()}  # fmt: skip

print("5/8 Validación cruzada de clasificadores (folds agrupados)...")
cv_results = cross_validate_models(models, X_sub, y_sub, g_sub)
cv_results.to_csv(ART / "cv_results.csv", index=False)
ev.plot_cv_boxplot(cv_results)
ev.plot_cv_lines(cv_results)

print("6/8 Métricas de clasificación en test...")
for model in models.values():
    model.fit(X_sub, y_sub)
metrics = ev.compute_metrics(models, X_test, y_test)
metrics.to_csv(ART / "metrics.csv", index=False)
ev.plot_confusion_matrices(models, X_test, y_test)
all_scores = ev.plot_threshold_curves(models, X_test, y_test)
ev.plot_comparison(all_scores, y_test)
ev.plot_metrics_bars(metrics)
ev.plot_loss_curve(models["MLP"])
print(metrics.to_string(index=False))
for name, model in models.items():
    joblib.dump(model, ART / "models" / f"{name}.joblib")

print("7/8 Regresores (GridSearchCV + métricas en test)...")
regressors = {n: tune_model(n, m, X_sub, yr_sub, g_sub, regression=True)
              for n, m in get_regressors().items()}  # fmt: skip
reg_metrics = ev.compute_regression_metrics(regressors, X_test, yr_test)
reg_metrics.to_csv(ART / "metrics_regression.csv", index=False)
ev.plot_regression_bars(reg_metrics)
print(reg_metrics.to_string(index=False))
for name, model in regressors.items():
    joblib.dump(model, ART / "models" / f"{name}.joblib")

print("8/8 Ganadores (ADR-03) y reentrenamiento del clasificador con todo el train...")
# Clasificación: mayor F1 medio en CV; si empatan (3 decimales), mayor recall en test.
rank = cv_results.groupby("model")["f1"].mean().round(3).rename("cv_f1").to_frame()
rank["recall"] = metrics.set_index("model")["recall"]
winner = rank.sort_values(["cv_f1", "recall"], ascending=False).index[0]
winner_reg = reg_metrics.sort_values("rmse").iloc[0]["model"]  # Regresión: menor RMSE

final = get_models()[winner].set_params(**{
    k: v for k, v in models[winner].get_params().items() if k.startswith("clf__")
}).fit(X_train, y_train)  # fmt: skip
scores = ev.risk_score(final, X_test)
joblib.dump(final, ART / "model.joblib")
final_metrics = ev.compute_metrics({winner: final}, X_test, y_test).iloc[0].to_dict()
(ART / "winner.json").write_text(
    json.dumps({"winner": winner, **final_metrics, "winner_regression": winner_reg}, indent=2)
)
print(f"  Clasificador ganador: {winner} -> {final_metrics}")
print(f"  Regresor ganador: {winner_reg}")

sample = X_test.assign(risk=scores, real=y_test.values)
sample = sample.sample(min(2000, len(sample)), random_state=SEED)
sample = sample.sort_values("risk", ascending=False).reset_index(names="id")
sample.to_csv(ART / "muestra_app.csv", index=False)
print("Listo.")
