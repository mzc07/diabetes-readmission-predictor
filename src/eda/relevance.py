"""
Archivo: relevance.py
Nombre: Juliana Rueda Perez (2251801)
Descripción: Ranking de relevancia de variables usando mutual information y Cramér's V.
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def mutual_info_ranking(
    df: pd.DataFrame, numeric_cols: list[str], target_binary_col: str
) -> pd.DataFrame:
    """Mutual information entre cada variable numérica y el target binario."""
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y = df[target_binary_col].astype(int)
    mi = mutual_info_classif(X, y, random_state=42, discrete_features=False)
    return (
        pd.DataFrame({"columna": numeric_cols, "mutual_info": mi})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )


def cramers_v(confusion_matrix) -> float:
    """Cramér's V a partir de una tabla de contingencia."""
    from scipy.stats import chi2_contingency

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return (chi2 / (n * (min(r, k) - 1))) ** 0.5


def cramers_v_ranking(
    df: pd.DataFrame, categorical_cols: list[str], target_col: str
) -> pd.DataFrame:
    """Cramér's V entre cada variable categórica y el target."""
    rows = []
    for col in categorical_cols:
        tabla = pd.crosstab(df[col].fillna("__missing__"), df[target_col])
        if tabla.shape[0] < 2 or tabla.shape[1] < 2:
            continue
        rows.append({"columna": col, "cramers_v": round(cramers_v(tabla), 3)})
    return (
        pd.DataFrame(rows)
        .sort_values("cramers_v", ascending=False)
        .reset_index(drop=True)
    )
