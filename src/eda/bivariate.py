"""
Archivo: bivariate.py
Nombre: Juliana Rueda Perez (2251801)
Descripción: Funciones de análisis bivariado (cada variable contra target o entre sí).
"""

import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

def correlation_matrix(
    df: pd.DataFrame, numeric_cols: list[str], method: str = "spearman"
) -> pd.DataFrame:
    """Matriz de correlación entre las columnas numéricas. Spearman por defecto."""
    return df[numeric_cols].corr(method=method)


def high_correlation_pairs(
    corr_matrix: pd.DataFrame, threshold: float = 0.8
) -> pd.DataFrame:
    """Pares de variables con correlación mayor al threshold."""
    pairs = []
    cols = corr_matrix.columns
    for i, col_i in enumerate(cols):
        for col_j in cols[i + 1 :]:
            r = corr_matrix.loc[col_i, col_j]
            if abs(r) > threshold:
                pairs.append(
                    {
                        "variable_1": col_i,
                        "variable_2": col_j,
                        "correlacion": round(r, 3),
                    }
                )
    return (
        pd.DataFrame(pairs).sort_values("correlacion", key=abs, ascending=False)
        if pairs
        else pd.DataFrame(columns=["variable_1", "variable_2", "correlacion"])
    )


def categorical_vs_target_chi2(df, target_col):
    results = []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if col == target_col:
            continue
        # Omitir columnas con un solo valor único (varianza cero)
        if df[col].nunique(dropna=True) <= 1:
            continue
        contingency_table = pd.crosstab(df[col], df[target_col])
        # Validar que no existan filas o columnas que sumen cero
        if (contingency_table.sum(axis=1) == 0).any() or (
            contingency_table.sum(axis=0) == 0
        ).any():
            continue
        chi2, p_val, dof, _ = chi2_contingency(contingency_table)
        results.append({"variable": col, "chi2": chi2, "p_value": p_val, "dof": dof})
    return pd.DataFrame(results).sort_values(by="p_value")


def numeric_vs_target_test(
    df: pd.DataFrame, numeric_cols: list[str], target_binary_col: str
) -> pd.DataFrame:
    """Test de Mann-Whitney entre cada variable numérica y el target binario."""
    rows = []
    grupo_positivo = df[df[target_binary_col]]
    grupo_negativo = df[~df[target_binary_col]]
    for col in numeric_cols:
        stat, p = stats.mannwhitneyu(
            grupo_positivo[col].dropna(),
            grupo_negativo[col].dropna(),
            alternative="two-sided",
        )
        rows.append(
            {
                "columna": col,
                "mediana_readmitido_lt30": grupo_positivo[col].median(),
                "mediana_resto": grupo_negativo[col].median(),
                "p_valor": p,
            }
        )
    return pd.DataFrame(rows).sort_values("p_valor")
