"""
Archivo: bivariate.py
Nombre: Juliana Rueda Perez (2251801)
Descripción: Funciones de análisis bivariado (cada variable contra target o entre sí).
"""

import pandas as pd
from scipy import stats


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


def categorical_vs_target_chi2(
    df: pd.DataFrame, categorical_cols: list[str], target_col: str
) -> pd.DataFrame:
    """Test chi-cuadrado entre cada columna categórica y el target."""
    rows = []
    for col in categorical_cols:
        tabla = pd.crosstab(df[col], df[target_col])
        if tabla.shape[0] < 2 or tabla.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(tabla)
        rows.append({"columna": col, "chi2": round(chi2, 2), "p_valor": p, "dof": dof})
    return pd.DataFrame(rows).sort_values("chi2", ascending=False)


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
