"""
Archivo: univariate.py
Nombre: Juliana Rueda Perez (2251801)
Descripción: Funciones de análisis univariado (resumen de cada variable por separado).
"""

import pandas as pd


def numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """describe() junto con skew y kurtosis, todo en una sola tabla."""
    desc = df[numeric_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc["skew"] = df[numeric_cols].skew()
    desc["kurtosis"] = df[numeric_cols].kurt()
    return desc


def categorical_summary(
    df: pd.DataFrame, categorical_cols: list[str], top_n: int = 5
) -> pd.DataFrame:
    """Cardinalidad y categorías más frecuentes por columna categórica."""
    rows = []
    for col in categorical_cols:
        vc = df[col].value_counts(normalize=True, dropna=False)
        rows.append(
            {
                "columna": col,
                "cardinalidad": df[col].nunique(dropna=True),
                "top_categorias": dict(vc.head(top_n).round(3)),
            }
        )
    return pd.DataFrame(rows)


def target_distribution(
    df: pd.DataFrame, target_col: str = "readmitted"
) -> pd.DataFrame:
    """Distribución del target original y de la versión binaria (readmitted_lt30)."""
    original = (
        df[target_col].value_counts(normalize=True).round(4).to_frame("proporcion")
    )
    original.index.name = "readmitted (original, 3 clases)"

    binaria = (
        (df[target_col] == "<30")
        .value_counts(normalize=True)
        .round(4)
        .to_frame("proporcion")
    )
    binaria.index = binaria.index.map({True: "readmitido <30 días", False: "resto"})
    binaria.index.name = "readmitted_lt30 (target binario)"

    return pd.concat([original, binaria], keys=["original", "binario"])
