"""
Archivo: quality.py
Nombre: Juliana Rueda Perez (2251801)
Descripción: Funciones de calidad de datos.
"""

import pandas as pd


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    % de nulos por columna, ordenado descendente.
    """
    pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    def sugerencia(p: float) -> str:
        if p > 90:
            return "eliminar"
        if p >= 30:
            return "evaluar binaria 'es_nulo'"
        if p > 0:
            return "imputar"
        return "sin faltantes"

    report = pct.to_frame("pct_faltante")
    report["accion_sugerida"] = report["pct_faltante"].apply(sugerencia)
    return report


def duplicate_report(df: pd.DataFrame, key_columns: list[str]) -> dict:
    """
    Reporta duplicados totales y duplicados por llave(s) de negocio.
    `patient_nbr` NO debe ser único — lo relevante es cuántos encuentros
    por paciente hay, para decidir el split agrupado.
    """
    result = {
        "filas_totales": len(df),
        "filas_duplicadas_completas": int(df.duplicated().sum()),
    }
    for col in key_columns:
        if col not in df.columns:
            continue
        conteo_por_llave = df[col].value_counts()
        result[f"{col}_valores_unicos"] = int(conteo_por_llave.shape[0])
        result[f"{col}_max_repeticiones"] = int(conteo_por_llave.max())
        result[f"{col}_pacientes_con_mas_de_1_encuentro"] = int(
            (conteo_por_llave > 1).sum()
        )
    return result


def sentinel_value_scan(
    df: pd.DataFrame, candidates: list[str] | None = None
) -> pd.DataFrame:
    """
    Busca valores centinela típicos que hayan sobrevivido en columnas
    categóricas tipo object, incluso después de la conversión a NaN aplicada en el loader.
    """
    if candidates is None:
        candidates = [
            "?",
            "Unknown/Invalid",
            "None",
            "NULL",
            "Not Available",
            "Not Mapped",
            "-1",
            "9999",
        ]

    rows = []
    for col in df.select_dtypes(include="object").columns:
        counts = df[col].value_counts(dropna=False)
        encontrados = {v: int(c) for v, c in counts.items() if v in candidates}
        if encontrados:
            rows.append({"columna": col, "centinelas_encontrados": encontrados})
    return pd.DataFrame(rows)


def numeric_out_of_range_report(
    df: pd.DataFrame, rules: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """
    Verifica rangos físicamente imposibles en columnas numéricas.
    """
    rows = []
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        fuera_rango = df[(df[col] < lo) | (df[col] > hi)]
        rows.append(
            {
                "columna": col,
                "rango_esperado": (lo, hi),
                "min_real": df[col].min(),
                "max_real": df[col].max(),
                "filas_fuera_de_rango": len(fuera_rango),
            }
        )
    return pd.DataFrame(rows)
