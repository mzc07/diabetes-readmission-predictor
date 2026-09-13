"""
Archivo: loader.py
Nombre: Juliana Rueda Perez (2251801)
Descripción: Carga y valida el dataset diabetic_data.csv desde disco.
"""

from pathlib import Path
import pandas as pd

# Columnas identificadoras
ID_COLUMNS = ["encounter_id", "patient_nbr"]

# Etiquetas reconocidas como "faltante" pero no codificadas como NaN
SENTINEL_VALUES = ["?", "None", "Not Available", "Not Mapped", "NULL"]


def load_diabetic_data(path: str | Path, treat_sentinels_as_na: bool = True) -> pd.DataFrame:
    """
    Carga diabetic_data.csv desde disco, reemplazando los centinelas conocidos
    por NaN real si `treat_sentinels_as_na` es True.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {path}")

    # Columnas donde "?" / "Not Available" / etc. SÍ significan faltante real
    sentinel_only_columns = [
        "race",
        "weight",
        "payer_code",
        "medical_specialty",
        "diag_1",
        "diag_2",
        "diag_3",
    ]

    # Reemplaza los centinelas conocidos por NaN real si `treat_sentinels_as_na` es True
    na_values = {col: SENTINEL_VALUES for col in sentinel_only_columns} if treat_sentinels_as_na else None

    # Carga el CSV con los valores NaN reemplazados si es necesario
    df = pd.read_csv(path, na_values=na_values, keep_default_na=True, low_memory=False)
    return df


def load_id_mappings(path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Parsea IDS_mapping.csv y devuelve un diccionario de DataFrames,
    uno por tabla de mapeo (admission_type_id, discharge_disposition_id, admission_source_id).
    """
    path = Path(path)
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)

    mappings: dict[str, pd.DataFrame] = {}
    current_name = None
    current_rows: list[list[str]] = []

    for row in raw.itertuples(index=False):
        row = list(row)
        # Fila en blanco -> separador entre tablas
        if all(v == "" or v is None for v in row):
            continue
        # Fila de encabezado de una nueva tabla, ej: ["admission_type_id", "description"]   
        if row[0].endswith("_id"):
            if current_name is not None:
                mappings[current_name] = pd.DataFrame(current_rows, columns=["id", "description"])
            current_name = row[0]
            current_rows = []
        else:
            current_rows.append(row[:2])

    if current_name is not None:
        mappings[current_name] = pd.DataFrame(current_rows, columns=["id", "description"])

    return mappings


def validate_schema(df: pd.DataFrame) -> list[str]:
    """
    Valida que las columnas esperadas existan y que las llaves
    identificadoras no tengan nulos. Devuelve una lista de strings 
    con los problemas encontrados,para reportarlos al notebook.
    """
    problems: list[str] = []

    expected_columns = ID_COLUMNS + ["age", "readmitted", "diag_1", "diag_2", "diag_3"]
    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        problems.append(f"Columnas esperadas ausentes: {missing_cols}")

    for id_col in ID_COLUMNS:
        if id_col in df.columns and df[id_col].isnull().any():
            problems.append(f"La columna identificadora '{id_col}' tiene valores nulos")

    if "encounter_id" in df.columns and df["encounter_id"].duplicated().any():
        n_dup = df["encounter_id"].duplicated().sum()
        problems.append(f"'encounter_id' debería ser único por fila, pero hay {n_dup} duplicados")

    if "readmitted" in df.columns:
        valores_validos = {"NO", "<30", ">30"}
        inesperados = set(df["readmitted"].dropna().unique()) - valores_validos
        if inesperados:
            problems.append(f"Valores inesperados en 'readmitted': {inesperados}")

    return problems
