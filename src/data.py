"""Descarga, carga, limpieza y split de los datos (Diabetes 130-US hospitals)."""

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
UCI_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

# discharge_disposition_id: fallecido u hospicio. Estos pacientes no pueden reingresar.
DECEASED_IDS = {11, 13, 14, 19, 20, 21}

# Target de regresión (ADR-02): readmitted como escala ordinal.
READMITTED_ORDINAL = {"NO": 0, ">30": 1, "<30": 2}


def download_data(raw_dir: Path | str = RAW_DIR, force: bool = False) -> None:
    """Descarga diabetic_data.csv e IDS_mapping.csv del repositorio UCI a data/raw."""
    raw_dir = Path(raw_dir)
    if not force and (raw_dir / "diabetic_data.csv").exists():
        print(f"Ya existe {raw_dir / 'diabetic_data.csv'}; nada que descargar.")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {UCI_URL} ...")
    with urlopen(UCI_URL, timeout=120) as resp:  # noqa: S310 (URL fija y de confianza)
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    archive.extractall(raw_dir)
    print(f"Listo: {sorted(p.name for p in raw_dir.iterdir())}")


def load_data(path: Path | str = RAW_DIR / "diabetic_data.csv") -> pd.DataFrame:
    """Lee el CSV. En este dataset '?' significa dato faltante."""
    return pd.read_csv(path, na_values="?", low_memory=False)


def load_ids_mapping(path: Path | str = RAW_DIR / "IDS_mapping.csv") -> dict[str, pd.DataFrame]:
    """IDS_mapping.csv trae 3 tablas (admission_type, discharge_disposition, admission_source)
    separadas por una línea en blanco. Devuelve {nombre_del_id: DataFrame(id, description)}."""
    tables, name, rows = {}, None, []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line == ",":
            if name and rows:
                tables[name] = pd.DataFrame(rows, columns=["id", "description"])
            name, rows = None, []
        elif line.endswith("_id,description"):
            name = line.split(",")[0]
        elif name:
            key, _, desc = line.partition(",")
            rows.append((int(key), desc.strip('" ')))
    if name and rows:
        tables[name] = pd.DataFrame(rows, columns=["id", "description"])
    return tables


def build_target(readmitted: pd.Series) -> pd.Series:
    """Clasificación: '<30' -> 1 (reingresa en menos de 30 días); 'NO' y '>30' -> 0."""
    return (readmitted == "<30").astype(int)


def build_target_reg(readmitted: pd.Series) -> pd.Series:
    """Regresión: NO=0, >30=1, <30=2 (escala ordinal de gravedad del reingreso)."""
    return readmitted.map(READMITTED_ORDINAL).astype(int)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Quita fallecidos/hospicio y agrega 'target' (binario) y 'target_reg' (ordinal).

    Se conservan todos los encuentros: la fuga por paciente se evita en el split
    con grupos (ver split_train_test), no borrando filas.
    """
    df = df[~df["discharge_disposition_id"].isin(DECEASED_IDS)].reset_index(drop=True)
    df["target"] = build_target(df["readmitted"])
    df["target_reg"] = build_target_reg(df["readmitted"])
    return df


def split_train_test(
    y: pd.Series, groups: pd.Series, test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Split estratificado por target y agrupado por paciente (patient_nbr).

    Un mismo paciente nunca queda en train y test a la vez. Usa el primer fold de
    StratifiedGroupKFold (con test_size=0.2 son 5 folds -> ~80/20).
    Devuelve (índices_train, índices_test) posicionales.
    """
    n_splits = round(1 / test_size)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(cv.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def group_subsample(y: pd.Series, groups: pd.Series, n: int, seed: int = 42) -> np.ndarray:
    """~n filas agrupadas por paciente (para SVM/SVR, que son lentos). Índices posicionales."""
    if n >= len(y):
        return np.arange(len(y))
    n_splits = max(2, round(len(y) / n))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, idx = next(cv.split(np.zeros(len(y)), y, groups))
    return idx


if __name__ == "__main__":
    download_data()
