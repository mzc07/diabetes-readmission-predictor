from pathlib import Path

import pandas as pd

DATA_RAW_PATH = Path('data/raw/diabetic_data.csv')

def read_diabetes_csv(diabetes_file_path:Path=DATA_RAW_PATH):
    if not diabetes_file_path:
        raise ValueError('No se pudo leer el dataset')

    DATAFRAME_DIABETES = pd.read_csv(f"{diabetes_file_path}",
    sep=",",
    encoding="utf-8",
    index_col=0)
    return DATAFRAME_DIABETES

if __name__ == '__main__':
   working_dataframe = read_diabetes_csv(DATA_RAW_PATH)
