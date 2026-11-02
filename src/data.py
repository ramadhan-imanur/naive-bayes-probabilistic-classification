"""
Modul Pra-pemrosesan Data & Ingesti Klinis.

Menyediakan fungsi untuk:
1. Memuat dataset tabular (CSV).
2. Imputasi nilai nol fisiologis tidak valid dengan nilai median positif.
3. Partisi data terstratifikasi (Train-Test Split).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def resolve_default_dataset_path(filename: str = "diabetes_pima.csv") -> Path:
    """
    Menemukan path file dataset secara cerdas dari berbagai konteks eksekusi.
    """
    current_dir = Path(__file__).resolve().parent.parent
    candidate_paths = [
        current_dir / "data" / "raw" / filename,
        current_dir / "data" / filename,
        current_dir.parent / "code" / "data" / "raw" / filename,
        current_dir.parent / "dataset" / filename,
        Path.cwd() / "data" / "raw" / filename,
        Path.cwd() / "data" / filename,
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    return candidate_paths[0]


def load_dataset(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Memuat dataset CSV ke dalam pandas DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan pada path: {path}")
    return pd.read_csv(path)


def clean_clinical_data(
    df: pd.DataFrame,
    zero_as_missing_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Membersihkan data klinis dengan menangani nilai nol yang secara biologis tidak mungkin.

    Pada dataset klinis diabetes (Pima Indians), nilai 0 pada Glucose, BloodPressure,
    SkinThickness, Insulin, dan BMI merepresentasikan pengukuran yang hilang (missing value).
    Fungsi ini mengganti nilai 0 tersebut dengan nilai median dari pengamatan valid (> 0).
    """
    if zero_as_missing_cols is None:
        zero_as_missing_cols = DEFAULT_ZERO_AS_MISSING

    df_cleaned = df.copy()
    for col in zero_as_missing_cols:
        if col in df_cleaned.columns:
            valid_mask = df_cleaned[col] > 0
            if valid_mask.any():
                median_val = df_cleaned.loc[valid_mask, col].median()
                df_cleaned[col] = df_cleaned[col].replace(0, median_val)
    return df_cleaned


def prepare_splits(
    df: pd.DataFrame,
    target_col: str = "Outcome",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Memisahkan fitur dan target, kemudian melakukan partisi train-test terstratifikasi.
    """
    if target_col not in df.columns:
        raise KeyError(f"Kolom target '{target_col}' tidak ditemukan di DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, feature_names
