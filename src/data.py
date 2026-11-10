"""
Modul Pra-pemrosesan Data & Ingesti Klinis Bebas Kebocoran Data (Zero Data Leakage).

Menyediakan:
1. Pemuatan dataset tabular klinis (CSV).
2. Kelas Transformer `ClinicalImputer` kompatibel scikit-learn Pipeline untuk
   imputasi nilai nol biologis tidak valid dengan median sampel latih valid (>0).
3. Fungsi utilitas imputasi data dan partisi train-test terstratifikasi tanpa kebocoran data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

DEFAULT_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class ClinicalImputer(BaseEstimator, TransformerMixin):
    """
    Imputer fisiologis klinis untuk menangani nilai nol yang tidak valid.

    Pada dataset klinis (misalnya Pima Indians Diabetes), nilai 0 pada atribut
    fisiologis tertentu (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
    merepresentasikan data yang tidak tercatat (missing values).

    Untuk mencegah kebocoran data (data leakage), nilai median dihitung secara
    eksklusif dari observasi bernilai positif (> 0) pada data latih (training set),
    kemudian ditransformasikan ke data uji atau lipatan validasi silang.
    """

    def __init__(self, zero_as_missing_cols: Optional[List[str]] = None) -> None:
        self.zero_as_missing_cols = zero_as_missing_cols

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> "ClinicalImputer":
        """
        Menghitung median dari observasi valid (> 0) pada setiap kolom target.
        """
        cols_to_check = (
            self.zero_as_missing_cols
            if self.zero_as_missing_cols is not None
            else DEFAULT_ZERO_AS_MISSING
        )

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_df = X
        else:
            self.feature_names_in_ = [f"col_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        self.medians_ = {}
        for col in cols_to_check:
            if col in X_df.columns:
                valid_mask = X_df[col] > 0
                if valid_mask.any():
                    self.medians_[col] = float(X_df.loc[valid_mask, col].median())
                else:
                    self.medians_[col] = 0.0

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Mengganti nilai 0 pada kolom target dengan median yang dipelajari saat fit().
        """
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        else:
            X_df = X.copy()

        for col, med in self.medians_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].replace(0, med)

        return X_df.values if is_numpy else X_df


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
    imputer: Optional[ClinicalImputer] = None,
) -> pd.DataFrame:
    """
    Membersihkan data klinis dengan menangani nilai nol yang secara biologis tidak mungkin.

    Jika objek `imputer` diberikan, transformasi menggunakan parameter median yang sudah difit.
    Jika tidak, imputer difit pada `df` yang diberikan. Catatan: Untuk pipeline partisi data,
    gunakan `prepare_splits` atau `ClinicalImputer` terpisah agar terhindar dari kebocoran data.
    """
    if imputer is None:
        imputer = ClinicalImputer(zero_as_missing_cols=zero_as_missing_cols)
        imputer.fit(df)

    return imputer.transform(df)


def prepare_splits(
    df: pd.DataFrame,
    target_col: str = "Outcome",
    test_size: float = 0.20,
    random_state: int = 42,
    impute_missing: bool = True,
    zero_as_missing_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Melakukan partisi train-test terstratifikasi pada data mentah, kemudian mengimputasi
    nilai nol biologis secara eksklusif menggunakan statistik dari himpunan data latih.

    Prosedur ini menjamin 100% bebas dari kebocoran data (zero data leakage) karena statistik
    distribusi data uji sama sekali tidak digunakan dalam tahap imputasi data latih.
    """
    if target_col not in df.columns:
        raise KeyError(f"Kolom target '{target_col}' tidak ditemukan di DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if impute_missing:
        imputer = ClinicalImputer(zero_as_missing_cols=zero_as_missing_cols)
        imputer.fit(X_train_raw)
        X_train = imputer.transform(X_train_raw)
        X_test = imputer.transform(X_test_raw)
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    return X_train, X_test, y_train, y_test, feature_names

