"""
Unit test untuk modul pra-pemrosesan data (src.data).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data import (
    DEFAULT_ZERO_AS_MISSING,
    ClinicalImputer,
    clean_clinical_data,
    load_dataset,
    prepare_splits,
    resolve_default_dataset_path,
)


@pytest.fixture
def sample_dataset_path() -> Path:
    return resolve_default_dataset_path()


@pytest.fixture
def dummy_clinical_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Pregnancies": [0, 1, 2, 3],
        "Glucose": [0, 100, 120, 140],  # One invalid 0
        "BloodPressure": [70, 0, 80, 90],  # One invalid 0
        "SkinThickness": [20, 30, 0, 40],  # One invalid 0
        "Insulin": [0, 80, 120, 160],  # One invalid 0
        "BMI": [25.0, 0.0, 30.0, 35.0],  # One invalid 0
        "DiabetesPedigreeFunction": [0.5, 0.4, 0.6, 0.7],
        "Age": [25, 30, 35, 40],
        "Outcome": [0, 0, 1, 1],
    })


def test_resolve_and_load_dataset(sample_dataset_path: Path):
    assert sample_dataset_path.exists()
    df = load_dataset(sample_dataset_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 768
    assert "Outcome" in df.columns


def test_clean_clinical_data_imputes_zeros(dummy_clinical_df: pd.DataFrame):
    df_clean = clean_clinical_data(dummy_clinical_df)

    for col in DEFAULT_ZERO_AS_MISSING:
        # Nilai nol pada fitur klinis seharusnya sudah terganti dengan median positif
        assert (df_clean[col] == 0).sum() == 0

    # Pregnancies diperbolehkan bernilai 0 (belum pernah hamil)
    assert (df_clean["Pregnancies"] == 0).sum() == 1


def test_prepare_splits(dummy_clinical_df: pd.DataFrame):
    X_train, X_test, y_train, y_test, feature_names = prepare_splits(
        dummy_clinical_df,
        target_col="Outcome",
        test_size=0.50,
        random_state=42,
    )

    assert len(X_train) == 2
    assert len(X_test) == 2
    assert len(feature_names) == 8
    assert "Outcome" not in feature_names
    # Stratified balance: each split gets 1 zero and 1 one
    assert (y_train == 1).sum() == 1
    assert (y_test == 1).sum() == 1


def test_clinical_imputer_no_data_leakage():
    """
    Memverifikasi bahwa ClinicalImputer hanya menggunakan data latih dan
    tidak membocorkan statistik data uji.
    """
    train_data = pd.DataFrame({"Glucose": [0, 100, 120]})  # Valid: 100, 120 -> Median = 110
    test_data = pd.DataFrame({"Glucose": [0, 200, 250]})   # Valid: 200, 250

    imputer = ClinicalImputer(zero_as_missing_cols=["Glucose"])
    imputer.fit(train_data)

    assert imputer.medians_["Glucose"] == 110.0

    # Transform test_data harus menggunakan median train (110.0), bukan median test (225.0)
    transformed_test = imputer.transform(test_data)
    assert transformed_test.loc[0, "Glucose"] == 110.0

