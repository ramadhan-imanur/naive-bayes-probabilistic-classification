"""
Modul Pemodelan Probabilistik Gaussian Naïve Bayes.

Mengimplementasikan abstraksi model klasifikasi berbasis distribusi normal Gauss
dengan ekstraksi parameter peluang formal (Prior, Mean, dan Variansi per kelas).
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


class GaussianNBClassifier:
    """
    Model Gaussian Naïve Bayes dengan antarmuka parametrik formal.

    Mengasumsikan atribut kontinu berdistribusi normal:
        P(x_j | c_k) = (1 / sqrt(2 * pi * sigma_{kj}^2)) * exp(-(x_j - mu_{kj})^2 / (2 * sigma_{kj}^2))
    """

    def __init__(self, **kwargs: Any) -> None:
        self.model = GaussianNB(**kwargs)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GaussianNBClassifier":
        """
        Melatih model Gaussian Naïve Bayes pada dataset latih.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Melakukan prediksi label kelas menggunakan aturan keputusan MAP (Maximum A Posteriori).
        """
        if not self.is_fitted:
            raise RuntimeError("Model belum dilatih. Panggil fit() terlebih dahulu.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Menghitung estimasi probabilitas posterior P(c_k | x).
        """
        if not self.is_fitted:
            raise RuntimeError("Model belum dilatih. Panggil fit() terlebih dahulu.")
        return self.model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_

    @property
    def class_prior_(self) -> np.ndarray:
        return self.model.class_prior_

    @property
    def theta_(self) -> np.ndarray:
        return self.model.theta_

    @property
    def var_(self) -> np.ndarray:
        return self.model.var_

    def get_parameters_dict(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Mengekstrak parameter Likelihood Gauss (mean & variance) untuk setiap fitur per kelas.
        """
        if not self.is_fitted:
            raise RuntimeError("Model belum dilatih. Panggil fit() terlebih dahulu.")

        parameters: Dict[str, Dict[str, float]] = {}
        for i, col in enumerate(feature_names):
            parameters[col] = {
                "mean_0": float(self.theta_[0][i]),
                "mean_1": float(self.theta_[1][i]),
                "var_0": float(self.var_[0][i]),
                "var_1": float(self.var_[1][i]),
            }
        return parameters
