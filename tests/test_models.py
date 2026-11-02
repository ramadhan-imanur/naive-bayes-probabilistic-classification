"""
Unit test untuk modul pemodelan (src.models).
"""

import numpy as np
import pandas as pd
import pytest

from src.models import GaussianNBClassifier


@pytest.fixture
def sample_features_and_target():
    np.random.seed(42)
    X = pd.DataFrame({
        "feat1": np.random.normal(loc=0.0, scale=1.0, size=100),
        "feat2": np.random.normal(loc=2.0, scale=1.5, size=100),
    })
    y = pd.Series(np.random.choice([0, 1], size=100, p=[0.6, 0.4]))
    return X, y


def test_model_unfitted_raises_runtime_error(sample_features_and_target):
    X, _ = sample_features_and_target
    clf = GaussianNBClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X)


def test_model_fit_and_priors(sample_features_and_target):
    X, y = sample_features_and_target
    clf = GaussianNBClassifier()
    clf.fit(X, y)

    assert clf.is_fitted
    priors = clf.class_prior_
    assert len(priors) == 2
    assert np.isclose(np.sum(priors), 1.0)


def test_model_parameter_extraction(sample_features_and_target):
    X, y = sample_features_and_target
    clf = GaussianNBClassifier()
    clf.fit(X, y)

    params = clf.get_parameters_dict(list(X.columns))
    assert "feat1" in params
    assert "feat2" in params

    assert "mean_0" in params["feat1"]
    assert "mean_1" in params["feat1"]
    assert "var_0" in params["feat1"]
    assert "var_1" in params["feat1"]
    assert params["feat1"]["var_0"] > 0


def test_predictions_shape_and_probability(sample_features_and_target):
    X, y = sample_features_and_target
    clf = GaussianNBClassifier()
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    # Probabilities row sum must equal 1
    assert np.allclose(np.sum(probs, axis=1), 1.0)
