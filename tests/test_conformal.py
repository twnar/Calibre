import numpy as np
import pandas as pd
import pytest

from calibre import conformal as cf
from calibre.datasets import load_example
from calibre.pipeline import DataError, detect_task, run_pipeline


def test_quantile_uses_finite_sample_correction():
    scores = np.arange(1, 11, dtype=float)  # n = 10
    # level = ceil(11 * 0.9) / 10 = 1.0 -> the maximum score
    assert cf.conformal_quantile(scores, 0.1) == 10.0


def test_quantile_validates_input():
    with pytest.raises(ValueError):
        cf.conformal_quantile([], 0.1)
    with pytest.raises(ValueError):
        cf.conformal_quantile([1.0], 1.5)


def test_regression_coverage_guarantee_on_synthetic_data():
    """Averaged over many trials, coverage should be >= 1 - alpha."""
    rng = np.random.default_rng(0)
    alpha, covs = 0.1, []
    for _ in range(300):
        cal = rng.normal(size=200)
        test = rng.normal(size=500)
        q = cf.conformal_quantile(np.abs(cal), alpha)
        covs.append(cf.interval_coverage(test, -q * np.ones(500), q * np.ones(500)))
    assert np.mean(covs) >= 1 - alpha - 0.005


def test_classification_sets_never_empty():
    proba = np.array([[0.4, 0.35, 0.25], [0.9, 0.05, 0.05]])
    sets = cf.classification_sets(proba, qhat=0.0)
    assert sets.sum(axis=1).min() >= 1


def test_detect_task():
    assert detect_task(pd.Series(["a", "b", "a"])) == "classification"
    assert detect_task(pd.Series([0, 1, 1, 0])) == "classification"
    assert detect_task(pd.Series(np.linspace(0, 100, 200))) == "regression"


@pytest.mark.parametrize("name", ["breast_cancer", "wine", "diabetes"])
def test_pipeline_hits_requested_coverage(name):
    df, target = load_example(name)
    result = run_pipeline(df, target, alpha=0.1)
    # Test sets are small, so allow some sampling noise around the 90% target.
    assert result["metrics"]["coverage"] >= 0.80
    assert len(result["sweep"]) == 10


def test_naive_intervals_undercover_on_regression():
    df, target = load_example("diabetes")
    m = run_pipeline(df, target, alpha=0.1)["metrics"]
    assert m["naive_coverage"] < m["coverage"]


def test_pipeline_rejects_bad_input():
    df = pd.DataFrame({"a": range(10), "y": range(10)})
    with pytest.raises(DataError):
        run_pipeline(df, "missing")
    with pytest.raises(DataError):
        run_pipeline(df, "y")  # too few rows
