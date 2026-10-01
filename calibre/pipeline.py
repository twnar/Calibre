"""End-to-end pipeline: CSV in, honest uncertainty report out."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from . import conformal as cf

ALPHA_GRID = [round(a, 2) for a in np.arange(0.05, 0.55, 0.05)]
MAX_ROWS = 20000
MAX_CATEGORIES = 50
MIN_ROWS = 60


class DataError(ValueError):
    """Raised for problems the user can fix by changing their data or settings."""


def detect_task(y: pd.Series) -> str:
    if y.dtype == bool or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    values = y.dropna()
    is_integer_like = np.all(np.isclose(values, np.round(values)))
    if is_integer_like and values.nunique() <= 10:
        return "classification"
    return "regression"


def _prepare_features(X: pd.DataFrame, warnings: list[str]):
    dropped = []
    for col in list(X.columns):
        s = X[col]
        n_unique = s.nunique(dropna=True)
        if n_unique <= 1:
            dropped.append(col)
        elif not pd.api.types.is_numeric_dtype(s) and s.dtype != bool and n_unique > MAX_CATEGORIES:
            dropped.append(col)
    if dropped:
        shown = ", ".join(map(str, dropped[:6])) + ("..." if len(dropped) > 6 else "")
        warnings.append(
            f"Ignored {len(dropped)} column(s) that cannot help a model "
            f"(constant, or text/IDs with more than {MAX_CATEGORIES} distinct values): {shown}."
        )
        X = X.drop(columns=dropped)
    if X.shape[1] == 0:
        raise DataError("No usable feature columns are left after cleaning.")

    X = X.copy()
    numeric, categorical = [], []
    for col in X.columns:
        s = X[col]
        if s.dtype == bool:
            X[col] = s.astype(int)
            numeric.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
        else:
            X[col] = s.astype(object).where(s.notna(), "missing").astype(str)
            categorical.append(col)

    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )
    return X, ColumnTransformer(transformers)


def _split(n: int, strat, seed: int):
    idx = np.arange(n)
    i_train, i_rest = train_test_split(idx, test_size=0.5, random_state=seed, stratify=strat)
    strat_rest = None if strat is None else strat[i_rest]
    i_cal, i_test = train_test_split(i_rest, test_size=0.5, random_state=seed, stratify=strat_rest)
    return i_train, i_cal, i_test


def run_pipeline(df: pd.DataFrame, target: str, alpha: float = 0.1, seed: int = 42) -> dict:
    """Train a model, calibrate it with split conformal prediction, and report.

    Data is split 50% train / 25% calibration / 25% test. The test part is never
    seen by the model or the calibration step, so the reported coverage is an
    honest out-of-sample measurement.
    """
    if target not in df.columns:
        raise DataError(f"Column '{target}' was not found in the data.")
    if not 0.01 <= alpha <= 0.5:
        raise DataError("Confidence must be between 50% and 99%.")

    warnings: list[str] = []
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=seed).reset_index(drop=True)
        warnings.append(f"Used a random sample of {MAX_ROWS:,} rows to keep things fast.")

    y_raw = df[target]
    task = detect_task(y_raw)
    X = df.drop(columns=[target])

    if task == "regression":
        y_num = pd.to_numeric(y_raw, errors="coerce")
        keep = np.isfinite(y_num.to_numpy())
        if not keep.all():
            warnings.append(f"Dropped {int((~keep).sum())} row(s) with a non-finite target.")
        X, y_raw = X[keep].reset_index(drop=True), y_num[keep].reset_index(drop=True)
    else:
        y_raw = y_raw.astype(str)
        counts = y_raw.value_counts()
        rare = counts[counts < 6].index
        if len(rare):
            keep = ~y_raw.isin(rare)
            warnings.append(
                f"Removed {len(rare)} class(es) with fewer than 6 examples: "
                f"{', '.join(map(str, rare[:5]))}."
            )
            X, y_raw = X[keep].reset_index(drop=True), y_raw[keep].reset_index(drop=True)
        if y_raw.nunique() < 2:
            raise DataError("The target needs at least two classes with 6+ examples each.")

    if len(X) < MIN_ROWS:
        raise DataError(f"Need at least {MIN_ROWS} usable rows; found {len(X)}.")

    X, prep = _prepare_features(X, warnings)

    if task == "classification":
        enc = LabelEncoder().fit(y_raw)
        y = enc.transform(y_raw)
        classes = [str(c) for c in enc.classes_]
        model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
        strat = y
    else:
        y = y_raw.to_numpy(dtype=float)
        classes = []
        model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
        strat = None

    i_train, i_cal, i_test = _split(len(X), strat, seed)
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X.iloc[i_train], y[i_train])

    result = {
        "task": task,
        "target": target,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "split": {"train": int(len(i_train)), "calibration": int(len(i_cal)), "test": int(len(i_test))},
        "alpha": alpha,
        "warnings": warnings,
    }

    if task == "classification":
        result.update(_classification_report(pipe, X, y, classes, i_cal, i_test, alpha))
    else:
        result.update(_regression_report(pipe, X, y, i_train, i_cal, i_test, alpha))
    return result


def _full_proba(pipe, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    proba = pipe.predict_proba(X)
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((len(X), n_classes))
    full[:, pipe.named_steps["model"].classes_] = proba
    return full


def _classification_report(pipe, X, y, classes, i_cal, i_test, alpha) -> dict:
    p_cal = _full_proba(pipe, X.iloc[i_cal], len(classes))
    p_test = _full_proba(pipe, X.iloc[i_test], len(classes))
    s_cal = cf.classification_scores(p_cal, y[i_cal])
    y_test = y[i_test]

    def at(a):
        q = cf.conformal_quantile(s_cal, a)
        sets = cf.classification_sets(p_test, q)
        return sets, cf.set_coverage(sets, y_test), float(sets.sum(axis=1).mean())

    sets, coverage, size = at(alpha)
    accuracy = float((p_test.argmax(axis=1) == y_test).mean())
    sizes = sets.sum(axis=1)

    sweep = []
    for a in ALPHA_GRID:
        _, cov, sz = at(a)
        sweep.append({"confidence": round(1 - a, 2), "coverage": cov, "size": sz})

    samples = []
    for k in range(min(10, len(i_test))):
        samples.append({
            "row": int(i_test[k]) + 1,
            "truth": classes[y_test[k]],
            "prediction": [classes[j] for j in np.flatnonzero(sets[k])],
            "top": classes[int(p_test[k].argmax())],
            "top_confidence": float(p_test[k].max()),
            "hit": bool(sets[k, y_test[k]]),
        })

    return {
        "classes": classes,
        "metrics": {
            "coverage": coverage,
            "accuracy": accuracy,
            "avg_set_size": size,
            "single_answer_share": float((sizes == 1).mean()),
            "multi_answer_share": float((sizes > 1).mean()),
        },
        "sweep": sweep,
        "samples": samples,
    }


def _regression_report(pipe, X, y, i_train, i_cal, i_test, alpha) -> dict:
    pred_cal = pipe.predict(X.iloc[i_cal])
    pred_test = pipe.predict(X.iloc[i_test])
    pred_train = pipe.predict(X.iloc[i_train])
    y_test = y[i_test]
    s_cal = cf.regression_scores(y[i_cal], pred_cal)

    # The "naive" baseline many people use: assume errors look like the model's
    # errors on its own training data, and that they are Gaussian.
    train_sd = float(np.std(y[i_train] - pred_train))

    def at(a):
        q = cf.conformal_quantile(s_cal, a)
        lo, hi = cf.regression_intervals(pred_test, q)
        z = norm.ppf(1 - a / 2)
        n_lo, n_hi = pred_test - z * train_sd, pred_test + z * train_sd
        return {
            "coverage": cf.interval_coverage(y_test, lo, hi),
            "width": float(2 * q),
            "naive_coverage": cf.interval_coverage(y_test, n_lo, n_hi),
            "naive_width": float(2 * z * train_sd),
            "q": q,
        }

    main = at(alpha)
    sweep = []
    for a in ALPHA_GRID:
        r = at(a)
        sweep.append({
            "confidence": round(1 - a, 2),
            "coverage": r["coverage"],
            "naive_coverage": r["naive_coverage"],
            "width": r["width"],
        })

    lo, hi = cf.regression_intervals(pred_test, main["q"])
    samples = []
    for k in range(min(10, len(i_test))):
        samples.append({
            "row": int(i_test[k]) + 1,
            "truth": float(y_test[k]),
            "prediction": float(pred_test[k]),
            "low": float(lo[k]),
            "high": float(hi[k]),
            "hit": bool(lo[k] <= y_test[k] <= hi[k]),
        })

    return {
        "metrics": {
            "coverage": main["coverage"],
            "mae": float(np.mean(np.abs(y_test - pred_test))),
            "avg_width": main["width"],
            "target_spread": float(np.std(y)),
            "naive_coverage": main["naive_coverage"],
            "naive_width": main["naive_width"],
        },
        "sweep": sweep,
        "samples": samples,
    }
