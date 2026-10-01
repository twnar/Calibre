"""Small built-in example datasets so the app works without any upload."""

from __future__ import annotations

import pandas as pd
from sklearn import datasets

EXAMPLES = {
    "breast_cancer": {
        "label": "Breast cancer diagnosis",
        "note": "Classification, 569 rows. Malignant or benign from cell measurements.",
        "loader": datasets.load_breast_cancer,
    },
    "wine": {
        "label": "Wine cultivars",
        "note": "Classification, 178 rows. Which of 3 cultivars a wine came from.",
        "loader": datasets.load_wine,
    },
    "diabetes": {
        "label": "Diabetes progression",
        "note": "Regression, 442 rows. Disease progression one year after baseline.",
        "loader": datasets.load_diabetes,
    },
}


def load_example(name: str) -> tuple[pd.DataFrame, str]:
    if name not in EXAMPLES:
        raise KeyError(f"Unknown example dataset: {name}")
    bunch = EXAMPLES[name]["loader"](as_frame=True)
    df = bunch.frame.copy()
    target = bunch.target.name if getattr(bunch.target, "name", None) else "target"
    if target not in df.columns:
        df["target"] = bunch.target
        target = "target"
    if name in ("breast_cancer", "wine") and hasattr(bunch, "target_names"):
        names = list(bunch.target_names)
        df[target] = df[target].map(lambda i: names[int(i)])
    return df, target
