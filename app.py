"""Calibre web app. Run with:  python app.py  then open http://127.0.0.1:5000"""

from __future__ import annotations

import io

import pandas as pd
from flask import Flask, jsonify, render_template, request

from calibre.datasets import EXAMPLES, load_example
from calibre.pipeline import DataError, run_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit


def _read_upload(file_storage) -> pd.DataFrame:
    raw = file_storage.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:  # pandas raises many error types here
        raise DataError("That file could not be read as a CSV.") from exc
    if df.shape[1] < 2:
        raise DataError("The file needs at least two columns: features and a target.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _load_source():
    """Return (dataframe, default_target) from either an example name or an upload."""
    example = request.form.get("example")
    if example:
        try:
            return load_example(example)
        except KeyError as exc:
            raise DataError(str(exc)) from exc
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        raise DataError("Choose an example dataset or upload a CSV file.")
    df = _read_upload(upload)
    return df, df.columns[-1]


@app.get("/")
def index():
    examples = [{"id": k, "label": v["label"], "note": v["note"]} for k, v in EXAMPLES.items()]
    return render_template("index.html", examples=examples)


@app.post("/api/inspect")
def inspect():
    try:
        df, default_target = _load_source()
    except DataError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(columns=list(df.columns), default_target=default_target, rows=int(len(df)))


@app.post("/api/run")
def run():
    try:
        df, default_target = _load_source()
        target = request.form.get("target") or default_target
        confidence = float(request.form.get("confidence", 90)) / 100
        result = run_pipeline(df, target, alpha=round(1 - confidence, 4))
    except DataError as exc:
        return jsonify(error=str(exc)), 400
    except ValueError as exc:
        return jsonify(error=f"Invalid setting: {exc}"), 400
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That file is larger than 20 MB."), 413


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
