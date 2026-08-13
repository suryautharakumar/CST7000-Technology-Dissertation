import os
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

from utils.ml_pipeline import (
    detect_problem_type,
    build_dataset_profile,
    train_models,
    prepare_prediction_input,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
ALLOWED_EXTENSIONS = {"csv", "xlsx"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-secret-key-for-production"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_dataset(path: str) -> pd.DataFrame:
    ext = path.rsplit(".", 1)[1].lower()
    if ext == "csv":
        return pd.read_csv(path)
    if ext == "xlsx":
        return pd.read_excel(path)
    raise ValueError("Unsupported file type")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or file.filename == "":
            flash("Please choose a CSV or Excel file.", "danger")
            return redirect(url_for("upload"))

        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload .csv or .xlsx only.", "danger")
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"{timestamp}_{filename}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], saved_name)
        file.save(path)

        try:
            df = read_dataset(path)
            if df.empty or df.shape[1] < 2:
                raise ValueError("Dataset must contain at least two columns and one row.")
        except Exception as exc:
            flash(f"Could not read dataset: {exc}", "danger")
            return redirect(url_for("upload"))

        session.clear()
        session["dataset_path"] = path
        session["filename"] = filename
        flash("Dataset uploaded successfully.", "success")
        return redirect(url_for("preview"))

    return render_template("upload.html")


@app.route("/preview")
def preview():
    dataset_path = session.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        flash("Please upload a dataset first.", "warning")
        return redirect(url_for("upload"))

    df = read_dataset(dataset_path)
    profile = build_dataset_profile(df)
    table_html = df.head(10).to_html(classes="table table-striped table-hover table-sm", index=False, border=0)
    return render_template("preview.html", profile=profile, table_html=table_html, columns=list(df.columns))


@app.route("/target", methods=["GET", "POST"])
def target():
    dataset_path = session.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        flash("Please upload a dataset first.", "warning")
        return redirect(url_for("upload"))

    df = read_dataset(dataset_path)
    columns = list(df.columns)

    if request.method == "POST":
        target_col = request.form.get("target_column")
        if target_col not in columns:
            flash("Please select a valid target column.", "danger")
            return redirect(url_for("target"))

        problem_type = detect_problem_type(df[target_col])
        session["target_column"] = target_col
        session["problem_type"] = problem_type
        flash(f"Target selected. Detected task: {problem_type.title()}.", "success")
        return redirect(url_for("preprocessing"))

    return render_template("target.html", columns=columns)


@app.route("/preprocessing")
def preprocessing():
    dataset_path = session.get("dataset_path")
    target_col = session.get("target_column")
    if not dataset_path or not target_col:
        flash("Please upload a dataset and select a target column first.", "warning")
        return redirect(url_for("upload"))

    df = read_dataset(dataset_path)
    missing_before = int(df.isna().sum().sum())
    feature_df = df.drop(columns=[target_col])
    numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    summary = {
        "target_column": target_col,
        "problem_type": session.get("problem_type", "unknown").title(),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "feature_columns": feature_df.shape[1],
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "missing_before": missing_before,
        "steps": [
            "Missing numerical values will be filled using median imputation.",
            "Missing categorical values will be filled using the most frequent value.",
            "Categorical variables will be converted using one-hot encoding.",
            "Numerical features will be scaled using standardisation.",
            "The dataset will be split into training and testing sets.",
        ],
    }
    return render_template("preprocessing.html", summary=summary)


@app.route("/train")
def train():
    dataset_path = session.get("dataset_path")
    target_col = session.get("target_column")
    problem_type = session.get("problem_type")
    if not dataset_path or not target_col or not problem_type:
        flash("Please upload a dataset and select a target column first.", "warning")
        return redirect(url_for("upload"))

    try:
        df = read_dataset(dataset_path)
        results = train_models(df, target_col, problem_type)
    except Exception as exc:
        flash(f"Model training failed: {exc}", "danger")
        return redirect(url_for("preprocessing"))

    model_path = os.path.join(app.config["MODEL_FOLDER"], "best_model.joblib")
    joblib.dump(results["model_bundle"], model_path)
    session["model_path"] = model_path
    session["best_model_name"] = results["best_model_name"]
    session["problem_type"] = problem_type

    return render_template(
        "results.html",
        metrics=results["metrics"],
        best_model_name=results["best_model_name"],
        problem_type=problem_type,
        chart_labels=results["chart_labels"],
        chart_values=results["chart_values"],
        chart_metric=results["chart_metric"],
    )


@app.route("/feature-importance")
def feature_importance():
    model_path = session.get("model_path")
    if not model_path or not os.path.exists(model_path):
        flash("Please train models first.", "warning")
        return redirect(url_for("upload"))

    bundle = joblib.load(model_path)
    importance = bundle.get("feature_importance", [])
    return render_template("feature_importance.html", importance=importance, best_model=session.get("best_model_name"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_path = session.get("model_path")
    dataset_path = session.get("dataset_path")
    target_col = session.get("target_column")
    if not model_path or not os.path.exists(model_path):
        flash("Please train models before making predictions.", "warning")
        return redirect(url_for("upload"))

    bundle = joblib.load(model_path)
    df = read_dataset(dataset_path)
    features = [c for c in df.columns if c != target_col]
    sample_values = df[features].head(1).to_dict(orient="records")[0] if len(df) else {}

    prediction = None
    probability = None

    if request.method == "POST":
        try:
            input_df = prepare_prediction_input(request.form, features, df)
            pred = bundle["pipeline"].predict(input_df)[0]
            prediction = str(pred)

            if bundle["problem_type"] == "classification" and hasattr(bundle["pipeline"], "predict_proba"):
                probs = bundle["pipeline"].predict_proba(input_df)[0]
                probability = round(float(max(probs)) * 100, 2)
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "danger")

    return render_template(
        "predict.html",
        features=features,
        sample_values=sample_values,
        prediction=prediction,
        probability=probability,
        problem_type=bundle["problem_type"],
        best_model=session.get("best_model_name"),
    )


@app.route("/reset")
def reset():
    session.clear()
    flash("Session reset. Please upload a new dataset.", "info")
    return redirect(url_for("upload"))


if __name__ == "__main__":
    app.run(debug=True)
