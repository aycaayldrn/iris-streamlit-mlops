"""
MLflow-enabled training script for Delivery 5 (Iris model zoo).
Trains RandomForest, LogisticRegression, SVM, KNN; logs runs, artifacts and
registers the best model (by F1-macro) as 'IrisModel'.
"""

import os
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from joblib import dump, load

# ------------------------
# Configuration
# ------------------------
EXPERIMENT_NAME = "iris-model-zoo"
REGISTERED_MODEL_NAME = "IrisModel"
APP_DIR = Path("app")
APP_DIR.mkdir(exist_ok=True)
LOCAL_MODEL_PATH = APP_DIR / "model.joblib"
MODEL_META_PATH = APP_DIR / "model_meta.json"
RANDOM_STATE = 42
TEST_SIZE = 0.3
VERSION_TAG = "v1.0.0"


# ------------------------
# Helpers
# ------------------------
def compute_metrics(y_true, y_pred, y_proba=None):
    """Return dict of required metrics. roc_auc may be None if unavailable."""
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro"))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro"))

    # Multiclass ROC-AUC: requires probability or decision scores and one-hot labels
    roc_auc = None
    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                from sklearn.preprocessing import label_binarize

                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)
                roc_auc = roc_auc_score(
                    y_true_bin,
                    y_proba,
                    average="macro",
                    multi_class="ovr",
                )
                roc_auc = float(roc_auc)
        except Exception:
            roc_auc = None

    metrics["roc_auc_macro"] = roc_auc
    return metrics


def save_confusion_matrix_image(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------
# Main
# ------------------------
def main():
    # 1. Data
    iris = load_iris()
    X, y = iris.data, iris.target
    labels = iris.target_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2. Model zoo
    models = {
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        # probability=True so we can compute ROC-AUC
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
    }

    # 3. Setup MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_info = {
        "name": None,
        "metrics": None,
        "run_id": None,
    }

    # 3a. Train and log each model
    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id

            # Log model info
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("version", VERSION_TAG)
            mlflow.log_param("model_type", name)

            # Log hyperparameters (best effort)
            try:
                params = model.get_params()
                for k, v in params.items():
                    if isinstance(v, (list, tuple, dict)):
                        mlflow.log_param(k, json.dumps(v))
                    else:
                        mlflow.log_param(k, v)
            except Exception:
                pass

            # Train
            model.fit(X_train, y_train)

            # Predictions and probabilities if available
            y_pred = model.predict(X_test)
            y_proba = None
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    y_proba = None

            # Compute metrics
            metrics = compute_metrics(y_test, y_pred, y_proba=y_proba)

            # Log metrics
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_macro", metrics["f1_macro"])
            mlflow.log_metric("precision_macro", metrics["precision_macro"])
            mlflow.log_metric("recall_macro", metrics["recall_macro"])
            if metrics["roc_auc_macro"] is not None:
                mlflow.log_metric("roc_auc_macro", metrics["roc_auc_macro"])

            # Classification report as artifact
            clf_report = classification_report(
                y_test,
                y_pred,
                target_names=labels,
            )
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                suffix=".txt",
            ) as f:
                f.write(clf_report)
                clf_report_path = f.name
            mlflow.log_artifact(clf_report_path, artifact_path="classification_report")
            os.remove(clf_report_path)

            # Confusion matrix image as artifact
            cm = confusion_matrix(y_test, y_pred)
            with tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
            ) as f:
                cm_path = f.name
            save_confusion_matrix_image(cm, labels, cm_path)
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")
            os.remove(cm_path)

            # Save raw model file as artifact (for audit)
            with tempfile.NamedTemporaryFile(
                suffix=".joblib",
                delete=False,
            ) as f:
                local_model_tmp = f.name
            dump(model, local_model_tmp)
            mlflow.log_artifact(local_model_tmp, artifact_path="models")
            os.remove(local_model_tmp)

            print(
                f"Finished run for {name} (run_id={run_id}). "
                f"f1_macro={metrics['f1_macro']:.4f}"
            )

            # Update best model by F1-macro
            if best_info["name"] is None or metrics["f1_macro"] > best_info["metrics"]["f1_macro"]:
                best_info["name"] = name
                best_info["metrics"] = metrics
                best_info["run_id"] = run_id

    # 4. Persist best model locally and metadata
    if best_info["name"] is None:
        raise RuntimeError("No models were trained / no best model selected.")

    print(
        f"Selected best model: {best_info['name']} "
        f"(f1_macro={best_info['metrics']['f1_macro']:.4f})"
    )

    # Retrain best model on full training data and save locally
    best_model_obj = models[best_info["name"]]
    best_model_obj.fit(X_train, y_train)
    dump(best_model_obj, str(LOCAL_MODEL_PATH))
    print(f"Saved best model to {LOCAL_MODEL_PATH}")

    # Write model_meta.json for the Streamlit app
    model_meta = {
        "best_model": best_info["name"],
        "metrics": {
            "accuracy": round(best_info["metrics"]["accuracy"], 6),
            "f1_macro": round(best_info["metrics"]["f1_macro"], 6),
        },
        "mlflow_run_id": best_info["run_id"],
        "version": VERSION_TAG,
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(model_meta, f, indent=2)
    print(f"Wrote model metadata to {MODEL_META_PATH}")

    # 5. Register the best model in the MLflow Model Registry
    try:
        # Reuse the run that produced the best metrics
        with mlflow.start_run(run_id=best_info["run_id"]):
            # Load the locally saved best model
            best_model_for_registry = load(LOCAL_MODEL_PATH)

            mlflow.sklearn.log_model(
                sk_model=best_model_for_registry,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
            print(f"Requested registration of {REGISTERED_MODEL_NAME} (new version).")
    except Exception as e:
        print(f"Could not register model in MLflow Model Registry: {e}")
        print(
            "This may happen if the tracking server is a simple file store "
            "or registry is not enabled."
        )

    print("Done.")


if __name__ == "__main__":
    main()

