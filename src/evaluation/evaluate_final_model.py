import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# ==============================
# Paths
# ==============================

ARTIFACTS_ROOT = Path("artifacts/final_model")
DATA_DIR = Path("data/processed")


# ==============================
# Utilities
# ==============================

def find_latest_run(root: Path) -> Path:
    runs = []
    for p in root.iterdir():
        if p.is_dir() and (p / "final_model.joblib").exists():
            runs.append(p)

    if not runs:
        raise RuntimeError(
            f"No valid model runs found under {root}. "
            "Expected directories containing final_model.joblib"
        )

    return max(runs, key=lambda p: p.stat().st_mtime)


# ==============================
# Loading
# ==============================

def load_artifacts():
    run_dir = find_latest_run(ARTIFACTS_ROOT)
    print(f"Using artifacts from: {run_dir}")

    model = joblib.load(run_dir / "final_model.joblib")

    with open(run_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(run_dir / "run_config.json") as f:
        run_config = json.load(f)

    decision_policy_path = run_dir / "decision_policy.json"
    if not decision_policy_path.exists():
        raise RuntimeError(
            "decision_policy.json not found. "
            "Decision must be frozen before Day 5 evaluation."
        )

    with open(decision_policy_path) as f:
        decision_policy = json.load(f)

    threshold = decision_policy.get("selected_threshold")
    if threshold is None:
        raise RuntimeError(
            "selected_threshold missing from decision_policy.json"
        )

    return model, feature_names, run_config, threshold, run_dir


def load_test_data(feature_names, run_config):
    """
    Load test data.
    Priority:
    1. data/processed/test.parquet if exists
    2. Reconstruct test split deterministically from full dataset
    """

    test_path = DATA_DIR / "test.parquet"

    if test_path.exists():
        df_test = pd.read_parquet(test_path)
        print("Loaded test.parquet")
    else:
        dataset_path = Path(run_config["dataset"])
        if not dataset_path.exists():
            raise RuntimeError(
                f"Dataset not found: {dataset_path}"
            )

        print("test.parquet not found – reconstructing test split from full dataset")

        df = pd.read_parquet(dataset_path)

        X = df[feature_names]
        y = df["binary_label"].astype(int)

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=run_config.get("test_size", 0.2),
            random_state=run_config.get("random_state", 42),
            stratify=y,
        )

        return X_test, y_test

    X_test = df_test[feature_names]
    y_test = df_test["binary_label"].astype(int)

    return X_test, y_test


# ==============================
# Evaluation
# ==============================

def evaluate(model, X, y, threshold):
    y_scores = model.predict_proba(X)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    return metrics, y_scores


def plot_roc(y, y_scores, out_dir: Path):
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png")
    plt.close()


def plot_pr(y, y_scores, out_dir: Path):
    precision, recall, _ = precision_recall_curve(y, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png")
    plt.close()


# ==============================
# Main
# ==============================

def main():
    model, feature_names, run_config, threshold, run_dir = load_artifacts()
    X_test, y_test = load_test_data(feature_names, run_config)

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)

    metrics, y_scores = evaluate(model, X_test, y_test, threshold)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(y_test, y_scores, output_dir)
    plot_pr(y_test, y_scores, output_dir)

    print("=== Final Test Evaluation ===")
    print(f"Run directory     : {run_dir}")
    print(f"Threshold         : {metrics['threshold']}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"Recall            : {metrics['recall']:.4f}")
    print(f"F1 Score          : {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()
