import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.ensemble import HistGradientBoostingClassifier
import joblib


# =========================
# Paths & config
# =========================
DATA_PATH = Path("data/processed/UNSW_Flow_features.parquet")
ARTIFACTS_DIR = Path("artifacts/final_model")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "binary_label"
RANDOM_STATE = 42

SUBSAMPLE_FRACTION = 0.2
TEST_SIZE = 0.2


# =========================
# Logging helper
# =========================
def log(step: str, message: str):
    print(f"[{step}] {message}")


# =========================
# Threshold evaluation
# =========================
def evaluate_thresholds(y_true, y_prob, thresholds):
    rows = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fp_rate = fp / (fp + tn)

        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp_rate": fp_rate,
                "fp_count": int(fp),
                "tp_count": int(tp),
            }
        )

    return pd.DataFrame(rows)


# =========================
# Main pipeline
# =========================
def train():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log("0/4", f"Run directory .......... {run_dir}")

    # [1/4] Load data
    df = pd.read_parquet(DATA_PATH)
    log("1/4", f"Load features .......... OK ({len(df):,} rows)")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Subsample (same as Day 3)
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=SUBSAMPLE_FRACTION,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y_sub,
        test_size=TEST_SIZE,
        stratify=y_sub,
        random_state=RANDOM_STATE,
    )

    log("2/4", "Train/Test split ...... OK (80/20, stratified)")

    # =========================
    # Save feature metadata
    # =========================
    feature_names = list(X_train.columns)
    with open(run_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    run_config = {
        "model": "HistGradientBoosting",
        "target_col": TARGET_COL,
        "n_features": len(feature_names),
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "dataset": str(DATA_PATH),
    }

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # =========================
    # Final model (best baseline)
    # =========================
    model = HistGradientBoostingClassifier(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        random_state=RANDOM_STATE,
    )

    log("3/4", "Training final model ... HistGradientBoosting")
    model.fit(X_train, y_train)

    # =========================
    # Score-based evaluation
    # =========================
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    log("4/4", f"ROC-AUC (scores only) . {roc_auc:.4f}")

    # =========================
    # Threshold tuning
    # =========================
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_df = evaluate_thresholds(y_test.values, y_prob, thresholds)

    selected = threshold_df.sort_values(
        by=["f1"], ascending=False
    ).iloc[0]

    selected_threshold = float(selected["threshold"])

    log(
        "DONE",
        f"Selected threshold ...... {selected_threshold:.3f} "
        f"(Precision={selected['precision']:.3f}, "
        f"Recall={selected['recall']:.3f})"
    )

    # =========================
    # Save artifacts
    # =========================
    joblib.dump(model, run_dir / "final_model.joblib")
    threshold_df.to_csv(run_dir / "threshold_analysis.csv", index=False)

    with open(run_dir / "decision_policy.json", "w") as f:
        json.dump(
            {
                "model": "HistGradientBoosting",
                "roc_auc": roc_auc,
                "selected_threshold": selected_threshold,
                "selection_criteria": "Max F1 (score-based tuning)",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    train()
