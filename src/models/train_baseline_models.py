import pandas as pd
from pathlib import Path
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from xgboost import XGBClassifier
import joblib


# =========================
# Paths & config
# =========================
DATA_PATH = Path("data/processed/UNSW_Flow_features.parquet")
MODEL_DIR = Path("artifacts/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "binary_label"
RANDOM_STATE = 42

SUBSAMPLE_FRACTION = 0.2  # 20% of data for faster baseline training
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = MODEL_DIR / RUN_ID


# =========================
# Logging helper
# =========================
def log(step: str, message: str):
    print(f"[{step}] {message}")


# =========================
# Evaluation helper
# =========================
def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)

    print(
    f"{name:<20} "
    f"ROC-AUC={metrics['ROC_AUC']:.4f} | "
    f"Precision={metrics['Precision']:.4f} | "
    f"Recall={metrics['Recall']:.4f} | "
    f"F1={metrics['F1']:.4f}")

    print("Confusion Matrix:")
    print(cm)

    return metrics


# =========================
# Main training pipeline
# =========================
def train():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log("0/5", f"Run directory .......... {RUN_DIR}")

    # [1/5] Load data
    df = pd.read_parquet(DATA_PATH)
    log("1/5", f"Load features .......... OK ({len(df):,} rows)")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # [2/5] Stratified subsample (for faster training)
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=SUBSAMPLE_FRACTION,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    log(
        "2/5",
        f"Stratified subsample ... OK ({len(X_sub):,} rows, "
        f"{SUBSAMPLE_FRACTION:.0%} of data)"
    )

    # [3/5] Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y_sub,
        test_size=0.2,
        stratify=y_sub,
        random_state=RANDOM_STATE,
    )
    # after train/test split
    feature_names = list(X_train.columns)

    with open(RUN_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    run_config = {
        "subsample_fraction": SUBSAMPLE_FRACTION,
        "random_state": RANDOM_STATE,
        "target_col": TARGET_COL,
        "n_features": len(feature_names),
    }

    with open(RUN_DIR / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    log("3/5", "Train/Test split ...... OK (80/20, stratified)")

    # =========================
    # Models (tree-based, tabular)
    # =========================
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    # [4/5] Train & evaluate
    results = {}

    for name, model in models.items():
        log("4/5", f"Training model ........ {name}")
        model.fit(X_train, y_train)

        metrics = evaluate_model(name, model, X_test, y_test)
        results[name] = metrics

        joblib.dump(model, RUN_DIR / f"{name}.joblib")

    # [5/5] Summary
    log("5/5", "Baseline comparison completed")

    print("\nSummary (sorted by ROC-AUC):")
    summary = (
        pd.DataFrame(results)
        .T.sort_values("ROC_AUC", ascending=False)
    )
    print(summary.round(4))
    summary.to_csv(RUN_DIR / "baseline_metrics.csv")


if __name__ == "__main__":
    train()
