"""
Unsupervised Network Risk Detection - Training Script

Assumptions:
- All feature engineering and preprocessing are performed upstream
  in src/build_features.py
- Input file contains:
    - Numeric, model-ready features
    - Optional classification columns (for evaluation only):
        * attack_label
        * binary_label

Design principles:
- Training is strictly UNSUPERVISED
- Classification labels are NEVER used for training
- Labels are removed from X and used only for post-hoc sanity checks

This script:
- Removes classification columns from X (hard rule)
- Trains an unsupervised anomaly detection model (Isolation Forest)
- Computes anomaly scores and risk tiers
- Saves model artifacts and training reports
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

CLASSIFICATION_COLS = [
    "attack_label",
    "binary_label",
]

SUPPORTED_MODELS = ("isolation_forest",)


# ============================================================
# Utilities
# ============================================================

def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError("Unsupported input format (expected .parquet or .csv)")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Model and scoring
# ============================================================

def build_model(random_state: int, contamination: float) -> IsolationForest:
    return IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )


def anomaly_scores(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    decision_function: higher = more normal
    We invert it so that higher = more anomalous
    """
    normality = pipe.decision_function(X)
    return -normality.astype(np.float64)


def compute_risk_tiers(scores: np.ndarray, low_q: float, high_q: float):
    if not (0.0 < low_q < high_q < 1.0):
        raise ValueError("Quantiles must satisfy 0 < low_q < high_q < 1")

    q_low = float(np.quantile(scores, low_q))
    q_high = float(np.quantile(scores, high_q))

    tiers = np.where(
        scores >= q_high,
        "HIGH",
        np.where(scores >= q_low, "MEDIUM", "LOW"),
    )

    return tiers, {
        "low_q": low_q,
        "high_q": high_q,
        "q_low": q_low,
        "q_high": q_high,
    }


def auc_sanity_check(df: pd.DataFrame, scores: np.ndarray) -> Optional[float]:
    """
    Post-hoc sanity check only.
    binary_label is NEVER used for training.
    """
    if "binary_label" not in df.columns:
        return None

    y = pd.to_numeric(df["binary_label"], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return None

    return float(roc_auc_score(y, scores))


# ============================================================
# Report
# ============================================================

@dataclass
class TrainReport:
    run_dir: str
    input_path: str
    n_rows: int
    n_features: int
    contamination: float
    score_mean: float
    score_std: float
    score_min: float
    score_max: float
    risk_thresholds: Dict[str, float]
    risk_counts: Dict[str, int]
    auc_if_binary_label_present: Optional[float]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an unsupervised anomaly detection model")
    p.add_argument("--input", required=True, help="Processed feature file (.parquet or .csv)")
    p.add_argument("--outdir", default="artifacts/unsupervised_model")
    p.add_argument("--run_tag", default="")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--contamination", type=float, default=0.01)
    p.add_argument("--low_q", type=float, default=0.95)
    p.add_argument("--high_q", type=float, default=0.99)
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    run_tag = args.run_tag or now_tag()
    run_dir = Path(args.outdir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    log("0/6", f"Run directory ........ {run_dir}")

    df = load_table(Path(args.input))
    log("1/6", f"Loaded data .......... {df.shape}")

    # --------------------------------------------------------
    # Build X safely (STRICT label exclusion)
    # --------------------------------------------------------

    X = df.drop(columns=CLASSIFICATION_COLS, errors="ignore")

    # Safety check: ensure no label leakage
    for col in CLASSIFICATION_COLS:
        if col in X.columns:
            raise RuntimeError(f"Label leakage detected: {col} found in training features")

    # Ensure numeric-only features
    if X.select_dtypes(include="number").shape[1] != X.shape[1]:
        raise RuntimeError(
            "Non-numeric columns found in input. "
            "build_features.py must output numeric-only features."
        )

    feature_names = list(X.columns)
    log("2/6", f"Training features .... {len(feature_names)}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", build_model(args.random_state, args.contamination)),
        ]
    )

    log("3/6", "Training model ......")
    pipe.fit(X)

    # --------------------------------------------------------
    # Scoring & risk framing
    # --------------------------------------------------------

    scores = anomaly_scores(pipe, X)
    tiers, thresholds = compute_risk_tiers(scores, args.low_q, args.high_q)

    auc = auc_sanity_check(df, scores)

    scored = df.copy()
    scored["anomaly_score"] = scores
    scored["risk_level"] = tiers

    risk_counts = scored["risk_level"].value_counts().to_dict()

    report = TrainReport(
        run_dir=str(run_dir),
        input_path=args.input,
        n_rows=len(df),
        n_features=len(feature_names),
        contamination=args.contamination,
        score_mean=float(scores.mean()),
        score_std=float(scores.std()),
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        risk_thresholds=thresholds,
        risk_counts={k: int(v) for k, v in risk_counts.items()},
        auc_if_binary_label_present=auc,
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    log("4/6", "Saving artifacts ...")

    joblib.dump(pipe, run_dir / "pipeline.joblib")
    save_json(run_dir / "feature_names.json", {"features": feature_names})
    save_json(run_dir / "train_report.json", asdict(report))

    scored.to_parquet(run_dir / "train_scored.parquet", index=False)

    log("5/6", "Done")
    if auc is not None:
        log("NOTE", f"AUC vs binary_label .... {auc:.4f} (sanity check only)")


if __name__ == "__main__":
    main()
