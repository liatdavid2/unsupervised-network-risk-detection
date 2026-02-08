"""
Predict Script (Production-Oriented CLI)
----------------------------------------
Supports:
- Single-flow prediction (default)
- Batch prediction (--batch)
- Strict input validation (fail-safe)
- Decision policy separation
- Rule-based risk framing
- Optional SHAP explanation (single prediction only)

Usage:
  Single prediction:
    python src/inference/predict.py --input flow.json

  Single prediction with SHAP:
    python src/inference/predict.py --input flow.json --explain shap

  Batch prediction:
    python src/inference/predict.py --input flows.parquet --batch
"""

from pathlib import Path
import argparse
import json
import sys

import joblib
import pandas as pd
import shap


# =========================
# Paths & Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "final_model"


# =========================
# Risk Rules (Domain-Based)
# =========================

RISK_RULES = [
    {
        "feature": "dur",
        "condition": lambda v: v > 10,
        "reason": "Unusually long connection duration",
    },
    {
        "feature": "sbytes",
        "condition": lambda v: v > 1_000_000,
        "reason": "Large outbound data transfer",
    },
    {
        "feature": "dbytes",
        "condition": lambda v: v > 1_000_000,
        "reason": "Large inbound data transfer",
    },
    {
        "feature": "ct_dst_sport_ltm",
        "condition": lambda v: v > 50,
        "reason": "High fan-out to destination ports",
    },
    {
        "feature": "ct_srv_dst",
        "condition": lambda v: v > 100,
        "reason": "High number of connections to the same destination",
    },
]


# =========================
# Helpers
# =========================

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def get_latest_run_dir(base_dir: Path) -> Path:
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No model runs found in {base_dir}")
    return max(run_dirs, key=lambda d: d.name)


def assign_risk_level(score: float, threshold: float) -> str:
    if score >= threshold + 0.2:
        return "HIGH"
    if score >= threshold:
        return "MEDIUM"
    return "LOW"


def explain_row(row: pd.Series) -> list[str]:
    reasons = []
    for rule in RISK_RULES:
        value = row.get(rule["feature"])
        if value is None:
            continue
        try:
            if rule["condition"](value):
                reasons.append(rule["reason"])
        except Exception:
            continue
    return reasons


API_REQUIRED_FEATURES = [
    "dur",
    "sbytes",
    "dbytes",
    "ct_dst_sport_ltm",
    "ct_srv_dst",
]

def validate_single_input(flow: dict, required_features: list[str]):
    missing = [f for f in API_REQUIRED_FEATURES if f not in flow]
    if missing:
        raise ValueError(
            f"Missing required API features: {missing}. Prediction rejected."
        )



def predict_single(
    flow: dict,
    model,
    feature_names: list[str],
    threshold: float,
):
    validate_single_input(flow, feature_names)

    row = pd.Series(flow)
    X = pd.DataFrame(
        [[row.get(f, 0) for f in feature_names]],
        columns=feature_names,
    )

    score = float(model.predict_proba(X)[0, 1])
    risk_level = assign_risk_level(score, threshold)
    reasons = explain_row(row)

    return {
        "risk_score": round(score, 4),
        "risk_level": risk_level,
        "reasons": reasons,
    }


def predict_batch(
    df: pd.DataFrame,
    model,
    feature_names: list[str],
    threshold: float,
):
    X = df[feature_names].copy()
    X = X.fillna(0)

    scores = model.predict_proba(X)[:, 1]
    results = []

    for idx, row in df.iterrows():
        score = float(scores[idx])
        results.append(
            {
                "row_index": int(idx),
                "risk_score": round(score, 4),
                "risk_level": assign_risk_level(score, threshold),
                "reasons": explain_row(row),
            }
        )

    return results


def explain_with_shap(flow: dict, model, feature_names: list[str], top_k: int = 5):
    

    row = pd.Series(flow)
    X = pd.DataFrame(
        [[row.get(f, 0) for f in feature_names]],
        columns=feature_names,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]

    shap_dict = dict(zip(feature_names, shap_values))

    # sort by absolute contribution and keep top-k
    top_shap = dict(
        sorted(
            shap_dict.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:top_k]
    )

    return top_shap



# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Production-style inference CLI")
    parser.add_argument("--input", required=True, help="Path to input file (json or parquet)")
    parser.add_argument("--batch", action="store_true", help="Run batch prediction (parquet only)")
    parser.add_argument(
        "--explain",
        choices=["shap"],
        help="Optional explanation method (single prediction only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    run_dir = get_latest_run_dir(FINAL_MODEL_ROOT)

    model = joblib.load(run_dir / "final_model.joblib")
    feature_names = load_json(run_dir / "feature_names.json")

    decision_policy = load_json(run_dir / "decision_policy.json")
    threshold = decision_policy["selected_threshold"]

    if args.batch:
        if input_path.suffix != ".parquet":
            print("ERROR: Batch mode requires a parquet file")
            sys.exit(1)

        df = pd.read_parquet(input_path)
        results = predict_batch(df, model, feature_names, threshold)
        print(json.dumps(results, indent=2))
        return

    # Single prediction
    if input_path.suffix != ".json":
        print("ERROR: Single prediction requires a JSON file")
        sys.exit(1)

    with open(input_path, "r") as f:
        flow = json.load(f)

    result = predict_single(flow, model, feature_names, threshold)

    if args.explain == "shap":
        result["shap_top_5_values"] = explain_with_shap(flow, model, feature_names)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
