"""
Offline Monitoring (Production-Oriented)
----------------------------------------
Computes data drift and prediction drift between a reference dataset and a current dataset.

Metrics:
- PSI (Population Stability Index) for numeric features (quantile bins based on reference)
- KS statistic (two-sample) for numeric features
- Prediction drift on model risk_score distribution (PSI + KS)

Alert logic:
- If PSI > psi_alert_threshold OR KS > ks_alert_threshold -> drift alert
- If number of drifted features >= min_drifted_features_for_retrain -> retraining candidate

Usage:
  Build a reference profile (recommended once per "release"):
    python src/monitoring/run_monitoring.py --build-reference \
      --reference data/processed/UNSW_Flow_features.parquet

  Compare current dataset to the stored reference profile:
    python src/monitoring/run_monitoring.py --current data/processed/UNSW_Flow_features.parquet

  Or compare using explicit reference parquet (no profile file needed):
    python src/monitoring/run_monitoring.py --reference data/processed/ref.parquet --current data/processed/current.parquet
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


# -------------------------
# Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "final_model"
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "artifacts" / "monitoring" / "reference_profile.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "monitoring" / "latest_report"


# -------------------------
# Utilities
# -------------------------

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_latest_run_dir(base_dir: Path) -> Path:
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No model runs found in {base_dir}")
    return max(run_dirs, key=lambda d: d.name)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# -------------------------
# Drift Metrics
# -------------------------

def _quantile_bin_edges(values: np.ndarray, n_bins: int) -> List[float]:
    """
    Returns quantile bin edges (including -inf, +inf) based on reference distribution.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        # degenerate
        return [-math.inf, math.inf]

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, qs)

    # Ensure strictly increasing edges (handle duplicates)
    edges = np.unique(edges)
    if edges.size < 2:
        return [-math.inf, math.inf]

    # Expand to cover tails
    edges[0] = -math.inf
    edges[-1] = math.inf
    return edges.tolist()


def _hist_prob(values: np.ndarray, edges: List[float]) -> np.ndarray:
    """
    Histogram probabilities for the given edges, with small smoothing to avoid zeros.
    """
    values = values[np.isfinite(values)]
    if len(edges) < 2:
        return np.array([1.0], dtype=float)

    counts, _ = np.histogram(values, bins=np.array(edges, dtype=float))
    counts = counts.astype(float)

    # Smooth to avoid zero probabilities (production-safe)
    eps = 1e-6
    probs = (counts + eps) / (counts.sum() + eps * len(counts))
    return probs


def psi(ref_probs: np.ndarray, cur_probs: np.ndarray) -> float:
    """
    PSI = sum((cur - ref) * ln(cur/ref))
    """
    ref_probs = np.asarray(ref_probs, dtype=float)
    cur_probs = np.asarray(cur_probs, dtype=float)

    if ref_probs.shape != cur_probs.shape:
        raise ValueError("PSI: ref_probs and cur_probs must have the same shape")

    # Both should be > 0 due to smoothing
    val = np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs))
    return float(val)


def ks_statistic(ref_values: np.ndarray, cur_values: np.ndarray) -> float:
    """
    Two-sample KS statistic without SciPy.
    Works for numeric arrays. Returns D in [0, 1].
    """
    ref = ref_values[np.isfinite(ref_values)]
    cur = cur_values[np.isfinite(cur_values)]

    if ref.size == 0 or cur.size == 0:
        return float("nan")

    ref_sorted = np.sort(ref)
    cur_sorted = np.sort(cur)

    all_values = np.sort(np.unique(np.concatenate([ref_sorted, cur_sorted])))
    if all_values.size == 0:
        return float("nan")

    # ECDF using searchsorted
    ref_cdf = np.searchsorted(ref_sorted, all_values, side="right") / ref_sorted.size
    cur_cdf = np.searchsorted(cur_sorted, all_values, side="right") / cur_sorted.size

    d = np.max(np.abs(ref_cdf - cur_cdf))
    return float(d)


# -------------------------
# Model Scoring
# -------------------------

def score_dataset(df: pd.DataFrame, model, feature_names: List[str]) -> np.ndarray:
    X = df[feature_names].copy()
    X = X.fillna(0)
    probs = model.predict_proba(X)[:, 1]
    return probs.astype(float)


# -------------------------
# Reference Profile
# -------------------------

def build_reference_profile(
    df_ref: pd.DataFrame,
    model,
    feature_names: List[str],
    n_bins: int,
) -> Dict:
    """
    Stores:
    - feature bin edges (quantile bins)
    - reference bin probabilities
    - reference summary stats
    - reference prediction distribution profile (risk_score)
    """
    profile = {
        "schema": {
            "n_bins": int(n_bins),
            "feature_names": feature_names,
        },
        "features": {},
        "predictions": {},
    }

    # Feature distributions
    for f in feature_names:
        v = pd.to_numeric(df_ref[f], errors="coerce").to_numpy(dtype=float)
        edges = _quantile_bin_edges(v, n_bins=n_bins)
        ref_probs = _hist_prob(v, edges).tolist()
        profile["features"][f] = {
            "bin_edges": edges,
            "ref_probs": ref_probs,
            "ref_summary": {
                "count": int(np.isfinite(v).sum()),
                "mean": safe_float(np.nanmean(v)),
                "std": safe_float(np.nanstd(v)),
                "p50": safe_float(np.nanpercentile(v, 50)),
                "p95": safe_float(np.nanpercentile(v, 95)),
            },
        }

    # Prediction distribution
    ref_scores = score_dataset(df_ref, model, feature_names)
    pred_edges = _quantile_bin_edges(ref_scores, n_bins=n_bins)
    pred_ref_probs = _hist_prob(ref_scores, pred_edges).tolist()
    profile["predictions"] = {
        "bin_edges": pred_edges,
        "ref_probs": pred_ref_probs,
        "ref_summary": {
            "count": int(np.isfinite(ref_scores).sum()),
            "mean": safe_float(np.nanmean(ref_scores)),
            "std": safe_float(np.nanstd(ref_scores)),
            "p50": safe_float(np.nanpercentile(ref_scores, 50)),
            "p95": safe_float(np.nanpercentile(ref_scores, 95)),
        },
    }

    return profile


# -------------------------
# Drift Report
# -------------------------

def compute_drift_report(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    profile: Dict,
    model,
    psi_alert_threshold: float,
    ks_alert_threshold: float,
    min_drifted_features_for_retrain: int,
) -> Dict:
    feature_names = profile["schema"]["feature_names"]
    n_bins = int(profile["schema"]["n_bins"])

    report = {
        "schema": {
            "n_bins": n_bins,
            "psi_alert_threshold": psi_alert_threshold,
            "ks_alert_threshold": ks_alert_threshold,
            "min_drifted_features_for_retrain": min_drifted_features_for_retrain,
        },
        "data_drift": [],
        "prediction_drift": {},
        "alerts": {
            "drifted_features": [],
            "retraining_candidate": False,
        },
    }

    # Feature drift
    drifted = []
    for f in feature_names:
        ref_info = profile["features"][f]
        edges = ref_info["bin_edges"]
        ref_probs = np.array(ref_info["ref_probs"], dtype=float)

        ref_vals = pd.to_numeric(df_ref[f], errors="coerce").to_numpy(dtype=float)
        cur_vals = pd.to_numeric(df_cur[f], errors="coerce").to_numpy(dtype=float)

        cur_probs = _hist_prob(cur_vals, edges)
        psi_val = psi(ref_probs, cur_probs)
        ks_val = ks_statistic(ref_vals, cur_vals)

        is_alert = (psi_val > psi_alert_threshold) or (np.isfinite(ks_val) and ks_val > ks_alert_threshold)
        if is_alert:
            drifted.append(f)

        report["data_drift"].append(
            {
                "feature": f,
                "psi": float(psi_val),
                "ks": float(ks_val) if np.isfinite(ks_val) else None,
                "alert": bool(is_alert),
            }
        )

    # Prediction drift
    ref_scores = score_dataset(df_ref, model, feature_names)
    cur_scores = score_dataset(df_cur, model, feature_names)

    pred_edges = profile["predictions"]["bin_edges"]
    pred_ref_probs = np.array(profile["predictions"]["ref_probs"], dtype=float)
    pred_cur_probs = _hist_prob(cur_scores, pred_edges)

    pred_psi = psi(pred_ref_probs, pred_cur_probs)
    pred_ks = ks_statistic(ref_scores, cur_scores)
    pred_alert = (pred_psi > psi_alert_threshold) or (np.isfinite(pred_ks) and pred_ks > ks_alert_threshold)

    report["prediction_drift"] = {
        "psi": float(pred_psi),
        "ks": float(pred_ks) if np.isfinite(pred_ks) else None,
        "alert": bool(pred_alert),
        "ref_summary": profile["predictions"]["ref_summary"],
        "cur_summary": {
            "count": int(np.isfinite(cur_scores).sum()),
            "mean": safe_float(np.nanmean(cur_scores)),
            "std": safe_float(np.nanstd(cur_scores)),
            "p50": safe_float(np.nanpercentile(cur_scores, 50)),
            "p95": safe_float(np.nanpercentile(cur_scores, 95)),
        },
    }

    # Alerts / retraining candidate
    report["alerts"]["drifted_features"] = drifted
    if (len(drifted) >= min_drifted_features_for_retrain) or pred_alert:
        report["alerts"]["retraining_candidate"] = True

    return report


def save_report(report: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    save_json(report, out_dir / "monitoring_report.json")

    # Flat CSV summary for quick scanning
    rows = []
    for item in report["data_drift"]:
        rows.append(
            {
                "type": "feature",
                "name": item["feature"],
                "psi": item["psi"],
                "ks": item["ks"],
                "alert": item["alert"],
            }
        )
    pred = report["prediction_drift"]
    rows.append(
        {
            "type": "prediction",
            "name": "risk_score",
            "psi": pred["psi"],
            "ks": pred["ks"],
            "alert": pred["alert"],
        }
    )
    pd.DataFrame(rows).to_csv(out_dir / "monitoring_summary.csv", index=False)


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Offline monitoring for drift detection (PSI/KS).")

    p.add_argument("--build-reference", action="store_true", help="Build and save reference profile from --reference parquet.")
    p.add_argument("--reference", type=str, help="Reference parquet path (baseline). Required for --build-reference, optional for compare.")
    p.add_argument("--current", type=str, help="Current parquet path (new data). Required for compare mode.")

    p.add_argument("--profile", type=str, default=str(DEFAULT_PROFILE_PATH), help="Path to reference profile JSON.")
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for monitoring report.")

    p.add_argument("--n-bins", type=int, default=10, help="Number of quantile bins for PSI calculation.")
    p.add_argument("--psi-threshold", type=float, default=0.2, help="PSI alert threshold (common: 0.1/0.2/0.3).")
    p.add_argument("--ks-threshold", type=float, default=0.1, help="KS alert threshold (common: 0.1).")
    p.add_argument("--min-drifted-features", type=int, default=5, help="If >= this many features drift, flag retraining candidate.")

    return p.parse_args()


def main():
    args = parse_args()

    run_dir = get_latest_run_dir(FINAL_MODEL_ROOT)
    model = joblib.load(run_dir / "final_model.joblib")
    feature_names = load_json(run_dir / "feature_names.json")

    profile_path = Path(args.profile)
    out_dir = Path(args.outdir)

    if args.build_reference:
        if not args.reference:
            print("ERROR: --build-reference requires --reference <parquet>")
            sys.exit(1)

        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"ERROR: Reference parquet not found: {ref_path}")
            sys.exit(1)

        df_ref = pd.read_parquet(ref_path)

        # Ensure required columns exist
        missing = [c for c in feature_names if c not in df_ref.columns]
        if missing:
            print(f"ERROR: Reference parquet missing {len(missing)} required feature columns, e.g.: {missing[:10]}")
            sys.exit(1)

        profile = build_reference_profile(
            df_ref=df_ref,
            model=model,
            feature_names=feature_names,
            n_bins=args.n_bins,
        )
        save_json(profile, profile_path)
        print(f"Saved reference profile: {profile_path}")
        return

    # Compare mode
    if not args.current:
        print("ERROR: compare mode requires --current <parquet> (or use --build-reference)")
        sys.exit(1)

    cur_path = Path(args.current)
    if not cur_path.exists():
        print(f"ERROR: Current parquet not found: {cur_path}")
        sys.exit(1)

    df_cur = pd.read_parquet(cur_path)
    missing_cur = [c for c in feature_names if c not in df_cur.columns]
    if missing_cur:
        print(f"ERROR: Current parquet missing {len(missing_cur)} required feature columns, e.g.: {missing_cur[:10]}")
        sys.exit(1)

    # Reference source:
    # - If explicit --reference parquet is provided: compute profile from it (one-off compare)
    # - Else load profile from JSON
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            print(f"ERROR: Reference parquet not found: {ref_path}")
            sys.exit(1)

        df_ref = pd.read_parquet(ref_path)
        missing_ref = [c for c in feature_names if c not in df_ref.columns]
        if missing_ref:
            print(f"ERROR: Reference parquet missing {len(missing_ref)} required feature columns, e.g.: {missing_ref[:10]}")
            sys.exit(1)

        profile = build_reference_profile(
            df_ref=df_ref,
            model=model,
            feature_names=feature_names,
            n_bins=args.n_bins,
        )
    else:
        if not profile_path.exists():
            print(f"ERROR: Reference profile not found: {profile_path}")
            print("Hint: build it first using --build-reference --reference <parquet>")
            sys.exit(1)
        profile = load_json(profile_path)

        # Optional sanity check: profile features match current run feature list
        prof_features = profile.get("schema", {}).get("feature_names", [])
        if prof_features != feature_names:
            print("WARNING: Reference profile feature_names differ from latest model feature_names.")
            print("This can be OK if you intentionally compare across releases, but verify compatibility.")

        # Need df_ref for KS and ref-score distribution in report
        # If no parquet is provided, we cannot recompute KS vs. ref data.
        # Therefore: require explicit reference parquet for KS OR skip KS.
        # To keep this script strict, require explicit --reference when using KS.
        # However, we already store ref_probs/edges, so PSI is OK.
        # We'll load ref parquet only if provided. Otherwise, we compute KS as None.
        df_ref = None

    # If df_ref is not available (profile-only), degrade KS to None safely.
    if df_ref is None:
        # Minimal reference DF for PSI-only mode: use current df as placeholder for KS arrays (will yield 0),
        # but we want "None" instead. We'll implement PSI-only by reusing current arrays but forcing ks=None.
        # Best practice: pass --reference to enable KS.
        print("WARNING: No --reference parquet provided; KS statistics will be reported as None. PSI remains valid.")
        df_ref = df_cur.copy()

        # Mark in profile that KS is not reliable in this run
        ks_enabled = False
    else:
        ks_enabled = True

    report = compute_drift_report(
        df_ref=df_ref,
        df_cur=df_cur,
        profile=profile,
        model=model,
        psi_alert_threshold=args.psi_threshold,
        ks_alert_threshold=args.ks_threshold if ks_enabled else float("inf"),  # effectively disable KS alerts
        min_drifted_features_for_retrain=args.min_drifted_features,
    )

    # If KS disabled, scrub KS values to None for clarity
    if not ks_enabled:
        for item in report["data_drift"]:
            item["ks"] = None
        report["prediction_drift"]["ks"] = None

    save_report(report, out_dir)

    drifted = report["alerts"]["drifted_features"]
    pred_alert = report["prediction_drift"]["alert"]
    retrain = report["alerts"]["retraining_candidate"]

    print(f"Saved monitoring report to: {out_dir}")
    print(f"Drifted features: {len(drifted)}")
    print(f"Prediction drift alert: {pred_alert}")
    print(f"Retraining candidate: {retrain}")

    # Print top drifted features by PSI
    df_summary = pd.DataFrame(report["data_drift"]).sort_values("psi", ascending=False).head(10)
    print("\nTop 10 features by PSI:")
    print(df_summary[["feature", "psi", "ks", "alert"]].to_string(index=False))


if __name__ == "__main__":
    main()
