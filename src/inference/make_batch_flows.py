from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_MODEL_ROOT = PROJECT_ROOT / "artifacts" / "final_model"
INPUT_FEATURES_PARQUET = PROJECT_ROOT / "data" / "processed" / "UNSW_Flow_features.parquet"

OUT_DIR = PROJECT_ROOT / "src" / "inference" / "examples_api_samples"
OUT_FILE = OUT_DIR / "batch_flows.parquet"

N_ATTACK = 10
N_BENIGN = 10
RANDOM_STATE = 42


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def get_latest_run_dir(base_dir: Path) -> Path:
    run_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No model runs found in {base_dir}")
    return max(run_dirs, key=lambda d: d.name)


def infer_attack_mask(df: pd.DataFrame) -> pd.Series:
    """
    Tries to infer which rows are attacks vs benign from common columns:
    - binary_label: 0/1
    - attack_label: 'normal' vs others
    Falls back to error if nothing is found.
    """
    if "binary_label" in df.columns:
        return df["binary_label"].astype(int) == 1

    if "attack_label" in df.columns:
        # Treat anything not 'normal' as attack
        return df["attack_label"].astype(str).str.lower().ne("normal")

    raise ValueError(
        "Could not infer attack/benign labels. Expected 'binary_label' or 'attack_label' in the parquet."
    )


def main():
    print("Loading latest feature list from final_model artifacts...")
    run_dir = get_latest_run_dir(FINAL_MODEL_ROOT)
    feature_names = load_json(run_dir / "feature_names.json")

    print("Loading processed dataset...")
    df = pd.read_parquet(INPUT_FEATURES_PARQUET)

    # Keep only rows that have all feature columns (missing columns would break batch inference)
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input parquet is missing {len(missing_cols)} required feature columns, e.g.: {missing_cols[:10]}"
        )

    attack_mask = infer_attack_mask(df)

    df_attack = df.loc[attack_mask].copy()
    df_benign = df.loc[~attack_mask].copy()

    if len(df_attack) == 0 or len(df_benign) == 0:
        raise ValueError(
            f"Not enough rows to sample. attack={len(df_attack)}, benign={len(df_benign)}"
        )

    n_attack = min(N_ATTACK, len(df_attack))
    n_benign = min(N_BENIGN, len(df_benign))

    print(f"Sampling attack={n_attack}, benign={n_benign} ...")
    sample_attack = df_attack.sample(n=n_attack, random_state=RANDOM_STATE)
    sample_benign = df_benign.sample(n=n_benign, random_state=RANDOM_STATE)

    batch_df = pd.concat([sample_attack, sample_benign], axis=0).sample(
        frac=1.0, random_state=RANDOM_STATE
    )

    # Optional: keep labels for your own verification (predict.py will ignore them)
    keep_cols = list(feature_names)
    for extra in ["binary_label", "attack_label", "flow_id", "source_ip", "destination_ip", "protocol", "state"]:
        if extra in batch_df.columns:
            keep_cols.append(extra)

    batch_df = batch_df[keep_cols].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_df.to_parquet(OUT_FILE, index=False)

    print(f"Saved: {OUT_FILE}")
    print("Preview label counts (if available):")
    if "binary_label" in batch_df.columns:
        print(batch_df["binary_label"].value_counts(dropna=False))
    elif "attack_label" in batch_df.columns:
        print(batch_df["attack_label"].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()
