import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold


# =========================
# Paths
# =========================
RAW_DATA_PATH = Path("data/raw/UNSW_Flow.parquet")
OUTPUT_DATA_PATH = Path("data/processed/UNSW_Flow_features.parquet")

TARGET_COL = "binary_label"


# =========================
# Columns to exclude
# =========================
NON_FEATURE_COLS = [
    "source_ip",
    "destination_ip",
    "protocol",
    "state",
    "service",
    "attack_label",
    "binary_label",
]

IDENTIFIER_COLS = [
    "src_ip",
    "dst_ip",
    "flow_id",
    "session_id",
    "timestamp",
    "start_time",
    "end_time",
]


# =========================
# Logging helper
# =========================
def log(step: str, message: str):
    print(f"[{step}] {message}")


# =========================
# Preprocessing helpers
# =========================
def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in IDENTIFIER_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def variance_filter(df: pd.DataFrame, threshold: float = 0.0):
    selector = VarianceThreshold(threshold=threshold)
    values = selector.fit_transform(df)

    selected_features = df.columns[selector.get_support()]
    df_selected = pd.DataFrame(values, columns=selected_features, index=df.index)

    return df_selected, list(selected_features)


def correlation_filter(df: pd.DataFrame, threshold: float = 0.95):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    df_filtered = df.drop(columns=to_drop)
    return df_filtered, to_drop


# =========================
# Main pipeline
# =========================
def build_features():
    # [1/6] Load data
    df = pd.read_parquet(RAW_DATA_PATH)
    log("1/6", f"Load data ............... OK ({len(df):,} rows)")

    # [2/6] Target validation
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    y = df[TARGET_COL]
    attack_rate = y.value_counts(normalize=True).get(1, 0) * 100
    log("2/6", f"Target detected ......... {TARGET_COL} ({attack_rate:.2f}% attacks)")

    # [3/6] Drop non-numeric / identifiers
    non_numeric_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    X = df.drop(
        columns=[c for c in NON_FEATURE_COLS if c in df.columns],
        errors="ignore"
    )
    X = drop_identifier_columns(X)
    X = X.select_dtypes(include=[np.number])

    log("3/6", f"Drop non-numeric ........ {len(non_numeric_cols)} columns removed")

    # [4/6] Numeric features + missing values
    before_features = X.shape[1]
    X = handle_missing_values(X)
    log("4/6", f"Numeric features ........ {before_features}")

    # [5/6] Feature selection
    X, _ = variance_filter(X, threshold=0.0)
    after_variance = X.shape[1]

    X, _ = correlation_filter(X, threshold=0.95)
    after_corr = X.shape[1]

    log("5/6", f"Feature selection ....... {after_variance} → {after_corr}")

    # [6/6] Save output
    X[TARGET_COL] = y.values
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    X.to_parquet(OUTPUT_DATA_PATH, index=False)

    log("6/6", f"Save features ........... {OUTPUT_DATA_PATH}")

    # Short summary
    print("\nClass distribution:")
    print(f"  normal : {(1 - attack_rate/100)*100:.2f}%")
    print(f"  attack : {attack_rate:.2f}%")

    return list(X.columns)


if __name__ == "__main__":
    build_features()
