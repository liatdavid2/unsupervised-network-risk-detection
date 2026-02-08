import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold


# =========================
# Paths
# =========================
RAW_DATA_PATH = Path("data/raw/UNSW_Flow.parquet")
OUTPUT_DATA_PATH = Path("data/processed/UNSW_Flow_features.parquet")


# =========================
# Columns to exclude from X
# =========================

LABEL_COLS = [
    "attack_label",
    "binary_label",
]

CATEGORICAL_COLS = [
    "source_ip",
    "destination_ip",
    "protocol",
    "state",
    "service",
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

def drop_columns_if_exist(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")


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
    # [1/5] Load data
    df = pd.read_parquet(RAW_DATA_PATH)
    log("1/5", f"Load data ............... OK ({len(df):,} rows)")

    # Keep labels aside (NOT used for feature selection)
    labels = df[LABEL_COLS].copy() if all(c in df.columns for c in LABEL_COLS) else None

    # [2/5] Drop non-feature columns
    X = df.copy()
    X = drop_columns_if_exist(X, LABEL_COLS)
    X = drop_columns_if_exist(X, CATEGORICAL_COLS)
    X = drop_columns_if_exist(X, IDENTIFIER_COLS)

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])
    log("2/5", f"Numeric features ........ {X.shape[1]}")

    # [3/5] Handle missing values
    X = handle_missing_values(X)
    log("3/5", "Missing values handled")

    # [4/5] Feature selection
    before = X.shape[1]

    X, _ = variance_filter(X, threshold=0.0)
    after_variance = X.shape[1]

    X, _ = correlation_filter(X, threshold=0.95)
    after_corr = X.shape[1]

    log("4/5", f"Feature selection ....... {before} → {after_variance} → {after_corr}")

    # [5/5] Save output (features + labels for evaluation only)
    OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if labels is not None:
        X = pd.concat([X, labels], axis=1)

    X.to_parquet(OUTPUT_DATA_PATH, index=False)
    log("5/5", f"Save features ........... {OUTPUT_DATA_PATH}")

    return list(X.columns)


if __name__ == "__main__":
    build_features()
