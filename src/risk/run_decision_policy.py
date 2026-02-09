from pathlib import Path
import pandas as pd
from src.risk.apply_decision_policy import load_policy, apply_policy


def get_latest_run(base_dir: Path) -> Path:
    runs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not runs:
        raise RuntimeError(f"No runs found in {base_dir}")
    return max(runs, key=lambda p: p.name)


def main():
    base_dir = Path("artifacts/unsupervised_model")

    run_dir = get_latest_run(base_dir)
    print(f"[INFO] Using latest run: {run_dir}")

    # Load scored data
    scored = pd.read_parquet(run_dir / "train_scored.parquet")

    # Load decision policy
    policy = load_policy(Path("policies/decision_policy.yaml"))

    # Apply policy
    soc_table = apply_policy(scored, policy)

    # ---- Preview decisions (what the SOC would see) ----
    print("\n[SAMPLE] SOC Decisions (first 10 rows):")
    print(
        soc_table[
            [
                "anomaly_score",
                "risk_level",
            ]
        ].head(10)
    )

    print("\n[SAMPLE] Risk level distribution:")
    print(soc_table["risk_level"].value_counts())

    # ---- Save outputs ----
    parquet_path = run_dir / "soc_decisions.parquet"
    csv_path = run_dir / "soc_decisions.csv"

    soc_table.to_parquet(parquet_path, index=False)
    soc_table.to_csv(csv_path, index=False)

    print(f"\n[DONE] Saved SOC decisions to:")
    print(f"  - {parquet_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
