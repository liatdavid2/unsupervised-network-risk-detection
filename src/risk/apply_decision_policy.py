import yaml
from pathlib import Path
import pandas as pd


def load_policy(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_policy(scored_df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """
    Adds SOC decision columns based on risk_level and policy rules.
    """
    out = scored_df.copy()

    tier_rules = policy["risk_tiers"]

    def decide(row):
        tier = row["risk_level"]
        rule = tier_rules.get(tier, {})
        return rule.get("action", "monitor")

    def severity(row):
        tier = row["risk_level"]
        rule = tier_rules.get(tier, {})
        return rule.get("severity", "informational")

    def sla(row):
        tier = row["risk_level"]
        rule = tier_rules.get(tier, {})
        return rule.get("sla_minutes")

    out["soc_action"] = out.apply(decide, axis=1)
    out["severity"] = out.apply(severity, axis=1)
    out["sla_minutes"] = out.apply(sla, axis=1)

    return out
