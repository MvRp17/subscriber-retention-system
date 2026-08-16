"""
Basic feature drift monitoring via Population Stability Index (PSI).

Run as a script to compare the two feature snapshots already in this repo
(a concrete, reproducible example of what this would check in production):
    python -m src.drift_monitor

In production this would run on a schedule (see retrain_pipeline.py),
comparing whatever feature table just got built against the table the
currently-deployed model was trained on, and alert (not silently retrain)
if key features have drifted — a model can keep scoring happily on data
that's quietly stopped resembling what it learned from.

PSI, not KS-test or a hosted drift tool: PSI is the standard credit-risk /
subscription-analytics metric for exactly this — it's simple to compute,
gives one interpretable number per feature, and the 0.1 / 0.25 thresholds
below are widely used rules of thumb, which makes this easy to explain to
a non-ML stakeholder (a compliance-flavored audience, appropriately, for a
metric that started in credit scoring).
"""
import sys

import numpy as np
import pandas as pd

from src import config
from src.churn_model import NUMERIC_FEATURES

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25
N_BINS = 10


def population_stability_index(reference: pd.Series, current: pd.Series, n_bins: int = N_BINS) -> float:
    """
    PSI between a reference distribution and a current one, using
    reference-defined quantile bins (so bin edges are fixed from the
    training distribution, not recomputed on the new data — recomputing
    them on `current` would hide exactly the shift we're trying to catch).
    """
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(reference.quantile(quantiles).to_numpy())
    if len(bin_edges) < 3:
        return 0.0  # feature has almost no variance in reference; nothing meaningful to compare

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Laplace-smooth to avoid log(0) / divide-by-zero on empty bins.
    ref_pct = (ref_counts + 1) / (ref_counts.sum() + n_bins)
    cur_pct = (cur_counts + 1) / (cur_counts.sum() + n_bins)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_FEATURES:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = population_stability_index(reference_df[col], current_df[col])
        if psi > PSI_SIGNIFICANT_THRESHOLD:
            status = "SIGNIFICANT DRIFT"
        elif psi > PSI_MODERATE_THRESHOLD:
            status = "moderate drift"
        else:
            status = "stable"
        rows.append({"feature": col, "psi": psi, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def main():
    reference_df = pd.read_parquet(config.FEATURES_DIR / "period1.parquet")
    current_df = pd.read_parquet(config.FEATURES_DIR / "period2.parquet")

    report = drift_report(reference_df, current_df)
    pd.set_option("display.max_rows", None)
    print("Feature drift: period1 (training reference) -> period2 (current)\n")
    print(report.to_string(index=False))

    flagged = report[report["psi"] > PSI_MODERATE_THRESHOLD]
    if not flagged.empty:
        print(f"\n{len(flagged)} feature(s) drifted beyond the 'moderate' threshold ({PSI_MODERATE_THRESHOLD}):")
        print(", ".join(flagged["feature"]))
    else:
        print(f"\nNo features exceeded the moderate drift threshold ({PSI_MODERATE_THRESHOLD}).")

    report_path = config.MODELS_DIR / "drift_report.csv"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    print(f"\nsaved {report_path}")


if __name__ == "__main__":
    sys.exit(main())
