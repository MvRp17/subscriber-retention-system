"""
LTV = expected remaining lifetime (from the Cox survival model) x margin
per period.

Run as a script (after survival_analysis.py and churn_model.py have run —
this reuses the saved Cox model and the registered champion classifier):
    python -m src.ltv_estimation

Method — conditional expected remaining lifetime under proportional hazards:

A Cox model gives each member i a survival function S_i(t) = S_0(t)^theta_i,
where S_0 is the shared baseline survival curve and theta_i =
exp(beta . x_i) is that member's hazard ratio relative to baseline. Given a
member has *already* survived to their current tenure t0 (all of them
have — that's what tenure_days means), the textbook mean-residual-life
formula for their expected remaining lifetime is:

    E[T - t0 | T > t0] = (1 / S_i(t0)) * integral_{t0}^{horizon} S_i(u) du

We can't integrate to infinity — the Cox baseline is only estimated out to
the longest tenure actually observed in the training data, so `horizon`
caps there. Implicitly, this assumes a flat hazard past that point (whoever
survives to the edge of our data keeps churning at that same rate
indefinitely) — a real limitation for very long-tenured members, called out
in README "Limitations".
"""
import pickle
import sys

import mlflow
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from src import config, mlflow_utils
from src.churn_model import build_xy
from src.survival_analysis import CoxPreprocessor, DURATION_COL, load_survival_frame


def expected_remaining_lifetime(cph: CoxPHFitter, preprocessor: CoxPreprocessor, df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized (no per-member Python loop) conditional expected remaining
    lifetime in days, via the S_i(t) = S_0(t)^theta_i identity above.
    """
    tenure = df[DURATION_COL].to_numpy(dtype=float)

    # Must be the SAME impute/clip/scale transform the model was fit on —
    # see CoxPreprocessor's docstring. Using raw covariates here would run
    # without error and just silently produce wrong hazard ratios.
    covariates = preprocessor.transform(df)
    theta = cph.predict_partial_hazard(covariates).to_numpy(dtype=float)  # shape (N,)

    baseline = cph.baseline_survival_.iloc[:, 0]  # Series: index=time, value=S_0(time)
    baseline_times = baseline.index.to_numpy(dtype=float)
    baseline_values = baseline.to_numpy(dtype=float)
    horizon = float(baseline_times.max())

    # Members whose tenure already exceeds the observed horizon have no
    # remaining-lifetime data to extrapolate from; clip them to just inside
    # the horizon so the integral below is well-defined (their estimated
    # remaining life will be ~0, an intentional floor — see module docstring).
    tenure_clipped = np.minimum(tenure, horizon - 1e-6)

    def baseline_survival_at(t: np.ndarray) -> np.ndarray:
        """Step-function lookup of S_0(t) at arbitrary times via searchsorted
        — this is the vectorized equivalent of lifelines' own interpolation,
        applied to a whole array at once instead of one t at a time."""
        idx = np.searchsorted(baseline_times, t, side="right") - 1
        idx = np.clip(idx, 0, len(baseline_values) - 1)
        return baseline_values[idx]

    n = len(df)
    grid_frac = np.linspace(0.0, 1.0, config.LTV_INTEGRATION_GRID_POINTS)  # shape (G,)
    # Per-member grid from their own tenure out to the shared horizon —
    # built as one (N, G) matrix via broadcasting, not a loop.
    grid = tenure_clipped[:, None] + (horizon - tenure_clipped[:, None]) * grid_frac[None, :]

    s0_grid = baseline_survival_at(grid.ravel()).reshape(n, config.LTV_INTEGRATION_GRID_POINTS)
    s_i_grid = s0_grid ** theta[:, None]  # S_i(t) = S_0(t) ** theta_i, broadcast over grid

    s_i_t0 = s_i_grid[:, 0]  # S_i(tenure) — first grid column by construction
    s_i_t0_safe = np.where(s_i_t0 > 1e-9, s_i_t0, 1e-9)
    conditional_survival = s_i_grid / s_i_t0_safe[:, None]

    relative_time = grid - tenure_clipped[:, None]  # integration variable: (t - t0)
    remaining_life = np.trapz(conditional_survival, x=relative_time, axis=1)
    return np.clip(remaining_life, 0, None)


def daily_revenue(df: pd.DataFrame) -> pd.Series:
    """
    Revenue-per-day-subscribed from each member's most recent plan. Falls
    back to the population median for members with no usable transaction
    (no plan on file, or a plan with 0 listed days) rather than dropping
    them from the LTV table entirely.
    """
    revenue = df["last_actual_amount_paid"] / df["last_payment_plan_days"].replace(0, np.nan)
    return revenue.fillna(revenue.median())


def load_champion_classifier():
    """
    Pulls the champion model straight from the MLflow Model Registry (by
    alias, not a hardcoded file path) — the same registry churn_model.py
    just wrote to. This is the point of a registry: downstream consumers
    don't need to know which run produced the model or where its artifact
    happens to live on disk.

    Deliberately does NOT use mlflow.pyfunc.load_model().predict() here.
    pyfunc's generic predict() calls the underlying flavor's own .predict(),
    which for the xgboost flavor means XGBClassifier.predict() — hard 0/1
    class labels, not probabilities. That's a silent, no-error bug: this
    script would have run fine and quietly written a churn_probability
    column that only ever contained exactly 0.0 or exactly 1.0, corrupting
    every downstream LTV/uplift segmentation without a single exception
    anywhere. Loading the model via its native flavor and calling
    .predict_proba() explicitly is the reliable way to get calibrated-scale
    probabilities regardless of which model type (xgboost, or a
    sklearn-flavored logreg/CalibratedClassifierCV) happens to be champion.
    """
    mlflow_utils.init_mlflow()
    model_uri = "models:/churn-classifier@champion"
    flavors = mlflow.pyfunc.load_model(model_uri).metadata.flavors
    if "xgboost" in flavors:
        return mlflow.xgboost.load_model(model_uri)
    return mlflow.sklearn.load_model(model_uri)


def main():
    mlflow_utils.init_mlflow()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_survival_frame()
    with open(config.MODELS_DIR / "cox_model.pkl", "rb") as f:
        saved = pickle.load(f)
    cph: CoxPHFitter = saved["model"]
    preprocessor: CoxPreprocessor = saved["preprocessor"]

    with mlflow.start_run(run_name="ltv_estimation"):
        df = df.copy()
        df["expected_remaining_days"] = expected_remaining_lifetime(cph, preprocessor, df)
        df["daily_revenue"] = daily_revenue(df)
        df["ltv"] = (
            df["expected_remaining_days"] * df["daily_revenue"] * config.ASSUMED_GROSS_MARGIN_RATE
        )

        classifier = load_champion_classifier()
        # build_xy() (not a raw df[ALL_FEATURES] slice) so the same
        # object-dtype defensive cast churn_model.py relies on applies here
        # too — the registry-loaded XGBoost model rejects a nullable-boolean
        # column exactly like fitting one does.
        X_for_scoring, _ = build_xy(df)
        # predict_proba, not predict() — see load_champion_classifier()'s
        # docstring for why that distinction is the whole ballgame here.
        df["churn_probability"] = classifier.predict_proba(X_for_scoring)[:, 1]

        # Retention-prioritization quadrant: the business-facing payoff of
        # combining LTV with churn risk. High LTV + high risk is exactly
        # the segment a retention offer budget should go to first; the
        # uplift model in uplift_model.py refines this further by asking
        # who among them would actually respond to an offer.
        #
        # Split on PERCENTILE RANK, not the raw median value. A tree
        # ensemble's predicted probabilities are often heavily tied (many
        # members land in the same leaf and get the identical score) —
        # comparing raw values against `.median()` degenerates when a tied
        # block straddles the median (a >=/< split can put 0% of members on
        # one side of a threshold that isn't strictly between two distinct
        # values). Ranking first guarantees a genuine ~50/50 split on each
        # axis regardless of how skewed or tied the underlying values are.
        ltv_rank = df["ltv"].rank(pct=True)
        risk_rank = df["churn_probability"].rank(pct=True)
        df["retention_priority_segment"] = np.select(
            [
                (ltv_rank >= 0.5) & (risk_rank >= 0.5),
                (ltv_rank >= 0.5) & (risk_rank < 0.5),
                (ltv_rank < 0.5) & (risk_rank >= 0.5),
            ],
            ["high_value_at_risk", "high_value_stable", "low_value_at_risk"],
            default="low_value_stable",
        )

        out_cols = [
            "msno",
            "period",
            DURATION_COL,
            "expected_remaining_days",
            "daily_revenue",
            "ltv",
            "churn_probability",
            "retention_priority_segment",
        ]
        out_path = config.FEATURES_DIR / "ltv_table.parquet"
        df[out_cols].to_parquet(out_path, index=False)

        summary = df.groupby("retention_priority_segment").agg(
            n_members=("msno", "count"), total_ltv=("ltv", "sum"), avg_ltv=("ltv", "mean")
        )
        print(summary)
        print(f"\nsaved {out_path}")

        mlflow.log_metric("median_ltv", df["ltv"].median())
        mlflow.log_metric("mean_expected_remaining_days", df["expected_remaining_days"].mean())
        mlflow.log_metric(
            "high_value_at_risk_total_ltv",
            df.loc[df["retention_priority_segment"] == "high_value_at_risk", "ltv"].sum(),
        )


if __name__ == "__main__":
    sys.exit(main())
