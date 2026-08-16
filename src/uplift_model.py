"""
Uplift modeling: who should receive a retention offer.

*** IMPORTANT — read before trusting any number this script prints ***
The KKBox dataset has no actual retention-offer experiment in it — no
randomized treatment, no A/B test. To demonstrate the uplift-modeling
technique honestly, this script SIMULATES a randomized retention offer on
top of the real covariates and real historical churn outcome:
  - treatment assignment is a coin flip (mimicking a real 50/50 A/B test)
  - the *effect* of that offer on each member is a hand-authored function
    of real covariates (defined in `true_treatment_effect()` below), used
    only to generate a simulated outcome and, separately, as ground truth
    for validating the fitted model
Every number downstream of this simulation — the Qini curve, the "who's a
persuadable" list — is a demonstration of the *method* on realistic data,
not a real business finding. In production, this model would instead be
fit on outcomes from an actual holdout-controlled retention campaign; see
README "Limitations" and the A/B testing note at the bottom of this file.

Run as a script (after ltv_estimation.py has produced ltv_table.parquet):
    python -m src.uplift_model
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from econml.metalearners import TLearner
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src import config, mlflow_utils
from src.churn_model import build_xy, load_period

TREATMENT_COL = "treatment"
OUTCOME_COL = "observed_retained"  # 1 = retained, 0 = churned — "higher is better," standard uplift convention
TRUE_EFFECT_COL = "true_treatment_effect"  # ground truth, simulation-only


# ---------------------------------------------------------------------
# Simulation: synthetic randomized retention offer
# ---------------------------------------------------------------------


def true_treatment_effect(df: pd.DataFrame) -> np.ndarray:
    """
    Hand-authored ground-truth CATE (change in P(retained) from receiving
    the offer), built to produce the four classic uplift segments out of
    real covariates already in the feature table:

      - persuadables: engagement is declining but not dead, and they're
        near a renewal decision point -> offer meaningfully helps
      - sure things: already highly engaged / auto-renewing -> would stay
        regardless, offer adds ~nothing
      - lost causes: long dormant (hasn't listened in ~a year) -> an email
        about a subscription they've mentally already left won't help
      - sleeping dogs: brand-new trial members -> a "please don't leave"
        offer this early can read as pushy and slightly backfire

    Every threshold here is a modeling choice for the simulation, not a
    fact about real KKBox users — the point is to create a realistic,
    heterogeneous effect surface to test whether the uplift model can
    recover it from data that only ever sees ONE arm per member (as in a
    real experiment; see simulate_experiment() below).
    """
    momentum = df["engagement_momentum"].fillna(1.0).clip(0, 3)
    days_since_log = df["days_since_last_log"].fillna(400)
    expires_in = df["membership_expires_in_days"].fillna(999)
    tenure = df["tenure_days"].fillna(0)
    auto_renew = df["is_auto_renew"].fillna(0)

    # Near-decision-point boost: peaks when membership expires within the
    # next ~30 days (a natural moment for an offer to matter).
    near_decision = np.exp(-((expires_in - 10) ** 2) / (2 * 20**2))

    # Declining-but-engaged boost: momentum around 0.4-0.8 (down, not gone).
    declining_engaged = np.exp(-((momentum - 0.6) ** 2) / (2 * 0.25**2))

    persuadable_effect = 0.20 * near_decision * declining_engaged

    # Sure things: long tenure + auto-renew already -> effect shrinks to ~0.
    sure_thing_damping = 1 - np.clip((tenure / 900) * auto_renew, 0, 1)

    # Lost causes: essentially inactive for ~a year -> effect shrinks to ~0.
    lost_cause_damping = np.clip(1 - (days_since_log / 365), 0, 1)

    effect = 0.03 + persuadable_effect  # small baseline lift + persuadable boost
    effect = effect * sure_thing_damping * lost_cause_damping

    # Sleeping dogs: very new members (<14 days tenure) get a small
    # negative effect — an early "don't go" offer can backfire.
    new_member_backfire = np.where(tenure < 14, -0.03, 0.0)

    total = effect + new_member_backfire
    return np.clip(total, -0.05, 0.30)


def simulate_experiment(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Standard potential-outcomes simulation:
      Y0 = potential outcome under control = the member's real historical
           outcome (1 - is_churn) — i.e. "what actually happened, absent
           this hypothetical offer program"
      Y1 = potential outcome under treatment = Y0, nudged toward the
           opposite state with probability |true_effect|
      T  = simulated random assignment (Bernoulli(0.5))
      observed_retained = Y1 if T==1 else Y0  — the fundamental problem of
           causal inference: only one potential outcome is ever "observed"
           per unit, exactly as in a real experiment.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    y0 = 1 - df["is_churn"].astype(int).to_numpy()  # retained under control
    true_effect = true_treatment_effect(df)

    flip_roll = rng.uniform(size=len(df))
    y1 = y0.copy()
    saved = (y0 == 0) & (true_effect > 0) & (flip_roll < true_effect)
    y1[saved] = 1
    backfired = (y0 == 1) & (true_effect < 0) & (flip_roll < -true_effect)
    y1[backfired] = 0

    treatment = rng.integers(0, 2, size=len(df))
    observed = np.where(treatment == 1, y1, y0)

    df[TREATMENT_COL] = treatment
    df[OUTCOME_COL] = observed
    df[TRUE_EFFECT_COL] = true_effect
    return df


# ---------------------------------------------------------------------
# Uplift model
# ---------------------------------------------------------------------


def prepare_uplift_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unlike XGBoost (churn_model.py, ltv_estimation.py's classifier),
    econml validates its own inputs before ever touching the wrapped
    XGBRegressor, and that validation rejects NaN outright regardless of
    whether the underlying model could've handled it natively — so, unlike
    those other callers, this one needs an actual median-fill, not just a
    dtype cast. Still starts from build_xy()'s cast, though: age_is_valid's
    nullable-boolean-turned-object-dtype column would otherwise skip
    `.median(numeric_only=True)` entirely (it's not recognized as numeric)
    and come back out still full of NaNs, un-imputed.
    """
    X, _ = build_xy(df)
    return X.fillna(X.median(numeric_only=True))


def fit_uplift_model(df: pd.DataFrame) -> TLearner:
    """
    T-learner: one regression model fit on treated units, one on control
    units, CATE = predicted-outcome-if-treated minus predicted-outcome-if-
    control. This is the right tool here (rather than a DML/causal-forest
    approach that adjusts for confounding) *because* treatment was randomly
    assigned in the simulation — there's no confounding to correct for, so
    the simplest valid estimator is the right choice, not a weaker one.
    A real observational deployment (offers targeted by a human, not
    randomized) would need DML or a doubly-robust estimator instead, since
    treatment assignment there would correlate with covariates.
    """
    X = prepare_uplift_features(df)
    T = df[TREATMENT_COL].to_numpy()
    Y = df[OUTCOME_COL].to_numpy().astype(float)

    learner = TLearner(
        models=XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=config.RANDOM_SEED)
    )
    learner.fit(Y=Y, T=T, X=X)
    return learner


def predict_uplift(learner: TLearner, df: pd.DataFrame) -> np.ndarray:
    X = prepare_uplift_features(df)
    return learner.effect(X=X)


# ---------------------------------------------------------------------
# Qini curve
# ---------------------------------------------------------------------


def qini_curve(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """
    Standard Qini curve (Radcliffe 2007): sort by predicted uplift
    descending, and at each cumulative population fraction compute the
    incremental "retained" count a targeting policy would have captured
    over what the same-size treated/control ratio would give by chance:

        qini(k) = sum(Y=1, T=1, top k) - sum(Y=1, T=0, top k) * (n_treated_topk / n_control_topk)

    This is directly comparable to a "random targeting" diagonal from
    (0, 0) to (N, overall_qini) — a model that's no better than random
    targeting produces a curve that IS that diagonal; a useful model bows
    above it. The area between the curve and the diagonal is the Qini
    coefficient (this function's caller computes that).
    """
    ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    treated = ordered[TREATMENT_COL].to_numpy()
    outcome = ordered[OUTCOME_COL].to_numpy()

    cum_treated = np.cumsum(treated)
    cum_control = np.cumsum(1 - treated)
    cum_y_treated = np.cumsum(outcome * treated)
    cum_y_control = np.cumsum(outcome * (1 - treated))

    # Avoid divide-by-zero in the earliest rows before both arms appear.
    ratio = np.divide(
        cum_treated, cum_control, out=np.zeros_like(cum_treated, dtype=float), where=cum_control > 0
    )
    qini_values = cum_y_treated - cum_y_control * ratio

    population_fraction = np.arange(1, len(ordered) + 1) / len(ordered)
    return pd.DataFrame({"population_fraction": population_fraction, "qini": qini_values})


def qini_coefficient(qini_df: pd.DataFrame) -> float:
    """Area between the model's Qini curve and the random-targeting
    diagonal, normalized by population size — the uplift-modeling analogue
    of the Gini coefficient for a ranking model."""
    x = qini_df["population_fraction"].to_numpy()
    y = qini_df["qini"].to_numpy()
    random_line = x * y[-1]  # diagonal from (0,0) to (1, qini at 100%)
    return float(np.trapz(y - random_line, x))


def plot_qini(curves: dict[str, pd.DataFrame], out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, qini_df in curves.items():
        ax.plot(qini_df["population_fraction"], qini_df["qini"], label=label)
    # Random-targeting reference line, from the model curve's own endpoint.
    any_curve = next(iter(curves.values()))
    ax.plot(
        [0, 1],
        [0, any_curve["qini"].iloc[-1]],
        "k--",
        label="random targeting",
    )
    ax.set_xlabel("Fraction of population targeted (sorted by predicted uplift)")
    ax.set_ylabel("Cumulative incremental retained members")
    ax.set_title("Qini curve — retention offer uplift model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    mlflow_utils.init_mlflow()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    period2 = load_period("period2")  # most recent snapshot — the "current" cohort to target
    experiment_df = simulate_experiment(period2)

    # Held-out evaluation, same discipline as churn_model.py's time-based
    # OOT split (stratified here on treatment rather than time, since a
    # single simulated experiment has no "future period" to hold out).
    # This was NOT optional during development: evaluating Qini in-sample
    # let the 200-tree base learner overfit to the noise in this one random
    # experiment realization — which specific unit happened to get "saved"
    # by chance, rather than the true underlying effect — and that
    # inflated the fitted model's Qini coefficient to 3x the oracle's
    # (mathematically impossible for genuine generalization, since the
    # oracle ranks by the actual ground-truth effect). A held-out split
    # makes that overfitting visible instead of silently rewarding it.
    train_df, test_df = train_test_split(
        experiment_df, test_size=0.3, stratify=experiment_df[TREATMENT_COL], random_state=config.RANDOM_SEED
    )

    with mlflow.start_run(run_name="uplift_model"):
        learner = fit_uplift_model(train_df)
        test_df = test_df.copy()
        test_df["predicted_uplift"] = predict_uplift(learner, test_df)

        model_qini = qini_curve(test_df, "predicted_uplift")
        oracle_qini = qini_curve(test_df, TRUE_EFFECT_COL)  # only possible b/c this is simulated
        random_qini = qini_curve(
            test_df.assign(_random_score=np.random.default_rng(0).normal(size=len(test_df))),
            "_random_score",
        )

        model_coef = qini_coefficient(model_qini)
        oracle_coef = qini_coefficient(oracle_qini)
        random_coef = qini_coefficient(random_qini)

        print(f"Qini coefficient — fitted model: {model_coef:.3f}")
        print(f"Qini coefficient — oracle (true simulated CATE, upper bound): {oracle_coef:.3f}")
        print(f"Qini coefficient — random targeting (sanity check, ~0): {random_coef:.3f}")

        plot_qini(
            {"fitted T-learner": model_qini, "oracle (simulation ground truth)": oracle_qini},
            config.PLOTS_DIR / "qini_curve.png",
        )

        mlflow.log_metric("qini_coefficient_model", model_coef)
        mlflow.log_metric("qini_coefficient_oracle", oracle_coef)
        mlflow.log_metric("qini_coefficient_random", random_coef)
        mlflow.log_param("uplift_method", "T-learner (econml), XGBRegressor base")
        mlflow.log_param("treatment_source", "SIMULATED randomized offer — see module docstring")

        # --- Business layer: combine uplift with LTV for a targeting list ---
        # Score the FULL population here, not just the held-out test
        # fraction: the Qini evaluation above is what tells us the model
        # generalizes; a real deployment scores everyone with the model
        # that evaluation already validated, the same relationship
        # churn_model.py has between its OOT eval and champion registration.
        experiment_df["predicted_uplift"] = predict_uplift(learner, experiment_df)
        ltv_table = pd.read_parquet(config.FEATURES_DIR / "ltv_table.parquet")
        targeting = experiment_df[["msno", "predicted_uplift"]].merge(
            ltv_table[["msno", "ltv", "churn_probability", "retention_priority_segment"]], on="msno"
        )
        targeting["expected_value_of_offer"] = targeting["predicted_uplift"] * targeting["ltv"]
        targeting = targeting.sort_values("expected_value_of_offer", ascending=False)

        out_path = config.FEATURES_DIR / "uplift_targeting_list.parquet"
        targeting.to_parquet(out_path, index=False)
        print(f"\nsaved {out_path}")
        print("\nTop 10 retention-offer targets by expected value (uplift x LTV):")
        print(
            targeting.head(10)[
                ["msno", "predicted_uplift", "ltv", "churn_probability", "expected_value_of_offer"]
            ].to_string(index=False)
        )

        by_segment = targeting.groupby("retention_priority_segment")["predicted_uplift"].mean()
        print("\nMean predicted uplift by LTV/risk segment (from ltv_estimation.py):")
        print(by_segment)


if __name__ == "__main__":
    sys.exit(main())
