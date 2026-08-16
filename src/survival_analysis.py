"""
Survival analysis: Kaplan-Meier + Cox Proportional Hazards on right-censored
subscriber tenure, and an explicit comparison against the point-in-time
churn classifier.

Run as a script (after feature_engineering.py):
    python -m src.survival_analysis

Framing (why this is a different question than churn_model.py):

The classifier answers "will THIS member churn in the ~30-day window used
to define is_churn?" — a fixed-horizon point prediction. Survival analysis
answers a different question: "given how long a member has already been
subscribed, what does their full retention curve look like, and which
covariates shift that curve up or down across their ENTIRE tenure?" A
member can have low 30-day churn risk right now but a hazard curve that's
climbing every month; the classifier is blind to that shape, survival
analysis is built around it.

Duration/event setup — a deliberate simplification, spelled out here rather
than left implicit:
  - duration = tenure_days (registration -> the period's cutoff date)
  - event    = is_churn (1 = churn observed in that period's label window,
               0 = still subscribed at the cutoff -> right-censored: we
               only know they survived AT LEAST that long)
This treats each (member, period) row as a single cross-sectional
observation of "how long have they lasted, and did they churn in this
window" rather than reconstructing full multi-episode subscription
histories from the raw transaction log. That's a reasonable read of a
snapshot-labeled dataset like this one, but it's a real simplification —
see README "Limitations" for what a production version (time-varying
covariates, recurrent-event survival models) would add. Both periods are
pooled for more data; a member appearing in both is mild pseudo-replication
we accept for this portfolio scope rather than correct for.
"""
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src import config, mlflow_utils
from src.churn_model import build_xgboost_model, build_xy, load_period

DURATION_COL = "tenure_days"
EVENT_COL = "is_churn"

# A smaller, hand-picked covariate set for Cox regression — not every churn
# model feature. Cox convergence and interpretability both degrade with too
# many correlated covariates (e.g. the four total_secs_last_Xd windows are
# highly collinear with each other); a focused set is more defensible in
# an interview than "I threw all 40 features at it."
COX_COVARIATES = [
    "is_auto_renew",
    "n_cancellations_lifetime",
    "last_plan_list_price",
    "total_secs_last_90d",
    "active_days_last_90d",
    "completion_ratio",
    "engagement_momentum",
    "days_since_last_transaction",
    "membership_expires_in_days",
]
CONTINUOUS_COX_COVARIATES = [c for c in COX_COVARIATES if c != "is_auto_renew"]


class CoxPreprocessor:
    """
    The impute/clip/scale transform Cox covariates go through before
    fitting — bundled into one object and pickled alongside the fitted
    CoxPHFitter so every consumer (the classifier comparison in this
    module, and expected_remaining_lifetime() in ltv_estimation.py) applies
    the *exact same* transform the model was fit on. predict_partial_hazard
    doesn't validate its input is on the right scale — feeding it raw,
    unstandardized covariates wouldn't error, it would just silently return
    hazard ratios in the wrong units.
    """

    def __init__(self):
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cox_df = df[COX_COVARIATES].copy()
        self.medians_ = cox_df.median()
        cox_df = cox_df.fillna(self.medians_)

        # See fit_cox_model()'s docstring note for why: this ratio explodes
        # when its denominator (prior-30d listening seconds) is near zero.
        cox_df["engagement_momentum"] = cox_df["engagement_momentum"].clip(upper=5)

        self.means_ = cox_df[CONTINUOUS_COX_COVARIATES].mean()
        self.stds_ = cox_df[CONTINUOUS_COX_COVARIATES].std()
        cox_df[CONTINUOUS_COX_COVARIATES] = (
            cox_df[CONTINUOUS_COX_COVARIATES] - self.means_
        ) / self.stds_
        return cox_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("call fit_transform() before transform()")
        cox_df = df[COX_COVARIATES].copy()
        cox_df = cox_df.fillna(self.medians_)
        cox_df["engagement_momentum"] = cox_df["engagement_momentum"].clip(upper=5)
        cox_df[CONTINUOUS_COX_COVARIATES] = (
            cox_df[CONTINUOUS_COX_COVARIATES] - self.means_
        ) / self.stds_
        return cox_df


def load_survival_frame() -> pd.DataFrame:
    """Pool both periods; keep the columns survival analysis needs."""
    period1 = load_period("period1")
    period2 = load_period("period2")
    df = pd.concat([period1, period2], ignore_index=True)
    df = df[df[DURATION_COL].notna() & (df[DURATION_COL] > 0)]
    return df


def fit_kaplan_meier(df: pd.DataFrame):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[DURATION_COL], event_observed=df[EVENT_COL], label="all members")
    return kmf


def plot_km_overall(kmf: KaplanMeierFitter, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Tenure (days since registration)")
    ax.set_ylabel("Survival probability (still subscribed)")
    ax.set_title("Kaplan-Meier survival curve — all sampled members")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_km_by_auto_renew(df: pd.DataFrame, out_path) -> None:
    """Stratified KM curves are where survival analysis earns its keep over
    a classifier: this shows auto-renew status shifts the ENTIRE retention
    curve, not just a single point-in-time probability."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for value, label in [(1, "auto-renew ON"), (0, "auto-renew OFF")]:
        subset = df[df["is_auto_renew"] == value]
        if subset.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(subset[DURATION_COL], subset[EVENT_COL], label=label)
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Tenure (days since registration)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier survival by auto-renew status")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_cox_model(df: pd.DataFrame) -> tuple[CoxPHFitter, CoxPreprocessor]:
    """
    Cox coefficients are "hazard change per one unit of the covariate" —
    directly comparable across covariates only if a "unit" means the same
    thing for each. It doesn't here: total_secs_last_90d ranges into the
    millions (seconds) while completion_ratio ranges over [0, 1], so an
    unscaled fit reports both a real ~30% hazard reduction (completion_ratio)
    AND a similarly-real per-second effect as "coefficient ~0.00" —
    technically correct, but unreadable and easy to misdescribe as "this
    feature doesn't matter." CoxPreprocessor standardizes every continuous
    covariate to mean 0 / std 1, so coefficients become comparable as
    "hazard change per 1 standard deviation," which is what
    cph.print_summary() is actually reporting below. is_auto_renew is left
    alone — standardizing a binary variable doesn't add information, just
    makes it harder to read as "auto-renew on vs off."
    """
    preprocessor = CoxPreprocessor()
    cox_df = preprocessor.fit_transform(df)
    cox_df[DURATION_COL] = df[DURATION_COL].to_numpy()
    cox_df[EVENT_COL] = df[EVENT_COL].to_numpy()

    cph = CoxPHFitter(penalizer=0.1)  # small L2 penalty: several covariates are correlated
    cph.fit(cox_df, duration_col=DURATION_COL, event_col=EVENT_COL)
    return cph, preprocessor


def compare_to_classifier(df: pd.DataFrame, cph: CoxPHFitter, preprocessor: CoxPreprocessor) -> dict:
    """
    Two point-in-time-vs-survival comparisons:
      1. Rank agreement (Spearman correlation) between the Cox model's
         partial hazard (survival risk score) and the XGBoost classifier's
         predicted churn probability, on the same members. High agreement
         would mean the two approaches are mostly redundant; meaningful
         disagreement means they're catching different things.
      2. Cox's concordance index vs. the classifier's ROC-AUC — both are
         "probability a randomly chosen churned member is ranked riskier
         than a randomly chosen retained one," so they're directly
         comparable even though one comes from a hazard model and the
         other from a binary classifier.
    """
    X, y = build_xy(df)
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = build_xgboost_model(scale_pos_weight)
    # No held-out split here — this is a descriptive comparison of the two
    # modeling lenses on the same data, not a claim about clf generalization
    # (churn_model.py already covers that with a proper OOT split).
    clf.set_params(early_stopping_rounds=None)
    clf.fit(X, y, eval_set=[(X, y)], verbose=False)
    clf_prob = clf.predict_proba(X)[:, 1]

    cox_df = preprocessor.transform(df)
    cox_risk = cph.predict_partial_hazard(cox_df)

    rho, _ = spearmanr(clf_prob, cox_risk)
    c_index = concordance_index(df[DURATION_COL], -cox_risk, df[EVENT_COL])
    # In-sample ROC-AUC (not OOT — churn_model.py already covers proper
    # held-out generalization). Comparable to c_index here because both are
    # fit and evaluated on the same rows; the point is the two numbers'
    # relative size, not either one as a deployment estimate.
    in_sample_roc_auc = roc_auc_score(y, clf_prob)

    return {
        "spearman_corr_clf_vs_cox_risk": float(rho),
        "cox_concordance_index": float(c_index),
        "classifier_in_sample_roc_auc": float(in_sample_roc_auc),
    }


def main():
    mlflow_utils.init_mlflow()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_survival_frame()
    print(f"survival analysis sample: {len(df):,} (member, period) rows")

    with mlflow.start_run(run_name="survival_analysis"):
        kmf = fit_kaplan_meier(df)
        plot_km_overall(kmf, config.PLOTS_DIR / "km_curve_overall.png")
        plot_km_by_auto_renew(df, config.PLOTS_DIR / "km_curve_by_auto_renew.png")

        median_survival = kmf.median_survival_time_
        print(f"median survival time (tenure at which 50% have churned): {median_survival} days")
        mlflow.log_metric("median_survival_days", median_survival)

        cph, preprocessor = fit_cox_model(df)
        cph.print_summary()
        cox_summary_path = config.MODELS_DIR / "cox_model_summary.csv"
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cph.summary.to_csv(cox_summary_path)
        mlflow.log_artifact(str(cox_summary_path))
        mlflow.log_metric("cox_concordance_index_train", cph.concordance_index_)

        comparison = compare_to_classifier(df, cph, preprocessor)
        print("\nsurvival vs. classifier comparison:", comparison)
        mlflow.log_metrics(comparison)

        # lifelines fitters don't have a save_model()/load_model() API (that
        # was a scikit-survival/other-library assumption on my part) —
        # they're plain picklable Python objects, so standard pickle is the
        # right tool, same as any other fitted sklearn-shaped estimator.
        # The preprocessor travels with it — see its docstring for why a
        # Cox model is never useful separated from the exact transform it
        # was fit on.
        with open(config.MODELS_DIR / "cox_model.pkl", "wb") as f:
            pickle.dump({"model": cph, "preprocessor": preprocessor}, f)
        print(f"\nsaved plots to {config.PLOTS_DIR}, Cox model to {config.MODELS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
