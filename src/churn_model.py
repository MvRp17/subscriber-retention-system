"""
Churn classification: logistic regression baseline -> XGBoost -> calibration.

Run as a script (after feature_engineering.py has produced both parquet
feature tables):
    python -m src.churn_model

Design notes:

1. Time-based split, not random. `period1` (cutoff 2017-01-31) is used for
   fitting; `period2` (cutoff 2017-02-28) is held out entirely as an
   out-of-time (OOT) test set and never touched during training or
   threshold selection. This mirrors deployment: you fit on the past and
   score the future, and members' behavior/features genuinely drift month
   to month. A `run_naive_random_split_diagnostic()` below quantifies
   exactly how much a random split would have overstated performance
   relative to this OOT test — see README "Why time-based splitting
   matters" for the numbers.

2. Logistic regression and XGBoost get *different* preprocessing on
   purpose, not out of inconsistency: logreg needs imputed+scaled inputs
   (it has no native way to handle NaNs or unscaled magnitudes), while
   XGBoost handles missing values and different feature scales natively.
   Feeding XGBoost the same median-imputed, scaled matrix as logreg would
   throw away one of its actual advantages.

3. Class imbalance (~5-6% churn) is handled via class weighting
   (`class_weight="balanced"` / `scale_pos_weight`) rather than
   resampling (SMOTE etc.) — simpler, no synthetic data, and reweighting
   the loss is enough here given the feature signal is fairly strong.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config, mlflow_utils

CATEGORICAL_FEATURES = ["city", "registered_via", "last_payment_method_id"]
NUMERIC_FEATURES = [
    "age",
    "age_is_valid",
    "gender_male",
    "tenure_days",
    "last_payment_plan_days",
    "last_plan_list_price",
    "last_actual_amount_paid",
    "is_auto_renew",
    "days_since_last_transaction",
    "membership_expires_in_days",
    "n_transactions_last_7d",
    "n_transactions_last_30d",
    "n_transactions_last_90d",
    "n_transactions_last_180d",
    "n_distinct_plans_last_180d",
    "n_cancellations_lifetime",
    "n_transactions_lifetime",
    "avg_discount_lifetime",
    "total_secs_last_7d",
    "num_100_last_7d",
    "num_unq_last_7d",
    "active_days_last_7d",
    "total_secs_last_30d",
    "num_100_last_30d",
    "num_unq_last_30d",
    "active_days_last_30d",
    "total_secs_last_90d",
    "num_100_last_90d",
    "num_unq_last_90d",
    "active_days_last_90d",
    "total_secs_last_180d",
    "num_100_last_180d",
    "num_unq_last_180d",
    "active_days_last_180d",
    "days_since_last_log",
    "num_25_lifetime",
    "num_50_lifetime",
    "num_75_lifetime",
    "num_985_lifetime",
    "num_100_lifetime",
    "completion_ratio",
    "engagement_momentum",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_churn"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_period(period: str) -> pd.DataFrame:
    """
    Reads the parquet feature table written by feature_engineering.py. The
    period-sampled feature tables are small (tens of thousands of rows,
    ~40 columns) once aggregated, so pandas is the right tool from here on
    — Spark's job was to reduce 30GB of raw logs down to this table, not
    to do the model training too.
    """
    path = config.FEATURES_DIR / f"{period}.parquet"
    return pd.read_parquet(path)


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[ALL_FEATURES].copy()
    # Defensive: a boolean Spark column with any nulls round-trips through
    # parquet into pandas as dtype=object (True/False/None), which XGBoost
    # rejects outright ("DataFrame.dtypes must be int, float, bool or
    # category"). feature_engineering.py avoids producing nullable booleans
    # in the first place, but this keeps build_xy() correct even against an
    # older feature table or a column added later without that care taken.
    object_cols = X.columns[X.dtypes == "object"]
    if len(object_cols):
        X[object_cols] = X[object_cols].astype(float)
    return X, df[TARGET].astype(int)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


def build_logreg_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=config.RANDOM_SEED
                ),
            ),
        ]
    )


def build_xgboost_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    # Categorical columns are left as raw integer codes (not one-hot) —
    # XGBoost's tree splits handle this fine for our modest cardinality
    # (city ~22 levels, payment_method_id ~40), and it keeps NaNs as
    # genuine missingness instead of imputing them away.
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "f1_at_threshold": f1_score(y_true, y_pred),
        "threshold": threshold,
    }


def best_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Pick the classification threshold on validation data (never on the
    OOT test set) that maximizes F1 — a reasonable default when there's no
    business-specified cost matrix yet."""
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def plot_calibration_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_path) -> None:
    """`curves` maps a label -> (mean_predicted_prob, fraction_of_positives)
    as returned by sklearn's calibration_curve."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for label, (prob_pred, prob_true) in curves.items():
        ax.plot(prob_pred, prob_true, marker="o", label=label)
    ax.set_xlabel("Mean predicted churn probability (bin)")
    ax.set_ylabel("Observed churn rate (bin)")
    ax.set_title("Calibration — out-of-time test set (period2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Diagnostic: why time-based splitting matters
# ---------------------------------------------------------------------


def run_naive_random_split_diagnostic(
    period1: pd.DataFrame, period2: pd.DataFrame, train_size: int
) -> dict:
    """
    Pools both periods and does a plain random 80/20 split, then trains the
    *same model architecture* (XGBoost) used for the genuine OOT number —
    comparing a logistic regression here against the XGBoost OOT result
    would confound "does the split strategy matter" with "which model is
    better," which is a different question. The training set is also
    subsampled down to `train_size` (period1's actual train-split size) so
    the comparison isolates split strategy, not "the naive split just saw
    more data."

    What a naive random split can get wrong relative to a real deployment:
      - the SAME member can appear in both the pooled-random train and test
        rows (they show up in both train.csv and train_v2.csv) — the model
        can partially fit member-specific behavior rather than learning
        signal that generalizes to members it's never seen at all.
      - it evaluates the model on data from the SAME time window it trained
        on, so it can't reveal whether the model still works after a real
        month-over-month distribution shift — which we know happened here
        (churn rate moved 6.4% -> 9.0% from period1 to period2).
    """
    pooled = pd.concat([period1, period2], ignore_index=True)
    X, y = build_xy(pooled)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )
    X_train, y_train = X_train.iloc[:train_size], y_train.iloc[:train_size]

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = build_xgboost_model(scale_pos_weight)
    model.set_params(early_stopping_rounds=None)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {"naive_random_split_roc_auc": roc_auc_score(y_test, y_prob)}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    mlflow_utils.init_mlflow()
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    period1 = load_period("period1")
    period2 = load_period("period2")  # out-of-time test — untouched until final eval

    X_full, y_full = build_xy(period1)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=config.RANDOM_SEED
    )
    X_oot, y_oot = build_xy(period2)

    print(f"train={len(X_train):,}  valid={len(X_valid):,}  oot_test={len(X_oot):,}")
    print(f"churn rate — train={y_train.mean():.3%}  valid={y_valid.mean():.3%}  oot={y_oot.mean():.3%}")

    results = {}

    # --- Logistic regression baseline ---
    with mlflow.start_run(run_name="logreg_baseline"):
        logreg = build_logreg_pipeline()
        logreg.fit(X_train, y_train)

        valid_prob = logreg.predict_proba(X_valid)[:, 1]
        threshold = best_f1_threshold(y_valid, valid_prob)
        oot_prob = logreg.predict_proba(X_oot)[:, 1]
        metrics = evaluate(y_oot, oot_prob, threshold)

        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics({f"oot_{k}": v for k, v in metrics.items()})
        mlflow.sklearn.log_model(logreg, "model")

        print("\n[logreg] OOT metrics:", metrics)
        print(classification_report(y_oot, (oot_prob >= threshold).astype(int)))
        results["logreg"] = {"model": logreg, "oot_prob": oot_prob, "metrics": metrics}

    # --- XGBoost ---
    with mlflow.start_run(run_name="xgboost") as xgb_run:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb_model = build_xgboost_model(scale_pos_weight)
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        valid_prob = xgb_model.predict_proba(X_valid)[:, 1]
        threshold = best_f1_threshold(y_valid, valid_prob)
        oot_prob = xgb_model.predict_proba(X_oot)[:, 1]
        metrics = evaluate(y_oot, oot_prob, threshold)

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_params(
            {
                "n_estimators": xgb_model.n_estimators,
                "max_depth": xgb_model.max_depth,
                "learning_rate": xgb_model.learning_rate,
            }
        )
        mlflow.log_metrics({f"oot_{k}": v for k, v in metrics.items()})
        mlflow.xgboost.log_model(xgb_model, "model")

        print("\n[xgboost] OOT metrics:", metrics)
        print(classification_report(y_oot, (oot_prob >= threshold).astype(int)))
        results["xgboost"] = {"model": xgb_model, "oot_prob": oot_prob, "metrics": metrics}

    # --- Post-hoc isotonic calibration of XGBoost, fit on the validation set ---
    with mlflow.start_run(run_name="xgboost_calibrated") as cal_run:
        calibrated = CalibratedClassifierCV(results["xgboost"]["model"], method="isotonic", cv="prefit")
        calibrated.fit(X_valid, y_valid)

        # Isotonic calibration rescales probabilities toward their true
        # observed frequencies — it does NOT preserve the raw model's
        # threshold. Reusing `threshold` (picked for the raw XGBoost's
        # probability scale) here would silently classify almost nothing as
        # churn, since calibrated probabilities are far less extreme than
        # the raw ones (that's the whole point of calibrating). Threshold
        # has to be re-picked on this model's own validation predictions.
        valid_prob_cal = calibrated.predict_proba(X_valid)[:, 1]
        threshold_cal = best_f1_threshold(y_valid, valid_prob_cal)
        oot_prob_cal = calibrated.predict_proba(X_oot)[:, 1]
        metrics_cal = evaluate(y_oot, oot_prob_cal, threshold_cal)

        mlflow.log_param("model_type", "xgboost_isotonic_calibrated")
        mlflow.log_metrics({f"oot_{k}": v for k, v in metrics_cal.items()})
        mlflow.sklearn.log_model(calibrated, "model")

        print("\n[xgboost + isotonic calibration] OOT metrics:", metrics_cal)
        results["xgboost_calibrated"] = {
            "model": calibrated,
            "oot_prob": oot_prob_cal,
            "metrics": metrics_cal,
        }

    # --- Calibration curves: raw logreg vs raw xgboost vs calibrated xgboost ---
    curves = {}
    for name in ["logreg", "xgboost", "xgboost_calibrated"]:
        prob_true, prob_pred = calibration_curve(
            y_oot, results[name]["oot_prob"], n_bins=10, strategy="quantile"
        )
        curves[name] = (prob_pred, prob_true)
    plot_calibration_curves(curves, config.PLOTS_DIR / "calibration_curve.png")
    print(f"\nsaved calibration_curve.png -> {config.PLOTS_DIR}")

    # --- Diagnostic: naive random split vs. genuine time-based OOT split ---
    naive = run_naive_random_split_diagnostic(period1, period2, train_size=len(X_train))
    print(f"\nnaive pooled-random-split ROC-AUC:  {naive['naive_random_split_roc_auc']:.4f}")
    print(f"genuine time-based OOT ROC-AUC:     {results['xgboost']['metrics']['roc_auc']:.4f}")
    with mlflow.start_run(run_name="time_split_diagnostic"):
        mlflow.log_metrics(naive)
        mlflow.log_metric("oot_roc_auc_for_comparison", results["xgboost"]["metrics"]["roc_auc"])

    # --- Register the best model (by OOT PR-AUC, the imbalance-appropriate metric) ---
    # Note: this selects by ranking quality, not calibration quality — the
    # right call for this repo, since every downstream consumer (LTV
    # segmentation, uplift targeting) only needs a *rank* of churn risk, not
    # a literal probability. A consumer computing a dollar figure directly
    # as P(churn) x revenue would want to champion the calibrated model
    # instead, even at a small PR-AUC cost — see the calibration curve and
    # the isotonic run's log_loss/Brier improvement above.
    best_name = max(results, key=lambda k: results[k]["metrics"]["pr_auc"])
    print(f"\nbest model by OOT PR-AUC: {best_name}")
    client = mlflow.MlflowClient()
    model_name = "churn-classifier"
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    # Re-log the winning model standalone so we have a clean run URI to register
    flavor = mlflow.sklearn if best_name != "xgboost" else mlflow.xgboost
    with mlflow.start_run(run_name=f"register_{best_name}"):
        flavor.log_model(results[best_name]["model"], "model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mv = client.create_model_version(model_name, model_uri, mlflow.active_run().info.run_id)
        client.set_registered_model_alias(model_name, "champion", mv.version)
    print(f"registered '{model_name}' v{mv.version} with alias 'champion'")


if __name__ == "__main__":
    sys.exit(main())
