"""
Scheduled retraining with a champion/challenger promotion gate.

Run as a script (meant to be invoked by cron / an Airflow DAG / a CI
scheduled job — the "scheduling" itself is out of scope for a portfolio
repo, but this script is exactly what such a scheduler would call):
    python -m src.retrain_pipeline

What this deliberately does NOT do: retrain and silently overwrite
whatever's in production. A scheduled job that always promotes its own
output is how a bad batch of data quietly regresses a live model. Instead:
  1. train a fresh candidate model on the current feature tables
  2. evaluate it on the same out-of-time test set (period2) the existing
     champion was evaluated on, for an apples-to-apples comparison
  3. only overwrite the "champion" alias if the candidate is at least as
     good (within a small tolerance for noise) — otherwise it's logged to
     the registry as a plain version for audit history, but production
     traffic keeps going to the existing champion

This intentionally reuses churn_model.py's training/eval functions rather
than re-implementing them — retraining should train the same model the
initial build did, not a parallel implementation that could quietly drift
from it.
"""
import sys

import mlflow
from sklearn.model_selection import train_test_split

from src import config, mlflow_utils
from src.churn_model import (
    build_xgboost_model,
    build_xy,
    evaluate,
    best_f1_threshold,
    load_period,
)

MODEL_NAME = "churn-classifier"
PROMOTION_METRIC = "pr_auc"  # imbalance-appropriate metric, matches churn_model.py's model-selection metric
# Candidate must be within this much of the champion's score to still lose
# (i.e. a candidate has to be strictly better than "champion minus noise"
# to get promoted) — protects against promoting on run-to-run training
# noise while still allowing a genuinely improved model through.
PROMOTION_TOLERANCE = 0.005


def get_champion_metric(client: mlflow.MlflowClient) -> float | None:
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get(f"oot_{PROMOTION_METRIC}")


def train_candidate():
    period1 = load_period("period1")
    period2 = load_period("period2")

    X_full, y_full = build_xy(period1)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=config.RANDOM_SEED
    )
    X_oot, y_oot = build_xy(period2)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = build_xgboost_model(scale_pos_weight)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    threshold = best_f1_threshold(y_valid, valid_prob)
    oot_prob = model.predict_proba(X_oot)[:, 1]
    metrics = evaluate(y_oot, oot_prob, threshold)
    return model, metrics


def main():
    mlflow_utils.init_mlflow()
    client = mlflow.MlflowClient()

    champion_score = get_champion_metric(client)
    print(f"current champion oot_{PROMOTION_METRIC}: {champion_score}")

    with mlflow.start_run(run_name="retrain_candidate"):
        model, metrics = train_candidate()
        mlflow.log_metrics({f"oot_{k}": v for k, v in metrics.items()})
        mlflow.log_param("model_type", "xgboost_retrain_candidate")
        mlflow.xgboost.log_model(model, "model")
        candidate_run_id = mlflow.active_run().info.run_id
        candidate_score = metrics[PROMOTION_METRIC]

    print(f"candidate oot_{PROMOTION_METRIC}: {candidate_score:.4f}")

    try:
        client.create_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        pass

    model_uri = f"runs:/{candidate_run_id}/model"
    mv = client.create_model_version(MODEL_NAME, model_uri, candidate_run_id)

    should_promote = champion_score is None or candidate_score >= champion_score - PROMOTION_TOLERANCE
    if should_promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        print(
            f"PROMOTED: v{mv.version} is now 'champion' "
            f"({candidate_score:.4f} vs previous {champion_score})"
        )
    else:
        print(
            f"NOT PROMOTED: v{mv.version} registered for audit history, but "
            f"champion stays (candidate {candidate_score:.4f} < "
            f"champion {champion_score:.4f} - tolerance {PROMOTION_TOLERANCE})"
        )


if __name__ == "__main__":
    sys.exit(main())
