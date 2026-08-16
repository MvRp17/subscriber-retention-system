"""
Shared MLflow setup so every training script logs to the same place.

Tracking store: a local SQLite file (mlflow.db) rather than the default
plain-file store. This is a one-line difference, but it's the difference
between "just logging metrics" and getting the actual Model Registry
(versioning, stage transitions like Staging -> Production) — the registry
needs a database-backed backend, and SQLite is the zero-infrastructure way
to get one on a laptop. In production this would point at Postgres/MySQL
instead of a local file; nothing else in the calling code would change.
"""
import mlflow

from src import config

TRACKING_URI = f"sqlite:///{config.PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "subscriber-retention"


def init_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
