"""
Central configuration: file paths, feature cutoffs, and sampling parameters.

Kept as one small module (not YAML/Hydra/etc.) on purpose — this project has a
handful of config values used by a handful of scripts. A config framework would
be over-engineering for this scale; a plain Python module is one `grep` away
from every place a value is used.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Round 1 (original competition) raw files
TRAIN_V1_CSV = DATA_DIR / "train.csv"
TRANSACTIONS_V1_CSV = DATA_DIR / "transactions.csv"
USER_LOGS_V1_CSV = DATA_DIR / "user_logs.csv"
MEMBERS_CSV = DATA_DIR / "members_v3.csv"

# Round 2 ("churn_comp_refresh") raw files — incremental extension into March 2017
REFRESH_DIR = DATA_DIR / "data" / "churn_comp_refresh"
TRAIN_V2_CSV = REFRESH_DIR / "train_v2.csv"
TRANSACTIONS_V2_CSV = REFRESH_DIR / "transactions_v2.csv"
USER_LOGS_V2_CSV = REFRESH_DIR / "user_logs_v2.csv"

# --- Time-based snapshots ------------------------------------------------
# The KKBox competition ships two labeled snapshots of the same subscriber
# base, one month apart. We treat these as two genuine points in time rather
# than pooling them into one dataset and random-splitting:
#   period1: features built from data up to 2017-01-31, label = train.csv
#            (is_churn defined off Feb-2017 membership expiry + 30-day
#            renewal window, per WSDMChurnLabeller.scala)
#   period2: features built from data up to 2017-02-28, label = train_v2.csv
#            (same definition, one month later)
# Training on period1 and evaluating out-of-time on period2 mirrors how the
# model would actually be used in production: fit on the past, score the
# future. See README "Why time-based splitting matters".
FEATURE_CUTOFFS = {
    "period1": "2017-01-31",
    "period2": "2017-02-28",
}
LABEL_FILES = {
    "period1": TRAIN_V1_CSV,
    "period2": TRAIN_V2_CSV,
}

# Rolling window lengths (days, relative to each cutoff) used for behavioral
# aggregates in feature_engineering.py.
LOOKBACK_WINDOWS_DAYS = [7, 30, 90, 180]

# --- Sampling ------------------------------------------------------------
# Full user_logs.csv is ~30GB (400M+ rows). On a single laptop (not a
# cluster) we take a stratified sample of members rather than the full
# population, so the Spark job finishes in minutes rather than hours while
# still exercising genuine distributed aggregation over the raw logs. See
# README "Limitations / what I'd do differently at scale" for the
# full-cluster version of this pipeline.
SAMPLE_FRACTION = 0.25
RANDOM_SEED = 42

# --- LTV estimation -------------------------------------------------------
# KKBox's public dataset has revenue (actual_amount_paid) but no cost data,
# so there's no way to derive a true margin from the data itself. This is a
# stand-in for a number that in a real engagement would come from finance
# (content licensing + infra cost allocation per subscriber) — swap it for
# the real figure and nothing else in ltv_estimation.py needs to change.
ASSUMED_GROSS_MARGIN_RATE = 0.35

# Number of points in the time grid used to numerically integrate each
# member's conditional survival curve into an expected-remaining-lifetime
# estimate. 100 is plenty of resolution for a smooth Cox survival curve;
# this is not "more windows = more accurate," it's quadrature resolution.
LTV_INTEGRATION_GRID_POINTS = 100

# --- Spark session sizing -------------------------------------------------
# Tuned for an 8-core / 16GB laptop running Spark in local mode. Leaves
# headroom for the OS and whatever else is running; not meant to be a
# cluster config.
SPARK_DRIVER_MEMORY = "8g"
SPARK_SHUFFLE_PARTITIONS = "16"  # default of 200 is tuned for clusters, not a laptop
