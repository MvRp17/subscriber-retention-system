"""
Label loading and stratified member sampling.

Kept separate from feature_engineering.py because "which members are in
scope" is a decision made once per period and reused by every downstream
stage (features, models, survival analysis, uplift) — it should not be
silently re-derived (and potentially drift) in multiple places.
"""
from pyspark.sql import DataFrame, SparkSession

from src import config


def load_labels(spark: SparkSession, period: str) -> DataFrame:
    """Load the raw (unsampled) is_churn labels for a period."""
    path = str(config.LABEL_FILES[period])
    return spark.read.csv(path, header=True, inferSchema=True)


def sample_members(labels_df: DataFrame, period: str) -> DataFrame:
    """
    Stratified sample of members, preserving the churned / retained ratio.

    Why stratified rather than a plain random fraction: churn is a minority
    class (~6% base rate in this dataset). A plain random sample keeps the
    same expected ratio in aggregate, but `sampleBy` fixes the ratio exactly
    per stratum, which matters for repeated experimentation — training set
    class balance doesn't jitter run to run just from sampling variance.

    Returns a single-column (msno) DataFrame — the "in scope for this
    period" member list that every other script joins against.
    """
    fractions = {0: config.SAMPLE_FRACTION, 1: config.SAMPLE_FRACTION}
    sampled_labels = labels_df.sampleBy(
        "is_churn", fractions=fractions, seed=config.RANDOM_SEED
    )
    return sampled_labels.select("msno").distinct()


def load_sampled_labels(spark: SparkSession, period: str) -> DataFrame:
    """Convenience: labels for exactly the sampled member set of a period."""
    labels_df = load_labels(spark, period)
    members = sample_members(labels_df, period)
    return labels_df.join(members, on="msno", how="inner")
