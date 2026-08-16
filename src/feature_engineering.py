"""
PySpark feature engineering: raw transactional/behavioral logs -> one row
per (msno, period) feature table.

Run as a script to build both periods:
    python -m src.feature_engineering

Design notes (the things worth defending in an interview):

1. Everything is computed "as of" a cutoff date per period (see
   config.FEATURE_CUTOFFS). No feature is allowed to see data past its
   period's cutoff — that's what makes period2 a genuine out-of-time
   test set for a model trained on period1, instead of a leaky random
   split. See README "Why time-based splitting matters".

2. Windowed aggregates (7/30/90/180-day) are computed with a single
   `groupBy` per source table using conditional (`F.when`) sums, rather
   than running one filtered pass per window. transactions.csv is ~1.6GB
   and user_logs.csv is ~30GB — scanning either of them four times just to
   get four window sizes would be a self-inflicted wound.

3. All heavy lifting (filtering, joins, aggregation) happens in Spark's
   distributed execution; nothing is `.toPandas()`'d until the final,
   already-aggregated, one-row-per-member feature table.
"""
import sys
from datetime import date

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src import config, data_prep
from src.spark_utils import get_spark

# ---------------------------------------------------------------------
# Members (static demographic features)
# ---------------------------------------------------------------------


def load_members(spark: SparkSession) -> DataFrame:
    return spark.read.csv(str(config.MEMBERS_CSV), header=True, inferSchema=True)


def build_member_features(members_df: DataFrame, cutoff: date) -> DataFrame:
    """
    Clean + derive tenure from the static members table.

    `bd` (self-reported age) is a known-dirty field in this dataset —
    it includes values like -7000 and 1000. We null out anything outside a
    plausible human age range rather than dropping the row (a bad age
    shouldn't cost us every other feature for that member), and add an
    `age_is_valid` flag so a model can learn "missing/bad age" as its own
    signal rather than have it silently imputed away.
    """
    cutoff_lit = F.lit(cutoff.isoformat()).cast("date")
    reg_date = F.to_date(F.col("registration_init_time").cast("string"), "yyyyMMdd")

    return members_df.select(
        "msno",
        F.col("city").cast("int").alias("city"),
        F.when(F.col("bd").between(10, 90), F.col("bd")).alias("age"),
        F.col("bd").between(10, 90).alias("age_is_valid"),
        F.when(F.col("gender") == "male", 1)
        .when(F.col("gender") == "female", 0)
        .alias("gender_male"),
        F.col("registered_via").cast("int").alias("registered_via"),
        F.datediff(cutoff_lit, reg_date).alias("tenure_days"),
    )


# ---------------------------------------------------------------------
# Transactions (billing / plan behavior)
# ---------------------------------------------------------------------


def load_transactions(spark: SparkSession, period: str) -> DataFrame:
    """
    period1 uses transactions.csv alone (it already covers through
    2017-02-28). period2 unions in transactions_v2.csv, which extends
    coverage to 2017-03-31 for a member set that partially overlaps
    transactions.csv — dedup on full row equality handles the overlap.
    See README "Limitations" for the case this doesn't perfectly resolve
    (a genuinely different row recorded for the same member/date in both
    files, which we can't distinguish from a real second transaction).
    """
    v1 = spark.read.csv(str(config.TRANSACTIONS_V1_CSV), header=True, inferSchema=True)
    if period == "period1":
        return v1
    v2 = spark.read.csv(str(config.TRANSACTIONS_V2_CSV), header=True, inferSchema=True)
    return v1.unionByName(v2).dropDuplicates()


def build_transaction_features(tx_df: DataFrame, cutoff: date) -> DataFrame:
    cutoff_lit = F.lit(cutoff.isoformat()).cast("date")
    tx = tx_df.withColumn(
        "transaction_date", F.to_date(F.col("transaction_date").cast("string"), "yyyyMMdd")
    ).withColumn(
        "membership_expire_date",
        F.to_date(F.col("membership_expire_date").cast("string"), "yyyyMMdd"),
    )
    # Only transactions strictly before/at the cutoff are visible to the model.
    tx = tx.filter(F.col("transaction_date") <= cutoff_lit)
    tx = tx.withColumn("days_before_cutoff", F.datediff(cutoff_lit, F.col("transaction_date")))

    # --- Most recent transaction as of cutoff: current plan/payment state ---
    recency_window = Window.partitionBy("msno").orderBy(
        F.col("transaction_date").desc(), F.col("days_before_cutoff").asc()
    )
    latest = (
        tx.withColumn("rn", F.row_number().over(recency_window))
        .filter(F.col("rn") == 1)
        .select(
            "msno",
            F.col("payment_method_id").alias("last_payment_method_id"),
            F.col("payment_plan_days").alias("last_payment_plan_days"),
            F.col("plan_list_price").alias("last_plan_list_price"),
            F.col("actual_amount_paid").alias("last_actual_amount_paid"),
            F.col("is_auto_renew").alias("is_auto_renew"),
            F.col("days_before_cutoff").alias("days_since_last_transaction"),
            F.datediff(F.col("membership_expire_date"), cutoff_lit).alias(
                "membership_expires_in_days"
            ),
        )
    )

    # --- Windowed frequency/behavior aggregates, one pass over the table ---
    def sum_within(days: int, colname: str):
        return F.sum(
            F.when(F.col("days_before_cutoff") <= days, F.col(colname)).otherwise(0)
        )

    def count_within(days: int):
        return F.sum(F.when(F.col("days_before_cutoff") <= days, 1).otherwise(0))

    windowed = tx.withColumn(
        "discount", F.col("plan_list_price") - F.col("actual_amount_paid")
    ).groupBy("msno").agg(
        *[
            count_within(d).alias(f"n_transactions_last_{d}d")
            for d in config.LOOKBACK_WINDOWS_DAYS
        ],
        F.countDistinct(
            F.when(F.col("days_before_cutoff") <= 180, F.col("plan_list_price"))
        ).alias("n_distinct_plans_last_180d"),
        F.sum(F.col("is_cancel").cast("int")).alias("n_cancellations_lifetime"),
        F.count("*").alias("n_transactions_lifetime"),
        F.avg("discount").alias("avg_discount_lifetime"),
    )

    return latest.join(windowed, on="msno", how="inner")


# ---------------------------------------------------------------------
# User logs (listening behavior) — the 30GB table
# ---------------------------------------------------------------------


def load_user_logs(spark: SparkSession, period: str) -> DataFrame:
    """
    period1: user_logs.csv alone (covers through 2017-02-28).
    period2: unions in user_logs_v2.csv, which is a clean incremental
    extension for March 2017 (non-overlapping date range with v1), so no
    dedup is needed here (unlike transactions).
    """
    v1 = spark.read.csv(str(config.USER_LOGS_V1_CSV), header=True, inferSchema=True)
    if period == "period1":
        return v1
    v2 = spark.read.csv(str(config.USER_LOGS_V2_CSV), header=True, inferSchema=True)
    return v1.unionByName(v2)


def build_user_log_features(logs_df: DataFrame, cutoff: date, sampled_msnos: DataFrame) -> DataFrame:
    """
    Filters to the sampled member cohort *before* the date cast/aggregation,
    via a broadcast join — this is the main lever that makes a 25% member
    sample actually cheaper to process than the full population, since
    everything downstream (parsing dates, grouping) now only touches the
    filtered rows for members we kept. Spark must still read every row of
    the ~30GB CSV once (there's no columnar predicate pushdown on CSV), but
    the shuffle/aggregation work shrinks with the sample.
    """
    cutoff_lit = F.lit(cutoff.isoformat()).cast("date")

    logs = logs_df.join(F.broadcast(sampled_msnos), on="msno", how="inner")
    logs = logs.withColumn("log_date", F.to_date(F.col("date").cast("string"), "yyyyMMdd"))
    logs = logs.filter(F.col("log_date") <= cutoff_lit)
    logs = logs.withColumn("days_before_cutoff", F.datediff(cutoff_lit, F.col("log_date")))

    def sum_within(days: int, colname: str):
        return F.sum(
            F.when(F.col("days_before_cutoff") <= days, F.col(colname)).otherwise(0)
        ).alias(f"{colname}_last_{days}d")

    def active_days_within(days: int):
        return F.countDistinct(
            F.when(F.col("days_before_cutoff") <= days, F.col("log_date"))
        ).alias(f"active_days_last_{days}d")

    agg_cols = []
    for d in config.LOOKBACK_WINDOWS_DAYS:
        agg_cols.append(sum_within(d, "total_secs"))
        agg_cols.append(sum_within(d, "num_100"))
        agg_cols.append(sum_within(d, "num_unq"))
        agg_cols.append(active_days_within(d))

    windowed = logs.groupBy("msno").agg(
        *agg_cols,
        F.min("days_before_cutoff").alias("days_since_last_log"),
        F.sum("num_25").alias("num_25_lifetime"),
        F.sum("num_50").alias("num_50_lifetime"),
        F.sum("num_75").alias("num_75_lifetime"),
        F.sum("num_985").alias("num_985_lifetime"),
        F.sum("num_100").alias("num_100_lifetime"),
    )

    # Completion ratio: full plays vs. all plays. A listener who skips
    # constantly (low ratio) behaves differently from one who finishes what
    # they start, independent of raw volume.
    total_plays = (
        F.col("num_25_lifetime")
        + F.col("num_50_lifetime")
        + F.col("num_75_lifetime")
        + F.col("num_985_lifetime")
        + F.col("num_100_lifetime")
    )
    windowed = windowed.withColumn(
        "completion_ratio",
        F.when(total_plays > 0, F.col("num_100_lifetime") / total_plays),
    )

    # Engagement momentum: last 30 days vs. the 30 days before that. A
    # declining listener (ratio < 1) is a different risk profile than a
    # steady or growing one, even at the same 30-day total.
    prior_30d_secs = F.sum(
        F.when(
            (F.col("days_before_cutoff") > 30) & (F.col("days_before_cutoff") <= 60),
            F.col("total_secs"),
        ).otherwise(0)
    ).alias("total_secs_prior_30d")
    momentum = logs.groupBy("msno").agg(prior_30d_secs)
    windowed = windowed.join(momentum, on="msno", how="left")
    windowed = windowed.withColumn(
        "engagement_momentum",
        F.when(
            F.col("total_secs_prior_30d") > 0,
            F.col("total_secs_last_30d") / F.col("total_secs_prior_30d"),
        ),
    ).drop("total_secs_prior_30d")

    return windowed


# ---------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------


def build_feature_table(spark: SparkSession, period: str) -> DataFrame:
    cutoff = date.fromisoformat(config.FEATURE_CUTOFFS[period])

    labels = data_prep.load_sampled_labels(spark, period)
    sampled_msnos = labels.select("msno")

    members = build_member_features(load_members(spark), cutoff)
    transactions = build_transaction_features(load_transactions(spark, period), cutoff)
    logs = build_user_log_features(load_user_logs(spark, period), cutoff, sampled_msnos)

    # Left joins from the label population: every sampled member gets a row
    # even if they have no transactions or logs before the cutoff (a
    # brand-new member, or a lapsed one with no recent activity) — those are
    # real, informative cases (nulls), not something to silently drop.
    features = (
        labels.join(members, on="msno", how="left")
        .join(transactions, on="msno", how="left")
        .join(logs, on="msno", how="left")
        .withColumn("period", F.lit(period))
        .withColumn("cutoff_date", F.lit(cutoff.isoformat()).cast("date"))
    )
    return features


def main():
    spark = get_spark("kkbox-feature-engineering")
    spark.sparkContext.setLogLevel("WARN")

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for period in config.FEATURE_CUTOFFS:
        print(f"\n=== Building features for {period} (cutoff={config.FEATURE_CUTOFFS[period]}) ===")
        features = build_feature_table(spark, period)
        features = features.cache()

        n_rows = features.count()
        n_churn = features.filter(F.col("is_churn") == 1).count()
        print(f"{period}: {n_rows:,} members, churn rate = {n_churn / n_rows:.3%}")

        out_path = str(config.FEATURES_DIR / f"{period}.parquet")
        features.write.mode("overwrite").parquet(out_path)
        print(f"wrote {out_path}")
        features.unpersist()

    spark.stop()


if __name__ == "__main__":
    sys.exit(main())
