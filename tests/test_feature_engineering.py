"""
Unit tests for the feature engineering functions in src/feature_engineering.py.

These use small, hand-built DataFrames rather than the real (30GB) data —
the goal is to pin down the *logic* (no leakage past cutoff, correct window
aggregation, correct handling of dirty/missing values), which a few
carefully chosen rows can prove just as well as the full dataset, in
milliseconds instead of minutes.
"""
from datetime import date

from src import feature_engineering as fe

CUTOFF = date(2017, 1, 31)


def test_member_age_cleaning(spark):
    """
    KKBox's `bd` (age) field contains known-garbage values (negative,
    zero, absurdly high). Anything outside a plausible human range should
    be nulled out and flagged, not fed to a model as a real age.
    """
    members = spark.createDataFrame(
        [
            ("m1", 1, 28, "male", 7, 20150101),  # valid
            ("m2", 1, -7000, "female", 7, 20150101),  # garbage: negative
            ("m3", 1, 0, None, 7, 20150101),  # garbage: zero
            ("m4", 1, 150, "male", 7, 20150101),  # garbage: implausible
        ],
        schema="msno string, city int, bd int, gender string, registered_via int, registration_init_time int",
    )
    result = {r["msno"]: r for r in fe.build_member_features(members, CUTOFF).collect()}

    assert result["m1"]["age"] == 28 and result["m1"]["age_is_valid"] is True
    assert result["m2"]["age"] is None and result["m2"]["age_is_valid"] is False
    assert result["m3"]["age"] is None and result["m3"]["age_is_valid"] is False
    assert result["m4"]["age"] is None and result["m4"]["age_is_valid"] is False

    assert result["m1"]["gender_male"] == 1
    assert result["m2"]["gender_male"] == 0
    assert result["m3"]["gender_male"] is None  # unspecified gender stays null


def test_member_tenure_days(spark):
    members = spark.createDataFrame(
        [("m1", 1, 30, "male", 7, 20160101)],
        schema="msno string, city int, bd int, gender string, registered_via int, registration_init_time int",
    )
    result = fe.build_member_features(members, CUTOFF).collect()[0]
    # 2016-01-01 -> 2017-01-31 is 396 days
    assert result["tenure_days"] == 396


TX_SCHEMA = (
    "msno string, payment_method_id int, payment_plan_days int, plan_list_price int, "
    "actual_amount_paid int, is_auto_renew int, transaction_date int, "
    "membership_expire_date int, is_cancel int"
)


def test_transactions_after_cutoff_are_excluded(spark):
    """The whole point of the cutoff parameter: a transaction dated after
    the cutoff must not influence that period's features at all."""
    tx = spark.createDataFrame(
        [
            ("m1", 41, 30, 149, 149, 1, 20170115, 20170215, 0),  # before cutoff
            ("m1", 41, 30, 149, 149, 1, 20170301, 20170401, 0),  # after cutoff — must be ignored
        ],
        schema=TX_SCHEMA,
    )
    result = fe.build_transaction_features(tx, CUTOFF).collect()[0]
    assert result["n_transactions_lifetime"] == 1
    assert result["last_plan_list_price"] == 149
    assert result["days_since_last_transaction"] == 16  # 2017-01-15 -> 2017-01-31


def test_transactions_most_recent_and_windows(spark):
    tx = spark.createDataFrame(
        [
            ("m1", 1, 30, 100, 100, 0, 20160820, 20160901, 0),  # 164 days before cutoff
            ("m1", 1, 30, 100, 100, 0, 20161215, 20170115, 0),  # 47 days before cutoff
            ("m1", 2, 30, 150, 120, 1, 20170125, 20170225, 0),  # 6 days before cutoff, most recent
        ],
        schema=TX_SCHEMA,
    )
    result = fe.build_transaction_features(tx, CUTOFF).collect()[0]

    # Most recent transaction wins for "current state" fields
    assert result["last_payment_method_id"] == 2
    assert result["is_auto_renew"] == 1
    assert result["last_actual_amount_paid"] == 120

    # Window counts: 1 in last 7d, 2 in last 90d, 3 in last 180d
    assert result["n_transactions_last_7d"] == 1
    assert result["n_transactions_last_90d"] == 2
    assert result["n_transactions_last_180d"] == 3
    assert result["n_distinct_plans_last_180d"] == 2  # list prices 100 and 150

    # avg discount across all 3: (0 + 0 + 30) / 3 = 10
    assert result["avg_discount_lifetime"] == 10.0


LOGS_SCHEMA = (
    "msno string, date int, num_25 int, num_50 int, num_75 int, num_985 int, "
    "num_100 int, num_unq int, total_secs double"
)


def test_user_logs_filters_to_sampled_cohort(spark):
    """A member with real listening history who isn't in the sampled
    cohort must not leak into the feature table — sampling has to happen
    before the aggregation, not after."""
    logs = spark.createDataFrame(
        [
            ("m1", 20170130, 0, 0, 0, 0, 10, 8, 1000.0),
            ("m2", 20170130, 0, 0, 0, 0, 99, 90, 9999.0),  # m2 not in sample
        ],
        schema=LOGS_SCHEMA,
    )
    sampled = spark.createDataFrame([("m1",)], schema="msno string")
    result = fe.build_user_log_features(logs, CUTOFF, sampled).collect()
    assert [r["msno"] for r in result] == ["m1"]


def test_user_logs_completion_ratio_and_momentum(spark):
    logs = spark.createDataFrame(
        [
            # prior 30d window (31-60 days before cutoff): 2016-12-15, 47 days out
            ("m1", 20161215, 0, 0, 0, 0, 10, 10, 500.0),
            # last 30d window: 2017-01-25, 6 days out
            ("m1", 20170125, 10, 0, 0, 0, 10, 15, 1000.0),
        ],
        schema=LOGS_SCHEMA,
    )
    sampled = spark.createDataFrame([("m1",)], schema="msno string")
    result = fe.build_user_log_features(logs, CUTOFF, sampled).collect()[0]

    # completion_ratio = num_100_lifetime / total_plays_lifetime = 20 / 30
    assert abs(result["completion_ratio"] - (20 / 30)) < 1e-9

    # engagement_momentum = last_30d_secs / prior_30d_secs = 1000 / 500
    assert abs(result["engagement_momentum"] - 2.0) < 1e-9

    assert result["active_days_last_30d"] == 1  # only 2017-01-25 falls in last-30d
    assert result["days_since_last_log"] == 6  # 2017-01-25 -> 2017-01-31
