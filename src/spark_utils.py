"""
SparkSession construction, shared across the ingestion/feature-engineering
scripts. Isolated here so every entry point builds Spark the same way.
"""
import os

from pyspark.sql import SparkSession

from src import config


def _ensure_java_home() -> None:
    """
    PySpark shells out to a JVM. On macOS with Homebrew, openjdk@17 is
    keg-only (not symlinked onto the system PATH), so we point JAVA_HOME at
    it explicitly instead of requiring the user to edit their shell profile.
    Only sets it if the environment doesn't already have one, so this is a
    no-op on any machine (e.g. CI, Docker) that already provides Java.
    """
    if os.environ.get("JAVA_HOME"):
        return
    homebrew_jdk = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    if os.path.isdir(homebrew_jdk):
        os.environ["JAVA_HOME"] = homebrew_jdk


def get_spark(app_name: str) -> SparkSession:
    """
    Build a local-mode SparkSession sized for an 8-core/16GB laptop.

    Notes on the config choices, since they'd otherwise look arbitrary:
    - spark.sql.shuffle.partitions=16: Spark's default (200) assumes a
      multi-node cluster; on one machine it just creates 200 tiny tasks with
      scheduling overhead dwarfing the actual work. Setting it near the core
      count keeps each partition doing meaningful work.
    - spark.driver.memory=8g: local mode runs the driver and the only
      "executor" in the same JVM, so this is the real memory ceiling for
      the whole job. Left with headroom under the 16GB physical limit.
    - adaptive query execution: lets Spark coalesce small shuffle partitions
      at runtime, which matters once we filter 30GB of logs down to a 25%
      member sample and the partition count from the read no longer fits
      the (now much smaller) filtered data.
    """
    _ensure_java_home()
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
