"""
Shared pytest fixtures. A single session-scoped local SparkSession is reused
across every test — starting the JVM is the slow part (~seconds), so paying
that cost once per test run (not once per test) keeps `pytest` fast enough
to run on every commit.
"""
import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spark_utils import get_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    spark = get_spark("pytest")
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()
