# Packages the pipeline code + a consistent Python/JVM environment for
# reproducible retraining, drift checks, and scoring.
#
# What this image is NOT for: running feature_engineering.py against the
# full raw dataset as a demo. That's a 30GB local read even on a laptop
# (see README "Limitations") — in production this stage runs on a real
# Spark cluster (EMR/Dataproc/Databricks), not in a single container. This
# image is for everything downstream of the parquet feature tables:
# retraining, drift monitoring, and serving the registered model — plus it
# happens to also be able to run feature_engineering.py correctly if you
# do mount the full dataset in, since the JVM/PySpark environment is here
# too.
FROM python:3.11-slim

# PySpark shells out to a `java` binary on PATH; Debian's openjdk package
# provides that without any JAVA_HOME wrangling (unlike the Homebrew/macOS
# dev setup in spark_utils.py, where openjdk is keg-only and needs a nudge).
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/

# Raw data and generated artifacts are mounted at runtime, not baked into
# the image: the raw CSVs are gigabytes and aren't ours to redistribute
# (see README), and artifacts/mlruns are meant to be produced fresh per
# run, not frozen into a layer.
VOLUME ["/app/data", "/app/artifacts", "/app/mlruns"]

ENV PYTHONUNBUFFERED=1

# There's no single "right" command for a multi-stage pipeline — override
# at `docker run` time, e.g.:
#   docker run -v $(pwd)/artifacts:/app/artifacts <image> python -m src.retrain_pipeline
#   docker run -v $(pwd)/artifacts:/app/artifacts <image> python -m src.drift_monitor
# Default to the test suite, since "does the feature pipeline logic still
# work in this exact environment" is the one thing worth checking with no
# extra arguments.
CMD ["python", "-m", "pytest", "tests/", "-v"]
