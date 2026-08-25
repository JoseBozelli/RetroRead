"""
Ensures every script logs to the same MLflow tracking store, regardless of environment
variables or MLflow version defaults -- avoids the tracking store silently splitting across
file-based and SQLite backends.
"""

import mlflow

from retroread.config import PROJECT_ROOT

TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"

def configure_tracking() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)