from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional

# Optional import (survive if mlflow is not installed)
try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    # Streamlit secrets first, then env
    try:
        import streamlit as st  # only used if available
        val = st.secrets.get(key)  # type: ignore[attr-defined]
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


def is_enabled() -> bool:
    """True if mlflow is importable AND a tracking URI is configured."""
    if mlflow is None:
        return False
    uri = _cfg("MLFLOW_TRACKING_URI")
    return bool(uri)


def _init_mlflow() -> None:
    """Configure MLflow once per process."""
    if not is_enabled():
        return
    uri = _cfg("MLFLOW_TRACKING_URI")
    exp = _cfg("MLFLOW_EXPERIMENT", "default")
    mlflow.set_tracking_uri(uri)                 # type: ignore
    try:
        mlflow.set_experiment(exp)               # type: ignore
    except Exception:
        # Fallback to default experiment if creation fails
        mlflow.set_experiment("default")         # type: ignore


@contextmanager
def start_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Context manager. If MLflow disabled, yields a dummy object.
    Usage:
        with start_run("summary-eval", tags={"mode":"auto"}) as run:
            log_params({"k": 10})
            log_metrics({"rouge_l": 0.21})
    """
    if not is_enabled():
        yield None
        return
    _init_mlflow()
    active = mlflow.start_run(run_name=run_name, tags=tags)  # type: ignore
    try:
        yield active
    finally:
        mlflow.end_run()  # type: ignore


def log_params(params: Dict[str, Any]) -> None:
    if not is_enabled():
        return
    mlflow.log_params({k: str(v) for k, v in params.items()})  # type: ignore


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if not is_enabled():
        return
    mlflow.log_metrics(metrics, step=step)  # type: ignore


def log_artifact_file(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a single file (e.g., a generated summary.txt, CSV, sqlite backup).
    """
    if not is_enabled():
        return
    mlflow.log_artifact(local_path, artifact_path=artifact_path)  # type: ignore
