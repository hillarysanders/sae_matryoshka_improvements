# mlflow_utils.py
from __future__ import annotations

from typing import Dict, Optional

from config import Config


def maybe_init_mlflow(cfg: Config):
    """Initialize MLflow if configured and installed; otherwise no-op."""
    if not cfg.mlflow_enabled:
        return None
    if not cfg.mlflow_uri:
        return None

    try:
        import mlflow  # type: ignore
    except Exception:
        print("[mlflow] mlflow not installed; continuing without MLflow.")
        return None

    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)
    run = mlflow.start_run(run_name=cfg.run_name)
    # log params (best-effort)
    try:
        mlflow.log_params({k: v for k, v in cfg.__dict__.items() if isinstance(v, (int, float, str, bool))})
    except Exception:
        pass
    return run


def mlflow_log_metrics(cfg: Config, metrics: Dict[str, float], step: int) -> None:
    if not cfg.mlflow_enabled or not cfg.mlflow_uri:
        return
    try:
        import mlflow  # type: ignore
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        return


def mlflow_log_artifact(cfg: Config, path: str) -> None:
    if not cfg.mlflow_enabled or not cfg.mlflow_uri:
        return
    try:
        import mlflow  # type: ignore
        mlflow.log_artifact(path)
    except Exception:
        return
