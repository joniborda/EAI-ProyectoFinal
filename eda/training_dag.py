from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from eda.train import _log_to_mlflow, compare_models


MODEL_ARTIFACTS = {
    "baseline_tm7_sw8_blend": "baseline_tm7_sw8_blend.joblib",
    "linear_regression": "linear_regression.joblib",
    "ridge": "ridge.joblib",
    "random_forest": "random_forest.joblib",
    "catboost": "catboost.joblib",
    "sarima": "sarima.joblib",
    "exponential_smoothing": "exponential_smoothing.joblib",
    "neuralprophet": "neuralprophet.joblib",
    "lstm": "lstm.pt",
    "temporal_fusion_transformer": "temporal_fusion_transformer.ckpt",
    "xgboost": "xgboost.json",
}

PREDICT_STRATEGIES = {
    "baseline_tm7_sw8_blend": "future_daily_baseline",
    "linear_regression": "one_step_window",
    "ridge": "one_step_window",
    "random_forest": "one_step_window",
    "catboost": "one_step_window",
    "xgboost": "one_step_window",
}


def _temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _remove_existing(path: Path) -> None:
    if path.exists():
        path.unlink()


def _atomic_copy(source_path: Path, destination_path: Path) -> None:
    tmp_path = _temp_path(destination_path)
    _remove_existing(tmp_path)
    shutil.copy2(source_path, tmp_path)
    tmp_path.replace(destination_path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp_path = _temp_path(path)
    _remove_existing(tmp_path)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def select_best_model(metrics: dict[str, Any], metric_name: str = "mae") -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for model_name, model_metrics in metrics.items():
        if not isinstance(model_metrics, dict) or "error" in model_metrics:
            continue
        metric_value = model_metrics.get(metric_name)
        if not isinstance(metric_value, int | float):
            continue
        if model_name not in MODEL_ARTIFACTS:
            continue
        candidates.append(
            {
                "model_name": model_name,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "metrics": model_metrics,
            }
        )

    if not candidates:
        raise ValueError(f"No hay modelos válidos para seleccionar por métrica '{metric_name}'.")

    return min(candidates, key=lambda item: item["metric_value"])


def promote_best_model(
    output_dir: str | Path,
    best_model: dict[str, Any],
    target_col: str,
) -> dict[str, Any]:
    output_base = Path(output_dir)
    model_name = best_model["model_name"]
    artifact_name = MODEL_ARTIFACTS[model_name]
    source_path = output_base / artifact_name
    if not source_path.exists():
        raise FileNotFoundError(f"No existe el artefacto del modelo ganador: {source_path}")

    suffix = source_path.suffix
    promoted_model_path = output_base / f"best_model{suffix}"
    _atomic_copy(source_path, promoted_model_path)

    deployment = {
        **best_model,
        "target_col": target_col,
        "artifact_name": artifact_name,
        "promoted_model_filename": promoted_model_path.name,
        "source_path": str(source_path),
        "promoted_model_path": str(promoted_model_path),
        "predict_strategy": PREDICT_STRATEGIES.get(model_name, "unsupported"),
    }
    deployment_path = output_base / "best_model.json"
    _atomic_write_json(deployment_path, deployment)

    return {
        "best_model": model_name,
        "best_model_path": promoted_model_path,
        "best_model_metadata": deployment_path,
        "metric_name": best_model["metric_name"],
        "metric_value": best_model["metric_value"],
        "predict_strategy": deployment["predict_strategy"],
    }


def run_training_dag(
    input_path: str | Path = "reports/eda/features/windows.npz",
    output_dir: str | Path = "reports/eda/models",
    series_path: str | Path = "reports/eda/features/features.jsonl",
    target_col: str = "orders",
    val_ratio: float = 0.2,
    random_state: int = 42,
    selection_metric: str = "mae",
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    training_result = compare_models(
        input_path=input_path,
        output_dir=output_dir,
        series_path=series_path,
        target_col=target_col,
        val_ratio=val_ratio,
        random_state=random_state,
        log_mlflow=False,
        hyperparameters=hyperparameters,
    )

    metrics_path = Path(training_result["metrics"])
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with Path(training_result["metadata"]).open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    best_model = select_best_model(metrics, metric_name=selection_metric)
    deployment = promote_best_model(
        output_dir=output_dir,
        best_model=best_model,
        target_col=target_col,
    )

    result = {
        **training_result,
        **deployment,
    }

    try:
        mlflow_run_id = _log_to_mlflow(
            metrics=metrics,
            metadata=metadata,
            output_base=Path(output_dir),
            target_col=target_col,
            val_ratio=val_ratio,
            hyperparameters=hyperparameters,
        )
        if mlflow_run_id is not None:
            result["mlflow_run_id"] = mlflow_run_id
    except Exception as exc:
        result["mlflow_error"] = str(exc)

    return result
