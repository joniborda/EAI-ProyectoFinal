#!/usr/bin/env python3
"""
Importancia de variables para modelos entrenados con ventanas (compare-models / training-dag).

Soporta: random_forest, xgboost, catboost, linear_regression, ridge.
Las ventanas aplanadas se agregan por nombre de feature (suma sobre los 28 pasos).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODELS_DIR = Path("reports/eda/models")
DEFAULT_METADATA = DEFAULT_MODELS_DIR / "metadata.json"
DEFAULT_BEST_MODEL = DEFAULT_MODELS_DIR / "best_model.json"

IMPORTANCE_MODELS = frozenset(
    {"random_forest", "xgboost", "catboost", "linear_regression", "ridge"}
)


def _load_metadata(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No existe metadata: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_model_name(models_dir: Path, model_name: str | None) -> str:
    if model_name:
        return model_name
    best_path = models_dir / "best_model.json"
    if not best_path.is_file():
        raise FileNotFoundError(
            "Indicá --model-name o ejecutá training-dag para generar best_model.json."
        )
    with best_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    name = payload.get("model_name")
    if not name:
        raise ValueError(f"best_model.json sin model_name: {best_path}")
    return str(name)


def _flattened_feature_names(feature_columns: list[str], window_size: int) -> list[str]:
    """Nombres alineados con ``x_window.reshape(n, -1)`` (paso temporal × feature)."""
    names: list[str] = []
    for step in range(window_size):
        offset = window_size - 1 - step
        for feat in feature_columns:
            names.append(f"{feat}@t-{offset}")
    return names


def _load_trained_model(models_dir: Path, model_name: str):
    if model_name not in IMPORTANCE_MODELS:
        supported = ", ".join(sorted(IMPORTANCE_MODELS))
        raise ValueError(
            f"El modelo '{model_name}' no expone importancia nativa en este script. "
            f"Usá uno de: {supported}."
        )

    if model_name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("XGBoost no está instalado.") from exc
        path = models_dir / "xgboost.json"
        if not path.is_file():
            raise FileNotFoundError(f"No existe {path}")
        model = xgb.XGBRegressor()
        model.load_model(path)
        return model

    path = models_dir / f"{model_name}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}")
    return joblib.load(path)


def _raw_importances(model, model_name: str) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_, dtype=float).ravel())
    if model_name == "catboost" and hasattr(model, "get_feature_importance"):
        return np.asarray(model.get_feature_importance(), dtype=float)
    raise ValueError(f"No se pudo extraer importancia de {model_name}.")


def _aggregate_by_feature(
    flat_names: list[str],
    importances: np.ndarray,
) -> pd.DataFrame:
    if len(flat_names) != len(importances):
        raise ValueError(
            f"Cantidad de features ({len(flat_names)}) != importancias ({len(importances)})."
        )
    totals: dict[str, float] = {}
    for name, value in zip(flat_names, importances):
        base = name.split("@", 1)[0]
        totals[base] = totals.get(base, 0.0) + float(value)

    total_sum = sum(totals.values()) or 1.0
    rows = [
        {
            "feature": feat,
            "importance": imp,
            "importance_pct": 100.0 * imp / total_sum,
        }
        for feat, imp in totals.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _plot_importance(df: pd.DataFrame, *, title: str, output_path: Path, top_n: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["importance_pct"], color="#4c78a8")
    ax.set_xlabel("Importancia (% del total agregado)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_feature_importance(
    models_dir: Path,
    metadata_path: Path,
    model_name: str | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    meta = _load_metadata(metadata_path)
    feature_columns = list(meta.get("feature_columns") or [])
    window_shape = meta.get("window_shape")
    if not feature_columns or not window_shape or len(window_shape) < 2:
        raise ValueError("metadata.json debe incluir feature_columns y window_shape.")

    window_size = int(window_shape[1])
    resolved_name = _resolve_model_name(models_dir, model_name)
    model = _load_trained_model(models_dir, resolved_name)

    flat_names = _flattened_feature_names(feature_columns, window_size)
    raw = _raw_importances(model, resolved_name)

    flat_df = pd.DataFrame(
        {"feature_flat": flat_names, "importance": raw}
    ).sort_values("importance", ascending=False)

    agg_df = _aggregate_by_feature(flat_names, raw)
    return resolved_name, agg_df, flat_df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Importancia de variables de un modelo entrenado (ventanas aplanadas).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directorio con modelos y metadata (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="metadata.json (default: <models-dir>/metadata.json)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="random_forest | xgboost | catboost | ridge | linear_regression "
        "(default: best_model.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Cantidad de features a mostrar en consola y gráfico",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="PNG de barras horizontales (opcional)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="CSV con importancia agregada por feature (opcional)",
    )
    args = parser.parse_args(argv)

    models_dir = args.models_dir
    metadata_path = args.metadata_path or (models_dir / "metadata.json")

    try:
        model_name, agg_df, _flat_df = compute_feature_importance(
            models_dir=models_dir,
            metadata_path=metadata_path,
            model_name=args.model_name,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    top_n = max(1, int(args.top))
    print(f"Modelo: {model_name}")
    print(f"Metadata: {metadata_path}")
    print(f"Features agregadas (top {top_n}):")
    print(
        agg_df.head(top_n)
        .assign(importance_pct=lambda d: d["importance_pct"].map(lambda x: f"{x:.2f}"))
        .to_string(index=False)
    )

    if args.csv_path is not None:
        args.csv_path.parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_csv(args.csv_path, index=False)
        print(f"\nCSV: {args.csv_path}")

    if args.output_path is not None:
        _plot_importance(
            agg_df,
            title=f"Importancia de variables — {model_name.replace('_', ' ').title()}",
            output_path=args.output_path,
            top_n=top_n,
        )
        print(f"Gráfico: {args.output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
