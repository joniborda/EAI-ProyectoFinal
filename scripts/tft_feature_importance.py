#!/usr/bin/env python3
"""
Interpretabilidad del Temporal Fusion Transformer (pytorch-forecasting).

Carga el checkpoint entrenado, predice sobre el tramo de validación y usa
``interpret_output`` / ``plot_interpretation`` para pesos de variables
(encoder, decoder, estáticas) y atención temporal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

DEFAULT_MODELS_DIR = Path("reports/eda/models")
DEFAULT_FEATURES = Path("reports/eda/features/features.jsonl")
DEFAULT_METADATA = DEFAULT_MODELS_DIR / "metadata.json"
DEFAULT_OUTPUT_DIR = Path("reports/eda/plots/tft_interpretation")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _hp_int(hyperparameters: dict, key: str, default: int) -> int:
    value = hyperparameters.get(key, default)
    return default if value is None else int(value)


def resolve_ckpt_path(models_dir: Path, ckpt_path: Path | None) -> Path:
    if ckpt_path is not None:
        path = Path(ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el checkpoint: {path}")
        return path

    best_meta = models_dir / "best_model.json"
    if best_meta.is_file():
        payload = _load_json(best_meta)
        if payload.get("model_name") == "temporal_fusion_transformer":
            promoted = models_dir / payload.get("promoted_model_filename", "best_model.ckpt")
            if promoted.is_file():
                return promoted

    for candidate in (
        models_dir / "temporal_fusion_transformer.ckpt",
        models_dir / "best_model.ckpt",
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No se encontró checkpoint TFT en {models_dir}. "
        "Indicá --ckpt-path o ejecutá compare-models / training-dag."
    )


def build_tft_validation_loader(
    features_path: Path,
    target_col: str,
    val_ratio: float,
    max_encoder_length: int,
    batch_size: int,
    *,
    adspend_scale: dict[str, float] | None = None,
    ckpt_path: Path | None = None,
):
    from eda.train import (
        _apply_tft_adspend_scale,
        _load_tft_adspend_scale,
        _prepare_tft_dataframe,
        _scale_tft_adspend_for_training,
        _tft_dataset_common_params,
        tft_adspend_scale_path_for_ckpt,
    )
    from pytorch_forecasting import TimeSeriesDataSet

    df = pd.read_json(features_path, lines=True, dtype=False)
    data = _prepare_tft_dataframe(df, target_col=target_col, val_ratio=val_ratio)

    split_idx = int(len(data) * (1 - val_ratio))
    if split_idx <= max_encoder_length:
        raise ValueError(
            f"Pocos datos para TFT (split_idx={split_idx}, max_encoder_length={max_encoder_length})."
        )
    training_cutoff = split_idx - 1

    scale = adspend_scale
    if scale is None and ckpt_path is not None:
        scale = _load_tft_adspend_scale(tft_adspend_scale_path_for_ckpt(ckpt_path))
    if scale is None:
        data, _ = _scale_tft_adspend_for_training(data, training_cutoff)
    else:
        data = _apply_tft_adspend_scale(data, scale)

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        **_tft_dataset_common_params(target_col, max_encoder_length),
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True,
    )
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    return validation, val_loader


def _importance_table(model, interpretation: dict, group: str) -> pd.DataFrame:
    key = f"{group}_variables"
    if key not in interpretation:
        return pd.DataFrame(columns=["group", "feature", "importance", "importance_pct"])

    names = list(getattr(model, f"{group}_variables"))
    values = interpretation[key].detach().cpu().numpy().astype(float).reshape(-1)
    n = min(len(names), len(values))
    if n == 0:
        return pd.DataFrame(columns=["group", "feature", "importance", "importance_pct"])

    names = names[:n]
    values = values[:n]
    total = float(values.sum()) or 1.0
    return pd.DataFrame(
        {
            "group": group,
            "feature": names,
            "importance": values,
            "importance_pct": 100.0 * values / total,
        }
    ).sort_values("importance", ascending=False)


def compute_tft_interpretation(
    *,
    ckpt_path: Path,
    features_path: Path,
    target_col: str,
    val_ratio: float,
    hyperparameters: dict,
    batch_size: int | None = None,
) -> tuple[object, dict, pd.DataFrame]:
    try:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Faltan torch / pytorch-forecasting. Instalá requirements.txt."
        ) from exc

    max_encoder_length = _hp_int(hyperparameters, "tft_max_encoder_length", 28)
    bs = batch_size or max(1, _hp_int(hyperparameters, "tft_batch_size", 32))

    _, val_loader = build_tft_validation_loader(
        features_path=features_path,
        target_col=target_col,
        val_ratio=val_ratio,
        max_encoder_length=max_encoder_length,
        batch_size=bs,
        ckpt_path=ckpt_path,
    )

    model = TemporalFusionTransformer.load_from_checkpoint(
        str(ckpt_path),
        map_location=torch.device("cpu"),
    )
    model.eval()

    raw_predictions = model.predict(val_loader, mode="raw", return_x=True)
    interpretation = model.interpret_output(raw_predictions.output, reduction="sum")

    tables = [
        _importance_table(model, interpretation, "static"),
        _importance_table(model, interpretation, "encoder"),
        _importance_table(model, interpretation, "decoder"),
    ]
    importance_df = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    return model, interpretation, importance_df


def save_interpretation_plots(model, interpretation: dict, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figs = model.plot_interpretation(interpretation)
    saved: list[Path] = []
    for name, fig in figs.items():
        out = output_dir / f"tft_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        saved.append(out)
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Importancia / interpretación de variables del TFT entrenado.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directorio de modelos (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Checkpoint .ckpt (default: best_model.ckpt o temporal_fusion_transformer.ckpt)",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES,
        help=f"features.jsonl (default: {DEFAULT_FEATURES})",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="metadata.json con val_ratio e hiperparámetros TFT",
    )
    parser.add_argument(
        "--target-col",
        default="orders",
        help="Columna objetivo (default: orders)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"PNG de interpretación (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="CSV con importancias agregadas (opcional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size para validación (default: metadata hp o 32)",
    )
    args = parser.parse_args(argv)

    metadata_path = args.metadata_path or (args.models_dir / "metadata.json")
    if not metadata_path.is_file():
        print(f"No existe metadata: {metadata_path}", file=sys.stderr)
        return 1
    if not args.features_path.is_file():
        print(f"No existe features: {args.features_path}", file=sys.stderr)
        return 1

    meta = _load_json(metadata_path)
    val_ratio = float(meta.get("val_ratio", 0.2))
    hyperparameters = dict(meta.get("hyperparameters") or {})

    try:
        ckpt = resolve_ckpt_path(args.models_dir, args.ckpt_path)
        model, _interpretation, importance_df = compute_tft_interpretation(
            ckpt_path=ckpt,
            features_path=args.features_path,
            target_col=args.target_col,
            val_ratio=val_ratio,
            hyperparameters=hyperparameters,
            batch_size=args.batch_size,
        )
        saved_plots = save_interpretation_plots(model, _interpretation, args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Checkpoint: {ckpt}")
    print(f"Features: {args.features_path}")
    print(f"val_ratio: {val_ratio}")
    print("\nImportancia de variables (validación, reduction=sum):")
    if importance_df.empty:
        print("(sin tabla de variables)")
    else:
        display = importance_df.copy()
        display["importance_pct"] = display["importance_pct"].map(lambda x: f"{x:.2f}")
        print(display.to_string(index=False))

    for path in saved_plots:
        print(f"Gráfico: {path}")

    csv_out = args.csv_path or (args.output_dir / "tft_variable_importance.csv")
    importance_df.to_csv(csv_out, index=False)
    print(f"CSV: {csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
