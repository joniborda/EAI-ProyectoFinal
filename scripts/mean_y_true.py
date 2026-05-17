#!/usr/bin/env python3
"""Calcula la media de ``y_true`` desde un JSONL de predicciones (p. ej. TFT)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("reports/eda/models/tft.prediction.example.jsonl")


def mean_y_true(input_path: Path) -> tuple[float, int]:
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    df = pd.read_json(input_path, lines=True)
    if "y_true" not in df.columns:
        raise ValueError(f"No existe la columna 'y_true' en {input_path}")

    values = pd.to_numeric(df["y_true"], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No hay valores válidos de y_true en {input_path}")

    return float(values.mean()), int(len(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Media de y_true en un JSONL con predicciones por fila.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"JSONL de entrada (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args(argv)

    try:
        mean, n = mean_y_true(args.input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Archivo: {args.input_path}")
    print(f"Observaciones: {n}")
    print(f"Media y_true: {mean:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
