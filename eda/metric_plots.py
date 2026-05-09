from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_metric_frame(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".jsonl":
        return pd.read_json(input_path, lines=True)
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return _frame_from_json_payload(payload)
    raise ValueError("Formato no soportado. Usar .csv, .json o .jsonl.")


def _frame_from_json_payload(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if not isinstance(payload, dict):
        raise ValueError("El JSON debe ser un objeto o una lista de objetos.")

    rows: list[dict[str, Any]] = []
    for model_name, metrics in payload.items():
        if not isinstance(metrics, dict) or "error" in metrics:
            continue
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    if rows:
        return pd.DataFrame(rows)

    return pd.DataFrame([payload])


def _metric_values(
    df: pd.DataFrame,
    metric_col: str,
    model_name: str | None,
    model_col: str | None,
) -> pd.Series:
    if model_name and model_col:
        if model_col not in df.columns:
            raise ValueError(f"No existe la columna de modelo: {model_col}")
        df = df[df[model_col].astype(str).str.lower() == model_name.lower()]

    if metric_col not in df.columns:
        raise ValueError(f"No existe la columna de métrica: {metric_col}")

    values = pd.to_numeric(df[metric_col], errors="coerce").dropna()
    if values.empty:
        target = f" para {model_name}" if model_name else ""
        raise ValueError(f"No hay valores válidos de {metric_col}{target}.")

    # Si el MAPE viene como proporcion (0.82), lo convertimos a porcentaje (82%).
    if values.abs().max() <= 1.5:
        values = values * 100
    return values


def _plot_kde(ax: Any, values: np.ndarray) -> None:
    if len(values) < 2 or np.isclose(values.std(), 0):
        return

    x_grid = np.linspace(values.min(), values.max(), 200)
    kde = gaussian_kde(values)
    bin_width = (values.max() - values.min()) / max(1, min(10, len(values)))
    y_grid = kde(x_grid) * len(values) * bin_width
    ax.plot(x_grid, y_grid, color="#2f76bd", linewidth=2)


def _draw_text(draw: ImageDraw.ImageDraw, position: tuple[int, int], text: str, fill: str = "black") -> None:
    draw.text(position, text, fill=fill, font=ImageFont.load_default())


def _plot_mape_distribution_with_pillow(
    output_file: Path,
    values: np.ndarray,
    label: str,
    bins: int,
    show: bool,
) -> None:
    width, height = 900, 600
    left, right, top, bottom = 85, 35, 65, 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    if np.isclose(values.min(), values.max()):
        center = float(values[0])
        margin = max(1.0, abs(center) * 0.1)
        hist_range = (center - margin, center + margin)
    else:
        hist_range = None

    counts, edges = np.histogram(values, bins=bins, range=hist_range)
    max_count = max(1, int(counts.max()))

    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    bar_gap = 3
    for idx, count in enumerate(counts):
        x0 = left + int(idx * plot_w / len(counts)) + bar_gap
        x1 = left + int((idx + 1) * plot_w / len(counts)) - bar_gap
        y1 = top + plot_h
        y0 = y1 - int((count / max_count) * plot_h)
        draw.rectangle((x0, y0, x1, y1), fill="#8fb1d1", outline="black")

    if len(values) >= 2 and not np.isclose(values.std(), 0):
        x_grid = np.linspace(edges[0], edges[-1], 200)
        kde = gaussian_kde(values)
        bin_width = edges[1] - edges[0]
        y_grid = kde(x_grid) * len(values) * bin_width
        y_max = max(max_count, float(y_grid.max()))
        points = [
            (
                left + int(((x - edges[0]) / (edges[-1] - edges[0])) * plot_w),
                top + plot_h - int((y / y_max) * plot_h),
            )
            for x, y in zip(x_grid, y_grid)
        ]
        if len(points) > 1:
            draw.line(points, fill="#2f76bd", width=3)

    for tick in range(0, max_count + 1):
        y = top + plot_h - int((tick / max_count) * plot_h)
        draw.line((left - 5, y, left, y), fill="black")
        _draw_text(draw, (left - 35, y - 6), str(tick))

    x_min, x_max = float(edges[0]), float(edges[-1])
    for idx in range(5):
        ratio = idx / 4
        x = left + int(ratio * plot_w)
        value = x_min + ratio * (x_max - x_min)
        draw.line((x, top + plot_h, x, top + plot_h + 5), fill="black")
        _draw_text(draw, (x - 20, top + plot_h + 10), f"{value:.1f}")

    _draw_text(draw, (width // 2 - 165, 25), f"{label} - Test MAPE Distribution")
    _draw_text(draw, (width // 2 - 145, height - 35), "Mean Absolute Percentage Error (%)")
    _draw_text(draw, (15, height // 2 - 10), "Frequency")

    image.save(output_file)
    if show:
        image.show()


def plot_mape_distribution(
    input_path: str | Path,
    output_path: str | Path = "reports/eda/plots/mape_distribution.png",
    model_name: str | None = "random_forest",
    metric_col: str = "mape",
    model_col: str | None = "model",
    bins: int = 10,
    show: bool = True,
) -> Path:
    """
    Genera un histograma de MAPE con curva KDE desde resultados de experimentos.

    El input puede ser CSV/JSONL con varias filas, por ejemplo columnas:
    model,test_mape. Tambien acepta un metrics.json simple, aunque para ver una
    distribucion real se necesitan varias mediciones del mismo modelo.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo de métricas: {input_file}")

    output_file = Path(output_path)
    _ensure_dir(output_file.parent)

    df = _load_metric_frame(input_file)
    values = _metric_values(
        df=df,
        metric_col=metric_col,
        model_name=model_name,
        model_col=model_col,
    )
    values_np = values.to_numpy(dtype=float)

    label = model_name.replace("_", " ").title() if model_name else "Modelo"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        values_np,
        bins=bins,
        color="#8fb1d1",
        edgecolor="black",
        alpha=0.85,
    )
    _plot_kde(ax, values_np)
    ax.set_title(f"{label} - Test MAPE Distribution")
    ax.set_xlabel("Mean Absolute Percentage Error (%)")
    ax.set_ylabel("Frequency")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.9)
    try:
        fig.savefig(output_file, dpi=150)
        if show:
            plt.show()
    except RecursionError:
        plt.close(fig)
        _plot_mape_distribution_with_pillow(
            output_file=output_file,
            values=values_np,
            label=label,
            bins=bins,
            show=show,
        )
    else:
        if not show:
            plt.close(fig)

    return output_file
