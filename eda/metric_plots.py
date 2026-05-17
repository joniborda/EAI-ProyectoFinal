from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import gaussian_kde

# Escalas compartidas entre modelos (distribución de error al cuadrado).
_SQUARED_ERROR_X_MAX = 9.6e6
_SQUARED_ERROR_Y_MAX = 250.0
# Eje Y: [0, 1] lineal (poca altura visual) y (1, Y_MAX] logarítmico.
_SQUARED_ERROR_Y_LINEAR_TOP = 1.0
_DIST_Y_LINEAR_FRAC = 0.05
_DIST_TICK_LABELSIZE = 20
_DIST_AXIS_LABEL_FONTSIZE = 20
_DIST_X_TICK_COUNT = 6
_DIST_Y_TICK_COUNT = 6
_DIST_HIST_BINS = 60
# Margen en el borde del eje para que no se recorten las etiquetas del último tick.
_DIST_AXIS_PAD_FRAC = 0.05


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
    if metric_col.lower() == "mape" and values.abs().max() <= 1.5:
        values = values * 100
    return values


def _distribution_metric_mode(metric_col: str) -> tuple[str, str | None]:
    """
    Devuelve (modo, columna).

    modo: mape | abs_error | squared_error (métricas derivadas de y_true/y_pred).
    columna: nombre en el dataframe si modo es mape u otra columna cruda.
    """
    key = metric_col.strip().lower()
    if key in ("rmse", "squared_error", "sq_error", "squared_residual"):
        return "squared_error", None
    if key in ("abs_error", "absolute_error", "ae"):
        return "abs_error", None
    return "column", metric_col


def _derived_error_series(
    df: pd.DataFrame,
    mode: str,
    model_name: str | None,
    model_col: str | None,
) -> pd.Series:
    if model_name and model_col:
        if model_col not in df.columns:
            raise ValueError(f"No existe la columna de modelo: {model_col}")
        df = df.loc[df[model_col].astype(str).str.lower() == model_name.lower()].copy()
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(
            "Para abs_error / squared_error / rmse hacen falta columnas 'y_true' y 'y_pred' "
            "(p. ej. reports/eda/models/mape_distribution.jsonl tras compare-models o training-dag)."
        )
    yt = pd.to_numeric(df["y_true"], errors="coerce")
    yp = pd.to_numeric(df["y_pred"], errors="coerce")
    ok = yt.notna() & yp.notna()
    if mode == "abs_error":
        values = (yt[ok] - yp[ok]).abs()
    else:
        values = (yt[ok] - yp[ok]) ** 2
    if values.empty:
        target = f" para {model_name}" if model_name else ""
        raise ValueError(f"No hay valores válidos de error{target}.")
    return values


def _distribution_xlabel(mode: str, *, for_matplotlib: bool) -> str:
    """
    Etiqueta del eje X según métrica.

    Matplotlib: mathtext (\\hat{y}, sqrt, etc.) para evitar tofu con DejaVu por defecto.
    Pillow: Unicode con fuente TrueType del sistema.
    """
    if mode == "abs_error":
        if for_matplotlib:
            return r"$|y - \hat{y}|$ (units of target)"
        return "|y - ŷ| (units of target)"
    if mode == "squared_error":
        if for_matplotlib:
            return r"$(y - \hat{y})^2\ (\sqrt{\mathrm{mean}} = \mathrm{RMSE})$"
        return "(y - ŷ)² (√mean = RMSE)"
    return "Mean Absolute Percentage Error (%)"


def _format_freq_tick(value: float, _pos: int | None) -> str:
    """Etiquetas compactas para conteos / densidad en el eje Y."""
    v = float(value)
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e6:
        return f"{v:.2e}"
    if av >= 1e4:
        return f"{v:.4g}"
    if av >= 100 and abs(v - round(v)) < 0.05 * av:
        return str(int(round(v)))
    if abs(v - round(v)) < 1e-3:
        return str(int(round(v)))
    return f"{v:.3g}"


def _distribution_hist_range(
    values: np.ndarray,
    mode: str,
) -> tuple[float, float] | None:
    if mode == "squared_error":
        return (0.0, _SQUARED_ERROR_X_MAX)
    if np.isclose(values.min(), values.max()):
        center = float(values[0])
        margin = max(1.0, abs(center) * 0.1)
        return (center - margin, center + margin)
    return None


def _dist_y_transform(y: np.ndarray | float, y_top: float) -> np.ndarray:
    """Normaliza frecuencia Y a [0, 1] con tramo lineal corto en [0, LINEAR_TOP]."""
    arr = np.atleast_1d(np.asarray(y, dtype=float))
    arr = np.clip(arr, 0.0, float(y_top))
    out = np.empty_like(arr)
    linear = arr <= _SQUARED_ERROR_Y_LINEAR_TOP
    out[linear] = _DIST_Y_LINEAR_FRAC * (arr[linear] / _SQUARED_ERROR_Y_LINEAR_TOP)
    above = ~linear
    if np.any(above):
        log_denom = np.log(y_top / _SQUARED_ERROR_Y_LINEAR_TOP)
        if log_denom > 0:
            out[above] = _DIST_Y_LINEAR_FRAC + (1.0 - _DIST_Y_LINEAR_FRAC) * (
                np.log(arr[above] / _SQUARED_ERROR_Y_LINEAR_TOP) / log_denom
            )
        else:
            out[above] = _DIST_Y_LINEAR_FRAC
    return out


def _dist_y_inverse(t: np.ndarray | float, y_top: float) -> np.ndarray:
    """Inversa de ``_dist_y_transform`` (para escala function de matplotlib)."""
    arr = np.atleast_1d(np.asarray(t, dtype=float))
    arr = np.clip(arr, 0.0, 1.0)
    out = np.empty_like(arr)
    linear = arr <= _DIST_Y_LINEAR_FRAC
    if _DIST_Y_LINEAR_FRAC > 0:
        out[linear] = _SQUARED_ERROR_Y_LINEAR_TOP * (arr[linear] / _DIST_Y_LINEAR_FRAC)
    above = ~linear
    if np.any(above) and y_top > _SQUARED_ERROR_Y_LINEAR_TOP:
        log_frac = (arr[above] - _DIST_Y_LINEAR_FRAC) / (1.0 - _DIST_Y_LINEAR_FRAC)
        out[above] = _SQUARED_ERROR_Y_LINEAR_TOP * np.exp(
            log_frac * np.log(y_top / _SQUARED_ERROR_Y_LINEAR_TOP)
        )
    return out


def _dist_y_ratio(y_val: float, y_top: float) -> float:
    return float(_dist_y_transform(y_val, y_top))


def _dist_y_forward_mpl(y: np.ndarray) -> np.ndarray:
    return _dist_y_transform(y, _SQUARED_ERROR_Y_MAX)


def _dist_y_inverse_mpl(t: np.ndarray) -> np.ndarray:
    return _dist_y_inverse(t, _SQUARED_ERROR_Y_MAX)


def _fixed_dist_y_ticks(y_top: float) -> np.ndarray:
    n = _DIST_Y_TICK_COUNT
    if n <= 1:
        return np.array([0.0])
    log_ticks = np.logspace(0.0, np.log10(float(y_top)), n - 1)
    return np.concatenate([[0.0], log_ticks])


def _format_dist_x_tick(value: float, _pos: int | None) -> str:
    v = float(value)
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e4 or av < 0.01:
        return f"{v:.2e}"
    return f"{v:.4g}"


def _apply_distribution_axes(
    ax: Any,
    *,
    x_range: tuple[float, float] | None = None,
    y_top: float | None = None,
) -> None:
    if x_range is not None:
        x_ticks = np.linspace(x_range[0], x_range[1], _DIST_X_TICK_COUNT)
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    else:
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=_DIST_X_TICK_COUNT, integer=False, prune=None)
        )

    if y_top is not None:
        y_ticks = _fixed_dist_y_ticks(float(y_top))
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    else:
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=_DIST_Y_TICK_COUNT, integer=False, prune=None)
        )

    ax.yaxis.set_major_formatter(FuncFormatter(_format_freq_tick))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_dist_x_tick))
    ax.tick_params(axis="y", labelsize=_DIST_TICK_LABELSIZE)
    ax.tick_params(axis="x", labelsize=_DIST_TICK_LABELSIZE)


def _plot_kde(
    ax: Any,
    values: np.ndarray,
    *,
    x_range: tuple[float, float] | None = None,
    bins: int = _DIST_HIST_BINS,
) -> None:
    if len(values) < 2 or np.isclose(values.std(), 0):
        return

    if x_range is not None:
        x_min, x_max = x_range
    else:
        x_min, x_max = float(values.min()), float(values.max())
    x_grid = np.linspace(x_min, x_max, 200)
    kde = gaussian_kde(values)
    span = x_max - x_min
    bin_width = span / max(1, bins) if span > 0 else 1.0
    y_grid = kde(x_grid) * len(values) * bin_width
    ax.plot(x_grid, y_grid, color="#2f76bd", linewidth=2)


# Tamaños en px (TrueType): cuerpo/títulos/leyenda; eje X un ~1.5× más grande que el cuerpo.
_TRUE_VS_PRED_AXIS_FONT_BASE = 12
_TRUE_VS_PRED_DATE_LABEL_SCALE = 1.5
_TRUE_VS_PRED_BODY_FONT_PX = max(
    8, int(round(_TRUE_VS_PRED_AXIS_FONT_BASE * _TRUE_VS_PRED_DATE_LABEL_SCALE))
)
_TRUE_VS_PRED_X_FONT_PX = max(
    10, int(round(_TRUE_VS_PRED_BODY_FONT_PX))
)
_TRUE_VS_PRED_TITLE_LINE_SKIP = int(round(18 * _TRUE_VS_PRED_DATE_LABEL_SCALE))


def _pillow_truetype_candidates() -> list[Path]:
    """Rutas típicas con glifos Latin-1 / Unicode (acentos en español)."""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    return [
        Path(windir) / "Fonts" / "segoeui.ttf",
        Path(windir) / "Fonts" / "arial.ttf",
        Path(windir) / "Fonts" / "calibri.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]


def _pillow_truetype(size: int) -> ImageFont.FreeTypeFont | None:
    for path in _pillow_truetype_candidates():
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".ttc":
                return ImageFont.truetype(str(path), size=size, index=0)
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return None


def _pillow_font(size: int | None = None, *, unicode_glyphs: bool = False) -> ImageFont.ImageFont:
    """
    Fuente para Pillow. Con ``unicode_glyphs=True`` intenta TrueType del sistema
    (necesario para acentos: í, á, ñ, …). Si no hay ninguna, cae al bitmap por defecto
    (puede omitir o deformar acentos).
    """
    px = 10 if size is None else int(size)
    if unicode_glyphs:
        ttf = _pillow_truetype(px)
        if ttf is not None:
            return ttf
    if size is None:
        return ImageFont.load_default()
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def _draw_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    fill: str = "black",
    font: ImageFont.ImageFont | None = None,
) -> None:
    draw.text(position, text, fill=fill, font=font or ImageFont.load_default())


def _plot_mape_distribution_with_pillow(
    output_file: Path,
    values: np.ndarray,
    bins: int,
    show: bool,
    *,
    title: str,
    x_axis_label: str,
    y_top: float | None = None,
    x_range: tuple[float, float] | None = None,
    y_dist_scale: bool = False,
) -> None:
    width, height = 920, 600
    left, right, top, bottom = 108, 58, 65, 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    hist_range = x_range
    if hist_range is None:
        hist_range = _distribution_hist_range(values, "column")

    counts, edges = np.histogram(values, bins=bins, range=hist_range)

    kde_peak = 0.0
    y_grid: np.ndarray | None = None
    x_grid_kde: np.ndarray | None = None
    if len(values) >= 2 and not np.isclose(values.std(), 0):
        x_lo, x_hi = float(edges[0]), float(edges[-1])
        x_grid_kde = np.linspace(x_lo, x_hi, 200)
        kde = gaussian_kde(values)
        bin_width = (edges[1] - edges[0]) if len(edges) > 1 else (x_hi - x_lo) / max(1, bins)
        y_grid = kde(x_grid_kde) * len(values) * bin_width
        kde_peak = float(y_grid.max())

    local_y_top = max(float(max(1, int(counts.max()))), kde_peak, 1.0)
    y_top_plot = float(y_top) if y_top is not None else local_y_top

    def _y_to_height(y_val: float) -> int:
        if y_top_plot <= 0:
            return 0
        if y_dist_scale:
            ratio = _dist_y_ratio(y_val, y_top_plot)
        else:
            y_val = max(0.0, float(y_val))
            ratio = min(y_val, y_top_plot) / y_top_plot
        return int(ratio * plot_h)

    draw.line((left, top, left, top + plot_h), fill="black", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

    bar_gap = 3
    for idx, count in enumerate(counts):
        x0 = left + int(idx * plot_w / len(counts)) + bar_gap
        x1 = left + int((idx + 1) * plot_w / len(counts)) - bar_gap
        y1 = top + plot_h
        y0 = y1 - _y_to_height(float(count))
        draw.rectangle((x0, y0, x1, y1), fill="#8fb1d1", outline="black")

    if y_grid is not None and x_grid_kde is not None and len(edges) > 1:
        span_x = edges[-1] - edges[0]
        points = [
            (
                left + int(((x - edges[0]) / span_x) * plot_w),
                top + plot_h - _y_to_height(float(y)),
            )
            for x, y in zip(x_grid_kde, y_grid)
        ]
        if len(points) > 1:
            draw.line(points, fill="#2f76bd", width=3)

    y_tick_vals = (
        _fixed_dist_y_ticks(y_top_plot)
        if y_dist_scale
        else np.linspace(0.0, y_top_plot, num=_DIST_Y_TICK_COUNT)
    )
    font_tick = _pillow_font(_DIST_TICK_LABELSIZE, unicode_glyphs=True)
    label_left = max(6, left - 98)
    for tidx, tv in enumerate(y_tick_vals):
        ypix = top + plot_h - _y_to_height(float(tv))
        draw.line((left - 5, ypix, left, ypix), fill="black")
        lbl = _format_freq_tick(float(tv), None)
        ty = ypix - 8
        if tidx == len(y_tick_vals) - 1:
            ty = min(ty, top + 4)
        _draw_text(draw, (label_left, ty), lbl, font=font_tick)

    x_min, x_max = float(edges[0]), float(edges[-1])
    x_tick_count = _DIST_X_TICK_COUNT
    for idx in range(x_tick_count):
        ratio = idx / (x_tick_count - 1) if x_tick_count > 1 else 0.0
        xpix = left + int(ratio * plot_w)
        value = x_min + ratio * (x_max - x_min)
        draw.line((xpix, top + plot_h, xpix, top + plot_h + 5), fill="black")
        lbl = _format_dist_x_tick(value, None)
        tick_bbox = draw.textbbox((0, 0), lbl, font=font_tick)
        tick_w = tick_bbox[2] - tick_bbox[0]
        tx = xpix - tick_w // 2
        if idx == 0:
            tx = max(left, tx)
        elif idx == x_tick_count - 1:
            tx = min(left + plot_w - tick_w, tx)
        _draw_text(draw, (tx, top + plot_h + 10), lbl, font=font_tick)

    font_title = _pillow_font(_DIST_TICK_LABELSIZE, unicode_glyphs=True)
    font_x = _pillow_font(_DIST_AXIS_LABEL_FONTSIZE, unicode_glyphs=True)
    font_y = _pillow_font(_DIST_AXIS_LABEL_FONTSIZE, unicode_glyphs=True)

    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    _draw_text(
        draw,
        (max(10, width // 2 - (title_bbox[2] - title_bbox[0]) // 2), 25),
        title,
        font=font_title,
    )
    x_bbox = draw.textbbox((0, 0), x_axis_label, font=font_x)
    _draw_text(
        draw,
        (max(10, width // 2 - (x_bbox[2] - x_bbox[0]) // 2), height - 35),
        x_axis_label,
        font=font_x,
    )
    _draw_text(draw, (15, height // 2 - 10), "Frequency", font=font_y)

    image.save(output_file)
    if show:
        image.show()


def plot_mape_distribution(
    input_path: str | Path,
    output_path: str | Path = "reports/eda/plots/mape_distribution.png",
    model_name: str | None = "random_forest",
    metric_col: str = "mape",
    model_col: str | None = "model",
    bins: int = _DIST_HIST_BINS,
    show: bool = True,
) -> Path:
    """
    Histograma con KDE de errores en validación.

    - metric_col=mape (default): columna ``mape`` (proporción o %).
    - metric_col=rmse o squared_error: distribución de (y-ŷ)² por fila; la raíz
      de la media de esos valores es el RMSE agregado (no hay un “RMSE por punto”).
    - metric_col=abs_error: |y-ŷ| por observación (misma escala que el target).

    Requiere ``y_true``/``y_pred`` para las métricas derivadas (p. ej. mape_distribution.jsonl).
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo de métricas: {input_file}")

    output_file = Path(output_path)
    _ensure_dir(output_file.parent)

    df = _load_metric_frame(input_file)
    label = model_name.replace("_", " ").title() if model_name else "Modelo"
    mode, col = _distribution_metric_mode(metric_col)

    if mode == "abs_error":
        values = _derived_error_series(df, "abs_error", model_name, model_col)
        dist_title = f"{label} - Test Absolute Error Distribution"
    elif mode == "squared_error":
        values = _derived_error_series(df, "squared_error", model_name, model_col)
        dist_title = f"{label} - Test Squared Error Distribution"
    else:
        values = _metric_values(
            df=df,
            metric_col=col or metric_col,
            model_name=model_name,
            model_col=model_col,
        )
        dist_title = f"{label} - Test MAPE Distribution"

    xlabel = _distribution_xlabel(mode, for_matplotlib=True)
    xlabel_pillow = _distribution_xlabel(mode, for_matplotlib=False)

    values_np = values.to_numpy(dtype=float)
    use_shared_scales = mode == "squared_error"
    hist_range = _distribution_hist_range(values_np, mode) if use_shared_scales else None
    shared_y_top = _SQUARED_ERROR_Y_MAX if use_shared_scales else None

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        values_np,
        bins=bins,
        range=hist_range,
        color="#8fb1d1",
        edgecolor="black",
        alpha=0.85,
    )
    _plot_kde(ax, values_np, x_range=hist_range, bins=bins)
    ax.set_title(dist_title)
    ax.set_xlabel(xlabel, fontsize=_DIST_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Frequency", labelpad=14, fontsize=_DIST_AXIS_LABEL_FONTSIZE)
    if hist_range is not None:
        x_lo, x_hi = hist_range
        x_pad = (x_hi - x_lo) * _DIST_AXIS_PAD_FRAC
        ax.set_xlim(x_lo, x_hi + x_pad)
    if shared_y_top is not None:
        ax.set_yscale("function", functions=(_dist_y_forward_mpl, _dist_y_inverse_mpl))
        y_pad = shared_y_top * _DIST_AXIS_PAD_FRAC
        ax.set_ylim(bottom=0.0, top=shared_y_top + y_pad)
    _apply_distribution_axes(ax, x_range=hist_range, y_top=shared_y_top)
    bottom = 0.18 if mode in ("squared_error", "abs_error") else 0.14
    right_margin = 0.93 if use_shared_scales else 0.97
    top_margin = 0.88 if use_shared_scales else 0.9
    fig.subplots_adjust(left=0.2, right=right_margin, bottom=bottom, top=top_margin)
    try:
        fig.savefig(output_file, dpi=150, bbox_inches="tight", pad_inches=0.08)
        if show:
            plt.show()
    except RecursionError:
        plt.close(fig)
        _plot_mape_distribution_with_pillow(
            output_file=output_file,
            values=values_np,
            bins=bins,
            show=show,
            title=dist_title,
            x_axis_label=xlabel_pillow,
            y_top=shared_y_top,
            x_range=hist_range,
            y_dist_scale=shared_y_top is not None,
        )
    else:
        if not show:
            plt.close(fig)

    return output_file


def _plot_true_vs_predictions_pillow(
    output_path: Path,
    x: np.ndarray,
    y_t: np.ndarray,
    y_p: np.ndarray,
    title_lines: list[str],
    x_label: str,
) -> None:
    """Gráfico True vs Pred sin matplotlib (evita bugs de savefig en algunos entornos)."""
    font_body = _pillow_font(_TRUE_VS_PRED_BODY_FONT_PX, unicode_glyphs=True)
    font_x = _pillow_font(_TRUE_VS_PRED_X_FONT_PX, unicode_glyphs=True)

    width, height = 1000, 575
    left, right, bottom = 118, 45, 62
    skip = _TRUE_VS_PRED_TITLE_LINE_SKIP
    pad_top = 18
    pad_below_titles = 18
    n_titles = len(title_lines)
    title_block = pad_top + max(0, n_titles) * skip + pad_below_titles
    # Menos líneas de título -> menos margen superior -> más alto el área del gráfico
    top = min(140, max(76, title_block))
    pw, ph = width - left - right, height - top - bottom

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 puntos para dibujar la serie.")

    x_min, x_max = float(x.min()), float(x.max())
    span_x = x_max - x_min if x_max > x_min else 1.0
    y_min = float(min(y_t.min(), y_p.min()))
    y_max = float(max(y_t.max(), y_p.max()))
    span_y = y_max - y_min
    if span_y < 1e-9:
        y_min -= 1.0
        y_max += 1.0
        span_y = y_max - y_min

    def px_x(xv: float) -> int:
        return int(left + (float(xv) - x_min) / span_x * pw)

    def px_y(yv: float) -> int:
        return int(top + ph - (float(yv) - y_min) / span_y * ph)

    # Ejes
    draw.line((left, top + ph, left + pw, top + ph), fill="black", width=2)
    draw.line((left, top, left, top + ph), fill="black", width=2)

    # Grilla horizontal ligera
    for k in range(1, 6):
        yy = y_min + (k / 6) * span_y
        ypix = px_y(yy)
        draw.line((left, ypix, left + pw, ypix), fill="#e0e0e0", width=1)

    pts_t = [(px_x(x[i]), px_y(y_t[i])) for i in range(n)]
    pts_p = [(px_x(x[i]), px_y(y_p[i])) for i in range(n)]
    draw.line(pts_t, fill="#1f77b4", width=2)
    draw.line(pts_p, fill="#d62728", width=2)

    ty = pad_top
    for line in title_lines:
        _draw_text(draw, (22, ty), line, font=font_body)
        ty += skip

    _draw_text(draw, (14, top + ph // 2), "Values", font=font_body)

    # Ticks Y (5)
    y_lbl_w = 88
    for k in range(0, 6):
        yy = y_min + (k / 5) * span_y
        ypix = px_y(yy)
        draw.line((left - 4, ypix, left, ypix), fill="black", width=1)
        _draw_text(draw, (left - y_lbl_w, ypix - 8), f"{yy:.2g}", font=font_body)

    # Ticks X (inicio, fin) y leyenda del eje, pegados bajo el eje para acortar la imagen
    x_num_y = top + ph + 8
    _draw_text(draw, (left - 5, x_num_y), f"{x_min:.4g}", font=font_x)
    _draw_text(draw, (left + pw - 68, x_num_y), f"{x_max:.4g}", font=font_x)
    x_label_y = x_num_y + 26
    _draw_text(draw, (left, x_label_y), x_label, font=font_x)

    # Leyenda simple
    leg_gap = int(round(22 * _TRUE_VS_PRED_DATE_LABEL_SCALE))
    leg_y = top + 10
    draw.line((left + pw - 160, leg_y + 6, left + pw - 140, leg_y + 6), fill="#1f77b4", width=3)
    _draw_text(draw, (left + pw - 135, leg_y - 2), "True", font=font_body)
    draw.line(
        (left + pw - 160, leg_y + leg_gap + 6, left + pw - 140, leg_y + leg_gap + 6),
        fill="#d62728",
        width=3,
    )
    _draw_text(draw, (left + pw - 135, leg_y + leg_gap - 2), "Prediction", font=font_body)

    img.save(output_path)


def plot_true_vs_predictions(
    input_path: str | Path,
    model_name: str,
    output_path: str | Path | None = None,
    *,
    target_col: str = "orders",
    product_id: str | None = None,
    model_col: str = "model",
    show: bool = True,
) -> Path:
    """
    Serie temporal de validación: valores reales vs predichos (desde mape_distribution.jsonl).

    El archivo debe contener filas con al menos: model, y_true, y_pred (y opcionalmente created, product_id).
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_file}")

    df = _load_metric_frame(input_file)
    if model_col not in df.columns:
        raise ValueError(f"No existe la columna '{model_col}' en el input.")
    mask = df[model_col].astype(str).str.lower() == model_name.lower()
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No hay filas para el modelo '{model_name}'.")

    if product_id is not None and "product_id" in df.columns:
        df = df[df["product_id"].astype(str) == str(product_id)]
        if df.empty:
            raise ValueError(f"No hay filas para product_id={product_id!r} en ese modelo.")

    for col in ("y_true", "y_pred"):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}'. Ejecutá compare-models o training-dag para generar mape_distribution.jsonl.")

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["y_true", "y_pred"])
    if df.empty:
        raise ValueError("No quedaron filas con y_true/y_pred numéricos.")

    if "created" in df.columns and df["created"].notna().any():
        df["_x"] = pd.to_datetime(df["created"], errors="coerce")
        df = df.dropna(subset=["_x"]).sort_values("_x")
        x = (df["_x"] - df["_x"].iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        x_label = "Días desde el inicio del tramo"
        df = df.drop(columns=["_x"])
    elif "sample_idx" in df.columns:
        df = df.sort_values("sample_idx")
        x = np.arange(len(df), dtype=float)
        x_label = "Timestamp"
    else:
        df = df.reset_index(drop=True)
        x = np.arange(len(df), dtype=float)
        x_label = "Timestamp"

    y_t = df["y_true"].to_numpy(dtype=float)
    y_p = df["y_pred"].to_numpy(dtype=float)
    mape_pct = float(np.mean(np.abs((y_t - y_p) / np.clip(np.abs(y_t), 1e-8, None))) * 100.0)

    out = Path(output_path) if output_path else Path(f"reports/eda/plots/true_vs_pred_{model_name}.png")
    _ensure_dir(out.parent)

    title_extra = f"Product: {product_id}" if product_id else f"Target: {target_col}"
    title_lines = [
        title_extra,
        f"Model: {model_name} - MAPE: {mape_pct:.2f}%",
    ]
    _plot_true_vs_predictions_pillow(out, x, y_t, y_p, title_lines, x_label)
    if show:
        Image.open(out).show()

    return out
