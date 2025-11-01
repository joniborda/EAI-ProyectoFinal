from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typer
import uvicorn

from .db import (
	fetch_sales_timeseries,
	fetch_daily_features,
	get_all_product_ids,
	test_connection,
	fetch_sales_totals_timeseries,
    fetch_global_daily_features,
)
from .forecaster_sarimax import forecast as sarimax_forecast, train_and_save as sarimax_train
from .forecaster_rnn import forecast as rnn_forecast, train_and_save as rnn_train


def _date_key(sd: Optional[date], ed: Optional[date]) -> str:
	s = sd.isoformat() if sd else "all"
	e = ed.isoformat() if ed else "all"
	return f"{s}_{e}"


def _prepared_dir() -> Path:
	p = Path("data") / "prepared"
	p.mkdir(parents=True, exist_ok=True)
	return p


def _prepared_dataset_path(scope: str, target: str, sd: Optional[date], ed: Optional[date]) -> Path:
	key = _date_key(sd, ed)
	safe_scope = scope.replace("/", "-").replace("\\", "-")
	return _prepared_dir() / f"prepared_{safe_scope}_{target}_{key}.npz"


def _align_features_to_y(dates_y: np.ndarray, y: np.ndarray, dates_x: Optional[np.ndarray], X: Optional[np.ndarray]) -> np.ndarray:
	if X is None or X.size == 0 or dates_x is None or dates_x.size == 0:
		return np.zeros((y.shape[0], 0), dtype=float)
	# Map dates_x to row index
	x_map = {int(d.astype("datetime64[D]").astype(int)): i for i, d in enumerate(dates_x)}
	out = np.zeros((dates_y.shape[0], X.shape[1]), dtype=float)
	for i, d in enumerate(dates_y):
		key = int(d.astype("datetime64[D]").astype(int))
		j = x_map.get(key)
		if j is not None:
			out[i] = X[j]
	return out


def _save_prepared_dataset(scope: str, target: str, sd: Optional[date], ed: Optional[date], dates_y: np.ndarray, y: np.ndarray, dates_x: Optional[np.ndarray], X: Optional[np.ndarray]) -> Path:
	X_aligned = _align_features_to_y(dates_y, y, dates_x, X)
	path = _prepared_dataset_path(scope, target, sd, ed)
	np.savez_compressed(path, dates=dates_y.astype("datetime64[D]"), y=y.astype(float), X=X_aligned.astype(float))
	return path


def _load_prepared_dataset(scope: str, target: str, sd: Optional[date], ed: Optional[date]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	path = _prepared_dataset_path(scope, target, sd, ed)
	if not path.exists():
		raise FileNotFoundError(f"Dataset preparado no encontrado: {path}. Ejecutá 'prepare_data' primero.")
	data = np.load(path)
	dates = data["dates"]
	y = data["y"]
	X = data["X"]
	return dates, y, X


def _generate_eda_plots(
	scope: str,
	target: str,
	dates: np.ndarray,
	y: np.ndarray,
	feat_dates: Optional[np.ndarray] = None,
	X: Optional[np.ndarray] = None,
	out_dir: Optional[Path] = None,
) -> None:
	if out_dir is None:
		out_dir = Path("reports") / "eda"
	out_dir.mkdir(parents=True, exist_ok=True)

	idx = pd.to_datetime(dates.astype("datetime64[D]").astype("datetime64[ns]")) if dates.size else None
	df = pd.DataFrame({"y": y}, index=idx) if idx is not None else pd.DataFrame({"y": y})

	feature_names = [
		"orders_count",
		"unique_customers",
		"avg_order_total_price",
		"num_channels",
		"num_sources",
		"avg_num_tags",
	]
	if X is not None and X.size:
		fidx = pd.to_datetime(feat_dates.astype("datetime64[D]").astype("datetime64[ns]")) if feat_dates is not None and feat_dates.size else None
		Xdf = pd.DataFrame(X, columns=feature_names, index=fidx) if fidx is not None else pd.DataFrame(X, columns=feature_names)
		if df.index.size and Xdf.index.size:
			df = df.join(Xdf, how="left").fillna(0.0)
		else:
			for i, name in enumerate(feature_names):
				df[name] = X[:, i]

	# Serie temporal
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.plot(df.index if df.index.size else range(len(df)), df["y"], lw=1.5)
	ax.set_title(f"Serie {scope} - {target}")
	ax.set_ylabel("y")
	fig.tight_layout()
	fig.savefig(out_dir / f"ts_{scope}_{target}.png", dpi=150)
	plt.close(fig)

	# Histograma
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.hist(df["y"].values, bins=30, color="#4e79a7", alpha=0.8)
	ax.set_title("Histograma de y")
	fig.tight_layout()
	fig.savefig(out_dir / f"hist_{scope}_{target}.png", dpi=150)
	plt.close(fig)

	# ACF básica
	try:
		from statsmodels.tsa.stattools import acf as _acf

		nlags = max(10, min(60, len(df) - 1))
		if nlags >= 10:
			acf_vals = _acf(df["y"].values, nlags=nlags, fft=True)
			fig, ax = plt.subplots(figsize=(8, 3))
			ax.stem(range(len(acf_vals)), acf_vals, use_line_collection=True)
			ax.set_title("Autocorrelación (ACF)")
			fig.tight_layout()
			fig.savefig(out_dir / f"acf_{scope}_{target}.png", dpi=150)
			plt.close(fig)
	except Exception:
		pass

	# Correlaciones y dispersión con features, si existen
	if X is not None and X.size:
		corr = df.corr(numeric_only=True)
		fig, ax = plt.subplots(figsize=(6, 5))
		im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
		ax.set_xticks(range(len(corr.columns)))
		ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
		ax.set_yticks(range(len(corr.index)))
		ax.set_yticklabels(corr.index, fontsize=7)
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		ax.set_title("Matriz de correlación")
		fig.tight_layout()
		fig.savefig(out_dir / f"corr_{scope}_{target}.png", dpi=150)
		plt.close(fig)

		for name in [
			"orders_count",
			"unique_customers",
			"avg_order_total_price",
			"num_channels",
			"num_sources",
			"avg_num_tags",
		]:
			if name in df.columns:
				fig, ax = plt.subplots(figsize=(5, 4))
				ax.scatter(df[name].values, df["y"].values, s=8, alpha=0.5)
				ax.set_xlabel(name)
				ax.set_ylabel("y")
				ax.set_title(f"y vs {name}")
				fig.tight_layout()
				fig.savefig(out_dir / f"scatter_{scope}_{target}_{name}.png", dpi=150)
				plt.close(fig)


def handle_test_db() -> None:
	try:
		test_connection()
		typer.secho("Conexión OK", fg=typer.colors.GREEN)
	except Exception as e:
		typer.secho(f"Error de conexión: {e}", fg=typer.colors.RED)
		raise typer.Exit(code=1)

def _do_eda(scope: str, tgt: str, dates: np.ndarray, y: np.ndarray, feat_dates=None, X=None, out_dir: Optional[Path] = None):
	if dates.size and dates[-1] == np.datetime64("today", "D"):
		y = y[:-1]
		dates = dates[:-1]
	if y.size == 0:
		typer.secho(f"Sin datos para {scope} ({tgt}), se omite.", fg=typer.colors.YELLOW)
		return
	_generate_eda_plots(scope, tgt, dates, y, feat_dates, X, out_dir)
	typer.secho(f"EDA generado para {scope} ({tgt}).", fg=typer.colors.GREEN)

def handle_prepare_data(product_id: str, start_date: Optional[str], end_date: Optional[str], target: str, out_dir: Optional[Path]) -> None:
	sd = date.fromisoformat(start_date) if start_date else None
	ed = date.fromisoformat(end_date) if end_date else None

	if product_id == "global":
		targets = ["quantity", "totalPrice"] if target == "both" else [target]
		for tgt in targets:
			dates, y = fetch_sales_totals_timeseries(sd, ed, target=tgt)  # type: ignore[arg-type]
			feat_dates, Xg = fetch_global_daily_features(sd, ed)
			_do_eda("global", tgt, dates, y, feat_dates, Xg, out_dir=out_dir)
			# Persistir dataset preparado
			if dates.size and dates[-1] == np.datetime64("today", "D"):
				dates = dates[:-1]
				y = y[:-1]
			path = _save_prepared_dataset("global", tgt, sd, ed, dates, y, feat_dates, Xg)
			typer.secho(f"Dataset preparado guardado: {path}", fg=typer.colors.GREEN)
		return

	if product_id == "all":
		ids = get_all_product_ids()
	else:
		ids = [product_id]

	for pid in ids:
		# features por producto
		dates_x, X = fetch_daily_features(pid, sd, ed)
		if target == "both":
			dates_q, y_q = fetch_sales_timeseries(pid, sd, ed, target="quantity")  # type: ignore[arg-type]
			_do_eda(f"product_{pid}", "quantity", dates_q, y_q, dates_x, X, out_dir=out_dir)
			if dates_q.size and dates_q[-1] == np.datetime64("today", "D"):
				dates_q = dates_q[:-1]
				y_q = y_q[:-1]
			path_q = _save_prepared_dataset(f"product_{pid}", "quantity", sd, ed, dates_q, y_q, dates_x, X)
			typer.secho(f"Dataset preparado guardado: {path_q}", fg=typer.colors.GREEN)
			dates_t, y_t = fetch_sales_timeseries(pid, sd, ed, target="totalPrice")  # type: ignore[arg-type]
			_do_eda(f"product_{pid}", "totalPrice", dates_t, y_t, dates_x, X, out_dir=out_dir)
			if dates_t.size and dates_t[-1] == np.datetime64("today", "D"):
				dates_t = dates_t[:-1]
				y_t = y_t[:-1]
			path_t = _save_prepared_dataset(f"product_{pid}", "totalPrice", sd, ed, dates_t, y_t, dates_x, X)
			typer.secho(f"Dataset preparado guardado: {path_t}", fg=typer.colors.GREEN)
		else:
			dates_y, y = fetch_sales_timeseries(pid, sd, ed, target=target)  # type: ignore[arg-type]
			_do_eda(f"product_{pid}", target, dates_y, y, dates_x, X, out_dir=out_dir)
			if dates_y.size and dates_y[-1] == np.datetime64("today", "D"):
				dates_y = dates_y[:-1]
				y = y[:-1]
			path = _save_prepared_dataset(f"product_{pid}", target, sd, ed, dates_y, y, dates_x, X)
			typer.secho(f"Dataset preparado guardado: {path}", fg=typer.colors.GREEN)


def handle_train(model: str, product_id: str, start_date: Optional[str], end_date: Optional[str], target: str) -> None:
	sd = date.fromisoformat(start_date) if start_date else None
	ed = date.fromisoformat(end_date) if end_date else None

	if product_id == "global":
		# Entrenamiento global agregado usando dataset preparado
		targets = ["quantity", "totalPrice"] if target == "both" else [target]
		for tgt in targets:
			try:
				_, y, X = _load_prepared_dataset("global", tgt, sd, ed)
			except FileNotFoundError as e:
				typer.secho(str(e), fg=typer.colors.RED)
				raise typer.Exit(code=2)
			if y.size == 0:
				typer.secho("Sin datos globales, se omite.", fg=typer.colors.YELLOW)
				continue
			if model == "sarimax":
				path = sarimax_train(y, "global", target=tgt)  # type: ignore[arg-type]
			elif model == "rnn":
				# Usar X exógeno global si está disponible en el dataset preparado
				path = rnn_train(y, "global", target=tgt, Xexo=X if X.size else None)  # type: ignore[arg-type]
				if path is None:
					typer.secho(
						f"Serie global demasiado corta para entrenar RNN (len={y.size}). Se omite.",
						fg=typer.colors.YELLOW,
					)
					continue
			else:
				typer.secho("Modelo no soportado (usar sarimax|rnn)", fg=typer.colors.RED)
				raise typer.Exit(code=3)
			typer.secho(f"Modelo {model} guardado (global {tgt}): {path}", fg=typer.colors.GREEN)
		return

	if product_id == "all":
		ids = get_all_product_ids()
		if not ids:
			typer.secho("No se encontraron productos.", fg=typer.colors.YELLOW)
			raise typer.Exit(code=0)
	else:
		ids = [product_id]

	for pid in ids:
		scope = f"product_{pid}"
		# Cargar dataset preparado (obligatorio)
		try:
			_, y, X = _load_prepared_dataset(scope, target if target != "both" else "quantity", sd, ed)
		except FileNotFoundError as e:
			typer.secho(str(e), fg=typer.colors.RED)
			typer.secho(
				f"Ejecutá primero: python -m sales_forecasting.cli prepare_data --product-id {pid} --target {target}",
				fg=typer.colors.YELLOW,
			)
			continue
		if y.size == 0:
			typer.secho(f"Sin datos para producto {pid}, se omite.", fg=typer.colors.YELLOW)
			continue
		if model == "sarimax":
			if target == "both":
				p1 = sarimax_train(y, pid, target="quantity")  # type: ignore[arg-type]
				# cargar dataset totalPrice
				try:
					_, y_tp, _ = _load_prepared_dataset(scope, "totalPrice", sd, ed)
				except FileNotFoundError as e:
					typer.secho(str(e), fg=typer.colors.RED)
					typer.secho(f"Ejecutá primero prepare_data para {pid} totalPrice", fg=typer.colors.YELLOW)
					y_tp = np.array([], dtype=float)
				if y_tp.size:
					p2 = sarimax_train(y_tp, pid, target="totalPrice")  # type: ignore[arg-type]
					typer.secho(f"Modelos {model} guardados: {p1}, {p2}", fg=typer.colors.GREEN)
				else:
					typer.secho(f"Modelo {model} guardado: {p1}", fg=typer.colors.GREEN)
				continue
			else:
				path = sarimax_train(y, pid, target=target)  # type: ignore[arg-type]
				typer.secho(f"Modelo {model} guardado: {path}", fg=typer.colors.GREEN)
				continue
		elif model == "rnn":
			if target == "both":
				# quantity
				X_aligned = X if X.size else None
				if X_aligned is not None:
					print("X_aligned shape: ", X_aligned.shape, "last 10 values: ", X_aligned[-10:])
				else:
					print("X_aligned: None")
				p1 = rnn_train(y, pid, target="quantity", Xexo=X_aligned)  # type: ignore[arg-type]
				if p1 is None:
					typer.secho(
						f"RNN corta para quantity (product_id={pid}, len={y.size}).",
						fg=typer.colors.YELLOW,
					)
				# totalPrice
				try:
					_, y_tp, X_tp = _load_prepared_dataset(scope, "totalPrice", sd, ed)
				except FileNotFoundError as e:
					typer.secho(str(e), fg=typer.colors.RED)
					typer.secho(f"Ejecutá primero prepare_data para {pid} totalPrice", fg=typer.colors.YELLOW)
					y_tp = np.array([], dtype=float)
					X_tp = np.zeros((0, 0), dtype=float)
				X_aligned_tp = X_tp if X_tp.size else None
				p2 = None
				if y_tp.size:
					p2 = rnn_train(y_tp, pid, target="totalPrice", Xexo=X_aligned_tp)  # type: ignore[arg-type]
				if p2 is None and y_tp.size:
					typer.secho(
						f"RNN corta para totalPrice (product_id={pid}, len={y_tp.size}).",
						fg=typer.colors.YELLOW,
					)
				if p1 or p2:
					typer.secho(
						f"Modelos {model} guardados: {p1 if p1 else ''} {p2 if p2 else ''}",
						fg=typer.colors.GREEN,
					)
				continue
			else:
				X_aligned = X if X.size else None
				if X_aligned is not None:
					print("X_aligned shape: ", X_aligned.shape, "last 10 values: ", X_aligned[-10:])
				else:
					print("X_aligned: None")
				path = rnn_train(y, pid, target=target, Xexo=X_aligned)  # type: ignore[arg-type]
				if path is None:
					typer.secho(
						f"Serie demasiado corta para entrenar RNN (product_id={pid}, len={y.size}). Se omite.",
						fg=typer.colors.YELLOW,
					)
					continue
		else:
			typer.secho("Modelo no soportado (usar sarimax|rnn)", fg=typer.colors.RED)
			raise typer.Exit(code=3)
		typer.secho(f"Modelo {model} guardado: {path}", fg=typer.colors.GREEN)


def handle_predict(model: str, product_id: str, horizon: int, target: str, out: Optional[Path]) -> None:
	def _predict_single(tgt: str):
		nonlocal model, product_id, horizon
		if model == "sarimax":
			return sarimax_forecast("global" if product_id == "global" else product_id, horizon, target=tgt)  # type: ignore[arg-type]
		elif model == "rnn":
			if product_id == "global":
				_, y = fetch_sales_totals_timeseries(target=tgt)  # type: ignore[arg-type]
				X_aligned = None
			else:
				_, y = fetch_sales_timeseries(product_id, target=tgt)  # type: ignore[arg-type]
				_, X = fetch_daily_features(product_id)
				if y.size == 0:
					typer.secho("No hay datos para predecir", fg=typer.colors.RED)
					raise typer.Exit(code=4)
				X_aligned = X[-y.shape[0] :, :] if X.size and X.shape[0] >= y.shape[0] else None
			return rnn_forecast("global" if product_id == "global" else product_id, horizon, y, target=tgt, Xexo=X_aligned)  # type: ignore[arg-type]
		else:
			typer.secho("Modelo no soportado (usar sarimax|rnn)", fg=typer.colors.RED)
			raise typer.Exit(code=3)

	if target == "both":
		preds_qty = _predict_single("quantity")
		preds_amt = _predict_single("totalPrice")
		lines = [
			"forecast_quantity,forecast_totalPrice",
			*[f"{float(q):.6f},{float(a):.6f}" for q, a in zip(preds_qty, preds_amt)],
		]
		typer.secho("\n".join(lines), fg=typer.colors.BLUE)
		if out:
			with out.open("w", encoding="utf-8") as f:
				f.write("forecast_quantity,forecast_totalPrice\n")
				for q, a in zip(preds_qty, preds_amt):
					f.write(f"{float(q):.6f},{float(a):.6f}\n")
			typer.secho(f"Guardado en {out}", fg=typer.colors.GREEN)
		return

	preds = _predict_single(target)

	# Mostrar y/o guardar
	lines = [f"{float(p):.4f}" for p in preds]
	typer.secho("\n".join(lines), fg=typer.colors.BLUE)
	if out:
		with out.open("w", encoding="utf-8") as f:
			f.write("forecast\n")
			for p in preds:
				f.write(f"{float(p):.6f}\n")
		typer.secho(f"Guardado en {out}", fg=typer.colors.GREEN)


def handle_run_api(host: str, port: int) -> None:
	uvicorn.run("sales_forecasting.api:app", host=host, port=port, reload=False)
