from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_plot(filename: str, output_dir: Path | None) -> None:
    if output_dir is None:
        return
    _ensure_dir(output_dir)
    plt.savefig(output_dir / filename, bbox_inches="tight", dpi=150)


def plot_monthly_orders_distribution(
    data_df: pd.DataFrame,
    with_plots: bool = True,
    output_dir: Path | None = None,
) -> None:
    """
    Recibe el DataFrame de órdenes con columna 'created_month' y grafica
    la distribución mensual de órdenes asegurando 12 barras (meses 1..12).
    No retorna nada.
    """
    if not with_plots:
        return
    
    df_month = (
        data_df
            .dropna(subset=['created_month'])
            .groupby('created_month')
            .size()
            .reset_index(name='orders')
            .sort_values('created_month')
    )
    # Asegurar 12 meses (1..12) con 0 en faltantes
    df_month = (
        df_month
            .set_index('created_month')
            .reindex(range(1, 13), fill_value=0)
            .reset_index()
            .rename(columns={'index': 'created_month'})
    )
    plt.figure(figsize=(10, 6))
    plt.bar(df_month['created_month'], df_month['orders'], label='Órdenes por mes')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad de órdenes')
    plt.title('Distribución de órdenes por mes')
    plt.legend()
    _save_plot("orders_by_month.png", output_dir)
    plt.show()


def plot_weekday_orders_distribution(
    data_df: pd.DataFrame,
    with_plots: bool = True,
    output_dir: Path | None = None,
) -> None:
    """
    Recibe el DataFrame de órdenes con columna 'created_weekday' y grafica
    la distribución por día de la semana asegurando 7 barras (0..6).
    No retorna nada.
    """
    if not with_plots:
        return
    
    df_weekday = (
        data_df
            .dropna(subset=['created_weekday'])
            .groupby('created_weekday')
            .size()
            .reset_index(name='orders')
            .sort_values('created_weekday')
    )
    # Asegurar días 0..6 con 0 en faltantes
    df_weekday = (
        df_weekday
            .set_index('created_weekday')
            .reindex(range(0, 7), fill_value=0)
            .reset_index()
            .rename(columns={'index': 'created_weekday'})
    )
    plt.figure(figsize=(10, 6))
    plt.bar(df_weekday['created_weekday'], df_weekday['orders'], label='Órdenes por día de la semana')
    plt.xlabel('Día de la semana')
    plt.ylabel('Cantidad de órdenes')
    plt.title('Distribución de órdenes por día de la semana')
    plt.legend()
    _save_plot("orders_by_weekday.png", output_dir)
    plt.show()


def group_orders_per_day_and_plot(
    data_df: pd.DataFrame,
    with_plots: bool = False,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Agrupa órdenes por 'created' (fecha, día) y grafica la serie temporal.
    Retorna el DataFrame agrupado con columnas ['created', 'orders'].
    """
    df_grouped = (
        data_df
            .dropna(subset=['created'])
            .groupby('created')
            .size()
            .reset_index(name='orders')
            .sort_values('created')
    )
    print("df_grouped:")
    print(df_grouped.head())
    print(df_grouped.describe())
    print(df_grouped.info())

    # Serie temporal de órdenes por día
    if with_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(df_grouped['created'], df_grouped['orders'], label='Órdenes por fecha de creación')
        plt.xlabel('Fecha de creación')
        plt.ylabel('Cantidad de órdenes')
        plt.title('Cantidad de órdenes por fecha de creación')
        plt.legend()
        _save_plot("orders_by_day.png", output_dir)
        plt.show()

    return df_grouped


def plot_daily_unique_customers(
    data_df: pd.DataFrame,
    with_plots: bool = True,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Agrupa por fecha 'created' y cuenta la cantidad de 'customerId' distintos por día.
    Grafica la serie temporal de clientes únicos por día.
    """
    df_unique_customers = (
        data_df
            .dropna(subset=['created', 'customerId'])
            .groupby('created')['customerId']
            .nunique()
            .reset_index(name='unique_customers')
            .sort_values('created')
    )
    print("df_unique_customers:")
    print(df_unique_customers.head())
    print(df_unique_customers.describe())
    print(df_unique_customers.info())

    if with_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(df_unique_customers['created'], df_unique_customers['unique_customers'], label='Clientes únicos por día')
        plt.xlabel('Fecha')
        plt.ylabel('Clientes únicos')
        plt.title('Clientes únicos por día')
        plt.legend()
        _save_plot("unique_customers_by_day.png", output_dir)
        plt.show()

    return df_unique_customers


def plot_daily_new_customers(
    data_df: pd.DataFrame,
    with_plots: bool = True,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Cuenta clientes que hicieron su primera compra por día usando
    order_number_for_customer == 1. Devuelve el DataFrame con
    columnas ['created', 'new_customers'] y grafica.
    """
    df_new_customers = (
        data_df
            .dropna(subset=['created', 'customerId', 'order_number_for_customer'])
            .loc[data_df['order_number_for_customer'] == 1]
            .groupby('created')['customerId']
            .nunique()
            .reset_index(name='new_customers')
            .sort_values('created')
    )

    print("df_new_customers:")
    print(df_new_customers.head())
    print(df_new_customers.describe())
    print(df_new_customers.info())

    if with_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(df_new_customers['created'], df_new_customers['new_customers'], label='Clientes nuevos por día')
        plt.xlabel('Fecha')
        plt.ylabel('Clientes nuevos')
        plt.title('Clientes nuevos por día (primera compra)')
        plt.legend()
        _save_plot("new_customers_by_day.png", output_dir)
        plt.show()

    return df_new_customers


def plot_explode_line_items(
    data_df: pd.DataFrame,
    with_plots: bool = True,
    output_dir: Path | None = None,
) -> None:
    """
    Explota las líneas de items de las órdenes y grafica la distribución de los productos.
    """
    df_line_items = data_df.copy()
    # Expandir solo las claves de interés del dict en 'line_items'
    line_details = (
        df_line_items["line_items"]
            .apply(lambda v: v if isinstance(v, dict) else {})
            .apply(pd.Series)
            .reindex(columns=['productId', 'quantity'])
    )
    df_line_items = pd.concat(
        [df_line_items.drop(columns="line_items"), line_details],
        axis=1
    )

    # Normalización de tipos
    df_line_items['productId'] = pd.to_numeric(df_line_items['productId'], errors='coerce').fillna(0).astype(int)
    df_line_items['quantity'] = pd.to_numeric(df_line_items['quantity'], errors='coerce').fillna(0).astype(int)

    print("df_line_items:")
    print(df_line_items.head())
    print(df_line_items.describe())
    print(df_line_items.info())

    # Se pueden agregar gráficas específicas aquí si es necesario


def run_analysis(
    input_dir: str | Path = "reports/eda/data",
    with_plots: bool = True,
    plots_dir: str | Path = "reports/eda/plots",
) -> None:
    """
    Carga datasets guardados, muestra resúmenes por consola y genera gráficos.
    """
    in_dir = Path(input_dir)
    orders_path = in_dir / "orders.jsonl"
    ad_spends_path = in_dir / "ad_spends.jsonl"

    if not orders_path.exists():
        raise FileNotFoundError(f"No existe el dataset de órdenes: {orders_path}")

    data_df = pd.read_json(orders_path, lines=True, dtype=False)
    output_dir = Path(plots_dir) if with_plots else None

    print("data_df (ordenes) cargado:")
    print(data_df.head())
    print(data_df.describe())
    print(data_df.info())

    # Gráficas y agrupaciones básicas
    df_group_orders_per_day = group_orders_per_day_and_plot(
        data_df, with_plots, output_dir=output_dir
    )
    plot_weekday_orders_distribution(data_df, with_plots, output_dir=output_dir)
    plot_monthly_orders_distribution(data_df, with_plots, output_dir=output_dir)

    df_daily_unique_customers = plot_daily_unique_customers(
        data_df, with_plots, output_dir=output_dir
    )
    df_daily_new_customers = plot_daily_new_customers(
        data_df, with_plots, output_dir=output_dir
    )

    plot_explode_line_items(data_df, with_plots, output_dir=output_dir)

    # Combinar con ad_spends si está disponible
    if ad_spends_path.exists():
        ad_spends_df = pd.read_json(ad_spends_path, lines=True, dtype=False)
        df_combined = build_combined_with_ad_spends(
            df_group_orders_per_day=df_group_orders_per_day,
            ad_spends_df=ad_spends_df,
            df_daily_unique_customers=df_daily_unique_customers,
            df_daily_new_customers=df_daily_new_customers,
        )

        combined_path = in_dir / "combined.jsonl"
        df_combined.to_json(combined_path, orient="records", lines=True, date_format="iso")

        print("Combined DataFrame:")
        print(df_combined.head())
        print(df_combined.describe())
        print(df_combined.info())
        print(f"Combined dataset guardado en: {combined_path}")


def build_combined_with_ad_spends(
    df_group_orders_per_day: pd.DataFrame,
    ad_spends_df: pd.DataFrame,
    df_daily_unique_customers: pd.DataFrame,
    df_daily_new_customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une órdenes diarias con ad_spends y agrega métricas de clientes.
    """
    df_group_orders_per_day = df_group_orders_per_day.copy()
    ad_spends_df = ad_spends_df.copy()
    df_daily_unique_customers = df_daily_unique_customers.copy()
    df_daily_new_customers = df_daily_new_customers.copy()

    # Normalizar tipos de fecha para evitar merges object vs datetime64
    df_group_orders_per_day["created"] = pd.to_datetime(
        df_group_orders_per_day["created"], errors="coerce"
    ).dt.date
    ad_spends_df["date"] = pd.to_datetime(
        ad_spends_df["date"], errors="coerce"
    ).dt.date
    df_daily_unique_customers["created"] = pd.to_datetime(
        df_daily_unique_customers["created"], errors="coerce"
    ).dt.date
    df_daily_new_customers["created"] = pd.to_datetime(
        df_daily_new_customers["created"], errors="coerce"
    ).dt.date

    df_combined = (
        pd.merge(df_group_orders_per_day, ad_spends_df, left_on='created', right_on='date', how='outer')
          .assign(day=lambda df: df['created'].fillna(df['date']))
          .drop(columns=['date'])
          .sort_values('day')
    )
    df_combined['orders'] = df_combined['orders'].fillna(0).astype(int)
    df_combined['created'] = pd.to_datetime(df_combined['created'])
    df_combined['created_weekday'] = df_combined['created'].dt.dayofweek
    df_combined['created_month'] = df_combined['created'].dt.month
    df_combined['created'] = df_combined['created'].dt.date

    df_combined = pd.merge(df_combined, df_daily_unique_customers, on='created', how='left')
    df_combined = pd.merge(df_combined, df_daily_new_customers, on='created', how='left')
    df_combined['unique_customers'] = df_combined['unique_customers'].fillna(0).astype(int)
    df_combined['new_customers'] = df_combined['new_customers'].fillna(0).astype(int)
    df_combined = df_combined.drop(columns=['day'])

    return df_combined



