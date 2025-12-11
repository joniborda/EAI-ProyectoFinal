from matplotlib import pyplot as plt
from eda.db import get_all_ad_spends, get_all_data
import pandas as pd

def handle_prepare_data(with_plots: bool = True) -> None:
    """
        Prepara los datos para el EDA.
    """
    data = get_all_data()

    df_data = pd.DataFrame(data)
    # Usamos una serie datetime para derivar campos y luego guardamos sólo la fecha en 'created'
    created_dt = pd.to_datetime(df_data['created'])
    # Nuevas columnas desde la fecha de creación
    # 0 = Lunes ... 6 = Domingo
    df_data['created_weekday'] = created_dt.dt.dayofweek
    # 1 = Enero ... 12 = Diciembre
    df_data['created_month'] = created_dt.dt.month
    # Guardamos sólo la fecha para agrupar por día
    df_data['created'] = created_dt.dt.date
    print("data_df original:")
    print(df_data.head())
    print(df_data.describe())
    print(df_data.info())

    # Agrupar y graficar órdenes por día (devuelve df_grouped)
    df_group_orders_per_day = group_orders_per_day_and_plot(df_data, with_plots)

    plot_weekday_orders_distribution(df_data, with_plots)

    plot_monthly_orders_distribution(df_data, with_plots)

    df_daily_unique_customers = plot_daily_unique_customers(df_data, with_plots)

    df_daily_new_customers = plot_daily_new_customers(df_data, with_plots)

    df_ad_spends = handle_analyze_ad_spends(with_plots)

    plot_explode_line_items(df_data, with_plots)

    # Unir por fecha: df_grouped.created con ad_spends_df.date
    df_combined = (
        pd.merge(df_group_orders_per_day, df_ad_spends, left_on='created', right_on='date', how='outer')
          .assign(day=lambda df: df['created'].fillna(df['date']))
          .drop(columns=['date'])
          .sort_values('day')
    )
    # Rellenar órdenes faltantes con 0 cuando no hubo pedidos
    df_combined['orders'] = df_combined['orders'].fillna(0).astype(int)
    df_combined['created'] = pd.to_datetime(df_combined['created'])
    # 0 = Lunes ... 6 = Domingo
    df_combined['created_weekday'] = df_combined['created'].dt.dayofweek
    # 1 = Enero ... 12 = Diciembre
    df_combined['created_month'] = df_combined['created'].dt.month
    df_combined['created'] = df_combined['created'].dt.date
    
    df_combined = pd.merge(df_combined, df_daily_unique_customers, left_on='created', right_on='created', how='left')
    df_combined = pd.merge(df_combined, df_daily_new_customers, left_on='created', right_on='created', how='left')

    df_combined['unique_customers'] = df_combined['unique_customers'].fillna(0).astype(int)
    df_combined['new_customers'] = df_combined['new_customers'].fillna(0).astype(int)
    # Elimino columna day que es una copia de created
    df_combined = df_combined.drop(columns=['day'])
    
    print("Combined DataFrame:")
    print(df_combined.head())
    print(df_combined.describe())
    print(df_combined.info())
    

def handle_analyze_ad_spends(with_plots: bool = True) -> None:
    ad_spends = get_all_ad_spends()
    ad_spends_df = pd.DataFrame(ad_spends)
    ad_spends_df['date'] = pd.to_datetime(ad_spends_df['date'])
    ad_spends_df['date'] = ad_spends_df['date'].dt.date
    ad_spends_df['adSpend'] = ad_spends_df['adSpend'].fillna(0).astype(float)
    ad_spends_df['rcRevenue'] = ad_spends_df['rcRevenue'].fillna(0).astype(float)
    ad_spends_df['totalRevenue'] = ad_spends_df['totalRevenue'].fillna(0).astype(float)
    ad_spends_df['totalMerchSold'] = ad_spends_df['totalMerchSold'].fillna(0).astype(int)
    ad_spends_df['totalNewCustomerBarsSold'] = ad_spends_df['totalNewCustomerBarsSold'].fillna(0).astype(int)
    ad_spends_df['totalRecurringCustomerBarsSold'] = ad_spends_df['totalRecurringCustomerBarsSold'].fillna(0).astype(int)

    if with_plots:
        print("ad_spends_df:")
        print(ad_spends_df.head())
        print(ad_spends_df.describe())
        print(ad_spends_df.info())

    return ad_spends_df


def plot_monthly_orders_distribution(data_df: pd.DataFrame, with_plots: bool = True) -> None:
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
    plt.show()


def plot_weekday_orders_distribution(data_df: pd.DataFrame, with_plots: bool = True) -> None:
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
    plt.show()


def group_orders_per_day_and_plot(data_df: pd.DataFrame, with_plots: bool = False) -> pd.DataFrame:
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
        plt.show()

    return df_grouped

def plot_daily_unique_customers(data_df: pd.DataFrame, with_plots: bool = True) -> None:
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
        plt.show()

    return df_unique_customers


def plot_daily_new_customers(data_df: pd.DataFrame, with_plots: bool = True) -> pd.DataFrame:
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
        plt.show()

    return df_new_customers

def plot_explode_line_items(data_df: pd.DataFrame, with_plots: bool = True) -> None:
    """
    Explota las líneas de items de las órdenes y grafica la distribución de los productos.
    """
    
    # df_line_items = data_df.explode("line_items").reset_index(drop=True)
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
    # df_line_items['totalPrice'] = df_line_items['totalPrice'].fillna(0).astype(float)

    print("df_line_items:")
    print(df_line_items.head())
    print(df_line_items.describe())
    print(df_line_items.info())