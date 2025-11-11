from matplotlib import pyplot as plt
from eda.db import get_all_ad_spends, get_all_data
import pandas as pd

def handle_prepare_data() -> None:
    """
        Prepara los datos para el EDA.
    """
    data = get_all_data()

    data_df = pd.DataFrame(data)
    # Usamos una serie datetime para derivar campos y luego guardamos sólo la fecha en 'created'
    created_dt = pd.to_datetime(data_df['created'])
    # Nuevas columnas desde la fecha de creación
    # 0 = Lunes ... 6 = Domingo
    data_df['created_weekday'] = created_dt.dt.dayofweek
    # 1 = Enero ... 12 = Diciembre
    data_df['created_month'] = created_dt.dt.month
    # Guardamos sólo la fecha para agrupar por día
    data_df['created'] = created_dt.dt.date
    print("data_df:")
    print(data_df.head())
    print(data_df.describe())
    print(data_df.info())

    # Agrupar por fecha de creación y contar cantidad de órdenes
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

    # Crear gráfico de series temporales para cantidad de órdenes por fecha de creación
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped['created'], df_grouped['orders'], label='Órdenes por fecha de creación')
    plt.xlabel('Fecha de creación')
    plt.ylabel('Cantidad de órdenes')
    plt.title('Cantidad de órdenes por fecha de creación')
    plt.legend()
    plt.show()

    # Crear gráfico de distribución de órdenes por día de la semana
    df_weekday = (
        data_df
            .dropna(subset=['created_weekday'])
            .groupby('created_weekday')
            .size()
            .reset_index(name='orders')
            .sort_values('created_weekday')
    )
    plt.figure(figsize=(10, 6))
    plt.bar(df_weekday['created_weekday'], df_weekday['orders'], label='Órdenes por día de la semana')
    plt.xlabel('Día de la semana')
    plt.ylabel('Cantidad de órdenes')
    plt.title('Distribución de órdenes por día de la semana')
    plt.legend()
    plt.show()

    # Crear gráfico de distribución de órdenes por mes
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

    ad_spends_df = handle_analyze_ad_spends()

    # Unir por fecha: df_grouped.created con ad_spends_df.date
    combined_df = (
        pd.merge(df_grouped, ad_spends_df, left_on='created', right_on='date', how='outer')
          .assign(day=lambda df: df['created'].fillna(df['date']))
          .drop(columns=['date'])
          .sort_values('day')
    )
    # Rellenar órdenes faltantes con 0 cuando no hubo pedidos
    combined_df['orders'] = combined_df['orders'].fillna(0).astype(int)
    combined_df['created'] = pd.to_datetime(combined_df['created'])
    # 0 = Lunes ... 6 = Domingo
    combined_df['created_weekday'] = combined_df['created'].dt.dayofweek
    # 1 = Enero ... 12 = Diciembre
    combined_df['created_month'] = combined_df['created'].dt.month
    combined_df['created'] = combined_df['created'].dt.date

    # Elimino columna day que es una copia de created
    combined_df = combined_df.drop(columns=['day'])
    
    print("Combined DataFrame:")
    print(combined_df.head())
    print(combined_df.describe())
    print(combined_df.info())
    

def handle_analyze_ad_spends() -> None:
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

    print("ad_spends_df:")
    print(ad_spends_df.head())
    print(ad_spends_df.describe())
    print(ad_spends_df.info())

    return ad_spends_df
    