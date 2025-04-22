# modules/sentiment_analysis.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# --------------------------------------------------------------------
#   Función cacheada para cargar y preparar datos de Fear & Greed
# --------------------------------------------------------------------
@st.cache_data
def load_fear_and_greed_data():
    """
    Descarga el Fear & Greed Index completo desde la API de Alternative.me
    y retorna un DataFrame con sus valores y la información del precio de BTC.
    """
    # Descargamos el histórico del índice
    r = requests.get('https://api.alternative.me/fng/?limit=0')
    
    # Convertimos a DataFrame
    df = pd.DataFrame(r.json()['data'])
    df['value'] = df['value'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df = df.iloc[::-1]  # Revertimos el orden (más antiguo primero)

    # Obtenemos precio de BTC
        # Obtenemos precio de BTC con manejo de errores y auto ajuste
    try:
        df_btc = yf.download('BTC-USD', auto_adjust=True, progress=False)[['Close']]
    except Exception as e:
        st.error(f"Error al descargar datos de BTC: {e}")
        return None, None, None, None, None, None


    df_btc.index.name = 'timestamp'
    df_btc.columns = ['Close']

    # Combinamos
    tog = df.merge(df_btc, on='timestamp', how='inner')
    
    # Calculamos cambio porcentual diario y generamos señal
    tog['change'] = tog['Close'].pct_change()
    tog['position'] = np.where(tog['value'] > 50, 1, 0)

    # Calculamos la estrategia y su capital acumulado
    strategy = tog['position'] * tog['change']
    tog['strategy_cum'] = (strategy + 1).cumprod()
    tog['btc_cum'] = (tog['change'] + 1).cumprod()

    # Filtramos DataFrames por clasificación
    df_Fear = tog[tog['value_classification'] == 'Fear']
    df_Neutral = tog[tog['value_classification'] == 'Neutral']
    df_Greed = tog[tog['value_classification'] == 'Greed']
    df_Extreme_Fear = tog[tog['value_classification'] == 'Extreme Fear']
    df_Extreme_Greed = tog[tog['value_classification'] == 'Extreme Greed']

    return tog, df_Fear, df_Neutral, df_Greed, df_Extreme_Fear, df_Extreme_Greed

# --------------------------------------------------------------------
#   Función principal que integra la lógica y muestra gráficos en Streamlit
# --------------------------------------------------------------------
def show_sentiment_analysis():
    st.header("Análisis de Sentimiento (Fear & Greed)")

    # Cargamos datos (cacheado)
    tog, df_Fear, df_Neutral, df_Greed, df_Extreme_Fear, df_Extreme_Greed = load_fear_and_greed_data()

    # -------------------------------------------------------
    # Gráfico 1: Precio de BTC vs. marcadores de Fear & Greed
    # -------------------------------------------------------
    fig = go.Figure()

    # Línea del precio de BTC
    fig.add_trace(go.Scatter(
        x=tog.index,
        y=tog['Close'],
        mode='lines',
        name='BTC Closing Price'
    ))

    # Puntos de Extreme Greed
    fig.add_trace(go.Scatter(
        x=df_Extreme_Greed.index,
        y=df_Extreme_Greed['Close'],
        mode='markers',
        name='Extreme Greed',
        marker_color='rgba(0, 102, 0, .8)'
    ))

    # Puntos de Greed
    fig.add_trace(go.Scatter(
        x=df_Greed.index,
        y=df_Greed['Close'],
        mode='markers',
        name='Greed',
        marker_color='rgba(153, 255, 102, .8)'
    ))

    # Puntos de Neutral
    fig.add_trace(go.Scatter(
        x=df_Neutral.index,
        y=df_Neutral['Close'],
        mode='markers',
        name='Neutral',
        marker_color='rgba(0, 255, 255, .8)'
    ))

    # Puntos de Fear
    fig.add_trace(go.Scatter(
        x=df_Fear.index,
        y=df_Fear['Close'],
        mode='markers',
        name='Fear',
        marker_color='rgba(255, 102, 102, .8)'
    ))

    # Puntos de Extreme Fear
    fig.add_trace(go.Scatter(
        x=df_Extreme_Fear.index,
        y=df_Extreme_Fear['Close'],
        mode='markers',
        name='Extreme Fear',
        marker_color='rgba(204, 0, 0, .8)'
    ))

    fig.update_layout(
        title='BTC Price vs. Fear & Greed Classifications',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------
    # Gráfico 2: Comparación de estrategias
    # ------------------------------------
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=tog.index,
        y=tog['strategy_cum'],
        mode='lines',
        name='F&G Strategy (Cumulative)'
    ))

    fig2.add_trace(go.Scatter(
        x=tog.index,
        y=tog['btc_cum'],
        mode='lines',
        name='BTC Buy & Hold (Cumulative)'
    ))

    fig2.update_layout(
        title='Estrategia F&G vs. BTC (rendimiento acumulado)',
        xaxis_title='Fecha',
        yaxis_title='Crecimiento Acumulado',
        hovermode='x unified'
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------------
    #   Mostrar tabla con las últimas filas como resumen
    # ---------------------------------------------------------
    st.subheader("Últimos valores del DataFrame combinado")
    st.dataframe(tog.tail(10))

    st.write("La estrategia de F&G se basa en ir en largo (posición 1) cuando el valor del índice es mayor a 50.")
    st.write("Con el gráfico 2, comparamos el rendimiento de esa estrategia frente a comprar y mantener BTC (Buy & Hold).")

    st.write("¡Espero que te sirva para tomar decisiones informadas en tus inversiones!")