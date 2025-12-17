# modules/technical_indicators.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from plotly.subplots import make_subplots
from modules.data_loader import load_market_data

# -----------------------------------------------------------
#  1) Funciones de indicadores
# -----------------------------------------------------------
def compute_SMA(series, window):
    """Calcula la Media Móvil Simple (SMA)."""
    return series.rolling(window=window).mean()

def compute_RSI(series, period=14):
    """Calcula el RSI (Relative Strength Index) para la serie dada."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_OBV(df):
    """Calcula el OBV (On Balance Volume) a partir del DataFrame."""
    obv = [0]
    close = df['Close'].reset_index(drop=True)
    volume = df['Volume'].reset_index(drop=True)

    for i in range(1, len(df)):
        prev_close = float(close.iloc[i-1])
        curr_close = float(close.iloc[i])
        curr_volume = float(volume.iloc[i])

        if curr_close > prev_close:
            obv.append(obv[-1] + curr_volume)
        elif curr_close < prev_close:
            obv.append(obv[-1] - curr_volume)
        else:
            obv.append(obv[-1])

    return pd.Series(obv, index=df.index)

def get_local_extrema(series, order=5):
    """
    Devuelve los índices de los extremos locales (mínimos y máximos)
    en una serie, utilizando 'order' para definir la ventana.
    """
    values = series.values
    minima_idx = argrelextrema(values, np.less, order=order)[0]
    maxima_idx = argrelextrema(values, np.greater, order=order)[0]
    return minima_idx, maxima_idx

def fit_trendlines_high_low(high, low, close):
    """
    Calcula líneas de soporte y resistencia usando regresión lineal.
    """
    x = np.arange(len(close))
    coefs_support = np.polyfit(x, low, 1)
    coefs_resist = np.polyfit(x, high, 1)
    return coefs_support, coefs_resist

# --------------------------------------------------------------------
#   Función cacheada para descargar y procesar los datos
# --------------------------------------------------------------------
@st.cache_data
def load_and_process_data(ticker: str, start_date: str, order_value=5, last_n=56):
    df = load_market_data(ticker, start_date)
    
    if df.empty:
        return None

    df['SMA20'] = compute_SMA(df['Close'], 20)
    df['SMA50'] = compute_SMA(df['Close'], 50)
    df['RSI14'] = compute_RSI(df['Close'], period=14)
    df['OBV']   = calculate_OBV(df)

    # Cortamos los últimos valores
    candles = df.iloc[-last_n:].copy()

    # Aseguramos nombres de columnas
    if isinstance(candles.columns, pd.MultiIndex):
        candles.columns = [col[0] for col in candles.columns]
    else:
        candles.columns = [str(col).strip() for col in candles.columns]

    # Verificamos columnas necesarias
    expected_cols = ['High', 'Low', 'Close', 'Volume']
    for col in expected_cols:
        if col not in candles.columns:
            st.error(f"La columna '{col}' no se encuentra en los datos descargados.")
            st.stop()

    # Forzamos valores numéricos
    candles = candles.apply(pd.to_numeric, errors='coerce').fillna(method='ffill')

    # Calculamos extremos locales sobre los últimos datos
    min_idx, max_idx = get_local_extrema(candles['Close'], order=order_value)
    supports = candles['Close'].iloc[min_idx]
    resistances = candles['Close'].iloc[max_idx]

    support_val = supports.median() if len(supports) > 0 else np.nan
    resist_val = resistances.median() if len(resistances) > 0 else np.nan

    candles['support_level'] = round(support_val, 0)
    candles['resistance_level'] = round(resist_val, 0)

    # Líneas de tendencia
    support_coefs, resist_coefs = fit_trendlines_high_low(
        candles['High'].values,
        candles['Low'].values,
        candles['Close'].values
    )

    x_vals = np.arange(len(candles))
    support_line = support_coefs[0] * x_vals + support_coefs[1]
    resist_line  = resist_coefs[0] * x_vals + resist_coefs[1]


    # Pivote Clásico (calculado con última vela de la ventana)
    last = candles.iloc[-1]
    pivot = (last['High'] + last['Low'] + last['Close']) / 3
    r1 = 2 * pivot - last['Low']
    s1 = 2 * pivot - last['High']
    r2 = pivot + (last['High'] - last['Low'])
    s2 = pivot - (last['High'] - last['Low'])

    # Volumen por rango de precios
    bins = pd.cut(candles['Close'], bins=20)
    volume_by_price = candles.groupby(bins)['Volume'].sum().sort_values(ascending=False)
    volume_levels = volume_by_price.head(3).index  # rangos con mayor volumen

    # Extraemos los valores centrales de los intervalos
    volume_support_resistance = [interval.mid for interval in volume_levels]

    # Agrega estos al return:
    return df, candles, support_line, resist_line, pivot, r1, s1, r2, s2, volume_support_resistance


# --------------------------------------------------------------------
#   Función principal de Streamlit
# --------------------------------------------------------------------
def show_technical_indicators():
    st.header("Indicadores Técnicos y Líneas de Tendencia")

    #ticker = st.sidebar.text_input("Ticker (YFinance)", "BTC-USD")

    modo_seleccion = st.sidebar.radio("Modo de selección de ticker:", ["Predefinido", "Escribir manualmente"])
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    if modo_seleccion == "Predefinido":
        ticker = st.sidebar.selectbox("Selecciona un Activo", tickers)
    else:
        ticker = st.sidebar.text_input("Escribe un ticker", "AAPL").upper()

    start_date = st.sidebar.text_input("Fecha de Inicio (YYYY-MM-DD)", "2023-01-01")
    order_value = st.sidebar.slider("Ventana para Soportes/Resistencias (order)", min_value=3, max_value=10, value=5)
    last_n = st.sidebar.slider("Velas a mostrar en el gráfico", min_value=20, max_value=150, value=56)

    data_result = load_and_process_data(ticker, start_date, order_value, last_n)

    if data_result is None:
        st.warning("No se encontraron datos para el rango y ticker especificados.")
        return

    df, candles, support_line, resist_line, pivot, r1, s1, r2, s2, volume_levels = data_result

    st.markdown("### 🧭 Niveles Clave de Soporte y Resistencia")
    st.markdown("""
    Estos niveles técnicos ayudan a identificar posibles zonas donde el precio podría **rebotar o romper**.
    - 🟦 *Pivote clásico*: cálculo basado en el último día (High, Low, Close).
    - 🟩 *R1 / R2*: resistencias derivadas del pivote.
    - 🟥 *S1 / S2*: soportes derivados del pivote.
    - 🟨 *Resistencias / Soportes medios*: basados en análisis de máximos/mínimos locales.
    - ⚫ *Volumen*: niveles de precio con mayor acumulación de volumen (zonas de interés del mercado).
    """)


    with st.expander("📘 Explicaciones de los Niveles Técnicos"):
        st.markdown("""
        ### 📌 ¿Qué es el **Nivel Pivote**?
        El **Nivel Pivote** es un punto de referencia calculado a partir del precio máximo, mínimo y cierre del día anterior.  
        Se utiliza para estimar zonas donde el precio podría **cambiar de dirección**.

        Si el precio está por encima del pivote, se considera que el mercado es **alcista** (optimista).  
        Si está por debajo, se considera **bajista** (pesimista).

        **Fórmula**:  
        `Pivote = (Máximo + Mínimo + Cierre) / 3`

        ---

        ### 🟩 R1 / R2 y 🟥 S1 / S2
        Son **niveles derivados del pivote**, que actúan como posibles **zonas de soporte y resistencia**:

        - **Resistencia R1 / R2**: zonas donde el precio podría **detenerse o retroceder** al subir.  
        - **Soporte S1 / S2**: zonas donde el precio podría **rebotar** al bajar.

        ---

        ### ⚫ ¿Qué es el **Nivel de Volumen 1 y 2**?
        Son precios donde se ha negociado **la mayor cantidad de volumen** recientemente.

        Estos niveles indican zonas de fuerte interés por parte de compradores y vendedores, y pueden actuar como **barreras naturales** para el movimiento del precio.

        Si el precio se acerca a uno de estos niveles, es común ver **reacciones del mercado**, como consolidaciones, rebotes o rupturas.
        """)


    # Define un formateador de valores
    def format_currency(value, currency='USD'):
        if currency == 'USD':
            return f"${value:,.0f}"
        return f"{value:,.0f}"

    # Puedes inferir la moneda a partir del ticker
    # Ejemplo: ticker = "BTC-USD" -> 'USD'
    
    currency = ticker.split("-")[-1] if "-" in ticker else "USD"

    # Fila 1 - Pivote clásico
    cols = st.columns(5)
    cols[0].metric("🟦  Nivel Pivote", format_currency(pivot, currency))
    cols[1].metric("🟩 Resistencia R1", format_currency(r1, currency))
    cols[2].metric("🟥 Soporte S1", format_currency(s1, currency))
    cols[3].metric("🟩 Resistencia R2", format_currency(r2, currency))
    cols[4].metric("🟥 Soporte S2", format_currency(s2, currency))

    # Fila 2 - Promedios + Volumen
    cols2 = st.columns(5)
    cols2[0].metric("💰 Precio Hoy", format_currency(int(df['Close'].iloc[-1]), currency))
    cols2[1].metric("📈 Resistencia Media (local)", format_currency(candles['resistance_level'].iloc[-1], currency))
    cols2[2].metric("📉 Soporte Medio (local)", format_currency(candles['support_level'].iloc[-1], currency))
    cols2[3].metric("⚫ Nivel Volumen 1", format_currency(volume_levels[0], currency))
    cols2[4].metric("⚫ Nivel Volumen 2", format_currency(volume_levels[1], currency))




    st.subheader(f"Últimos {last_n} días de: {ticker}")

    fig = go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker}: Soportes, Resistencias e Indicadores Técnicos", "Volumen")
    )

    # --- Gráfico de velas (fila 1)

    fig.add_trace(go.Candlestick(
        x=candles.index,
        open=candles['Open'],
        high=candles['High'],
        low=candles['Low'],
        close=candles['Close'],
        name='Velas'
    ), row=1, col=1)


    fig.add_trace(go.Scatter(
        x=candles.index,
        y=candles['SMA20'],
        line=dict(color='lime'),
        name='SMA20'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=candles.index,
        y=candles['SMA50'],
        line=dict(color='orange'),
        name='SMA50'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=candles.index,
        y=support_line,
        line=dict(color='blue', dash='dot'),
        name='Support Trend'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=candles.index,
        y=resist_line,
        line=dict(color='yellow', dash='dot'),
        name='Resist Trend'
    ), row=1, col=1)

    # --- Gráfico de volumen (fila 2)
    fig.add_trace(go.Bar(
        x=candles.index,
        y=candles['Volume'],
        name='Volumen',
        marker_color='gray',
        opacity=0.5
    ), row=2, col=1)

    # Luego agrega estas líneas horizontales en el gráfico Plotly:
    fig.add_trace(go.Scatter(
        x=candles.index,
        y=[pivot] * len(candles),
        line=dict(color='lightblue', dash='dot'),
        name='Pivot'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=candles.index,
        y=[r1] * len(candles),
        line=dict(color='green', dash='dash'),
        name='R1'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=candles.index,
        y=[s1] * len(candles),
        line=dict(color='red', dash='dash'),
        name='S1'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=candles.index,
        y=[r2] * len(candles),
        line=dict(color='darkgreen', dash='dot'),
        name='R2'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=candles.index,
        y=[s2] * len(candles),
        line=dict(color='darkred', dash='dot'),
        name='S2'
    ), row=1, col=1)

    # Volumen (niveles con mayor acumulación)
    for i, level in enumerate(volume_levels):
        fig.add_trace(go.Scatter(
            x=candles.index,
            y=[level] * len(candles),
            line=dict(color='gray', dash='dot'),
            name=f'Volumen Nivel {i+1}'
        ), row=1, col=1)



    # Layout
    fig.update_layout(
        title=f"{ticker}: Indicadores Técnicos + Volumen",
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(t=60, b=50),
        legend=dict(x=0.01, y=0.99)
    )

    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


    st.markdown("""
    **Notas**:
    - Se detectan soportes y resistencias basados en extremos locales (`argrelextrema`) y líneas de tendencia.
    """)
