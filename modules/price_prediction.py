import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow import keras
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

@st.cache_data
def load_data(ticker: str, start_date: str) -> pd.DataFrame:

    df = yf.download(ticker, start=start_date)

    # 🔧 Aplanar columnas si tienen MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)


    return df

@st.cache_data
def preprocess_data(df: pd.DataFrame, n_past=14, n_future=1):
    if 'Adj Close' in df.columns:
        df.drop(columns=['Adj Close'], inplace=True)

    # Ordenamos las columnas explícitamente
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.warning(f"Falta la columna '{col}'. Columnas reales: {df.columns.tolist()}")
            st.stop()
    df = df[required_cols]

    df.index = pd.to_datetime(df.index)
    df_for_training = df.astype(float)

    scaler = StandardScaler()
    scaler.fit(df_for_training)
    df_scaled = scaler.transform(df_for_training)

    trainX, trainY = [], []
    for i in range(n_past, len(df_scaled) - n_future + 1):
        trainX.append(df_scaled[i - n_past:i, :])
        trainY.append(df_scaled[i + n_future - 1:i + n_future, 0])  # 0 = 'Open'

    return np.array(trainX), np.array(trainY), scaler, df_for_training

@st.cache_resource
def train_model(trainX, trainY, lr=0.02, epochs=20, batch_size=20):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    return model

def show_price_prediction():
    st.header("Predicción de Precios con LSTM")

    #ticker = st.sidebar.text_input("Ticker (YFinance)", "BTC-USD").upper()

    #_______________
    with st.sidebar:
        st.markdown("### ⚙️ Parámetros del modelo")

        modo_seleccion = st.radio("Modo:", ["Predefinido", "Escribir manualmente"])
        tickers = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"]
        
        if modo_seleccion == "Predefinido":
            ticker = st.selectbox("Activo", tickers)
        else:
            ticker = st.text_input("Ticker manual", "BTC-USD").upper()

        start_date = st.text_input("Fecha inicio", "2024-01-01")
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.02, step=0.01)
        epochs = st.slider("Épocas", 5, 100, 30)
        batch_size = st.slider("Batch size", 10, 100, 20)
        n_days_for_prediction = st.slider("Días a predecir", 5, 30, 15)

  #______________


    df = load_data(ticker, start_date)
    if df.empty:
        st.warning("No se encontraron datos para este ticker o rango de fechas.")
        return

    st.success("Datos descargados correctamente")

    # Mostrar ambos
    st.write(f"Vista previa de los datos : {ticker} ")
    st.dataframe(df.tail(5).round(0))

    trainX, trainY, scaler, df_for_training = preprocess_data(df)
    model = train_model(trainX, trainY, lr=learning_rate, epochs=epochs, batch_size=batch_size)

    last_date = df.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days_for_prediction)

    prediction = model.predict(trainX[-n_days_for_prediction:])

    st.success("✅ Predicción completada.")

    n_features = df_for_training.shape[1]

    # Construimos matriz para desescalar
    prediction_padded = np.zeros((prediction.shape[0], n_features))
    prediction_padded[:, 0] = prediction.ravel()  # solo columna 'Open'

    y_pred_future = scaler.inverse_transform(prediction_padded)[:, 0]

    df_forecast = pd.DataFrame({
        'Date': future_dates,
        'Open': y_pred_future
    })


    # Asegurarse de que 'df' tiene el índice como datetime
    df.index = pd.to_datetime(df.index)

    # Convertir la columna 'Date' de df_forecast a índice
    df_forecast = df_forecast.set_index('Date')
    df_forecast.index = pd.to_datetime(df_forecast.index)

    # 📊 Gráfico interactivo con ambas series
    fig = go.Figure()

    from plotly.subplots import make_subplots

    # Crear figura con 2 filas: precios y volumen
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{ticker}: Precio de Apertura - Real vs. Pronóstico", "Volumen")
    )

    # --- Gráfico de precios (fila 1)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Open'],
        mode='lines',
        name='Open (Real)',
        line=dict(color='royalblue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_forecast.index,
        y=df_forecast['Open'],
        mode='lines',
        name='Open (Pronóstico)',
        line=dict(color='orangered')
    ), row=1, col=1)

    # --- Gráfico de volumen (fila 2)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volumen',
        marker_color='gray',
        opacity=0.5
    ), row=2, col=1)

    # Layout del gráfico
    fig.update_layout(
        template='plotly_dark',
        height=700,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=60, b=50)
    )

    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)

    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("📅 Valores pronosticados")
    st.dataframe(df_forecast.round(0))

    st.write("Último valor real (Open):", int(df['Open'].iloc[-1]))
    st.write(f"Primer valor pronosticado: {int(df_forecast['Open'].iloc[0])}")
    
