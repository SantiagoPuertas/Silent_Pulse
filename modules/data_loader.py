# modules/data_loader.py

import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data(ttl=3600)
def load_market_data(ticker: str, start_date: str):
    df = yf.download(
        ticker,
        start=start_date,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return None

    # Aplanar columnas si Yahoo devuelve MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    return df
