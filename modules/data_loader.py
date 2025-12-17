# modules/data_loader.py

import streamlit as st
import yfinance as yf
import pandas as pd

@st.cache_data(ttl=3600)
def load_market_data(ticker: str, period="5y"):
    t = yf.Ticker(ticker)
    df = t.history(period=period, auto_adjust=True)

    if df is None or df.empty:
        return None

    df.index = pd.to_datetime(df.index)
    return df