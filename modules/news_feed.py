# modules/news_feed.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from datetime import date, timedelta, datetime
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from modules.data_loader import load_market_data

import warnings
warnings.filterwarnings("ignore")

# Asegúrate de tener descargado el lexicón VADER
nltk.download('vader_lexicon', quiet=True)

# -------------------------------------------------------------------
#  1) Funciones de ayuda
# -------------------------------------------------------------------
def get_sources(api_key, category=None):
    """
    Devuelve la lista de 'id' de fuentes de noticias (en inglés)
    de la categoría dada. Si category es None, devuelve todas las posibles.
    """
    newsapi = NewsApiClient(api_key=api_key)
    sources = newsapi.get_sources()
    if category is not None:
        rez = [source['id'] for source in sources['sources'] 
               if source['category'].lower() == category.lower() and source['language'] == 'en']
    else:
        rez = [source['id'] for source in sources['sources'] if source['language'] == 'en']
    return rez

@st.cache_data
def get_articles_sentiments(api_key, keywrd, my_date, sources_list=None):
    """
    Obtiene las noticias de la fecha dada (my_date) relacionadas con 'keywrd',
    calcula el sentiment con VADER y devuelve un DataFrame con las columnas:
    ['Sentiment','URL','Title','Description'].
    """
    newsapi = NewsApiClient(api_key=api_key)
    # Convertimos 'my_date' a datetime si es string
    if isinstance(my_date, str):
        my_date = datetime.strptime(my_date, '%d-%b-%Y')

    if sources_list:
        articles = newsapi.get_everything(
            q=keywrd,
            from_param=my_date.isoformat(),
            to=(my_date + timedelta(days=1)).isoformat(),
            language="en",
            sources=",".join(sources_list),
            sort_by="relevancy",
            page_size=100
        )
    else:
        articles = newsapi.get_everything(
            q=keywrd,
            from_param=my_date.isoformat(),
            to=(my_date + timedelta(days=1)).isoformat(),
            language="en",
            sort_by="relevancy",
            page_size=100
        )

    sia = SentimentIntensityAnalyzer()
    date_sentiments_list = []
    seen_titles = set()

    for article in articles.get('articles', []):
        # Evitamos duplicados basados en el título
        if article['title'] not in seen_titles:
            seen_titles.add(article['title'])
            title = str(article.get('title', ''))
            desc = str(article.get('description', ''))
            article_content = f"{title}. {desc}"
            # Sentimiento
            sentiment_score = sia.polarity_scores(article_content)['compound']
            date_sentiments_list.append([
                sentiment_score,
                article.get('url', ''),
                title,
                desc
            ])

    df_res = pd.DataFrame(date_sentiments_list, columns=['Sentiment', 'URL', 'Title', 'Description'])
    return df_res



@st.cache_data
def load_news_data(api_key, keyword_all='Apple', keyword_business='Technology'):
    today = date.today()
    start_date = today - timedelta(days=28)
    cache_file = "static/cached_news_data.csv"

    # Cargar si existe
    if os.path.exists(cache_file):
        existing_df = pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
        # Filtrar fechas que ya están
        last_cached_date = existing_df.index.max().date()
        if last_cached_date >= today:
            return existing_df  # Ya está actualizado

        # Continuar desde el último día +1
        start_date = last_cached_date + timedelta(days=1)
    else:
        existing_df = pd.DataFrame()

    # Descargar fuentes si hace falta
    business_sources = get_sources(api_key, 'business')

    sentiment_all_score = []
    sentiment_business_score = []
    date_list = []

    current_day = start_date
    while current_day <= today:
        df_all = get_articles_sentiments(api_key, keyword_all, current_day)
        df_bus = get_articles_sentiments(api_key, keyword_business, current_day, sources_list=business_sources)

        sentiment_all_score.append(df_all['Sentiment'].mean())
        sentiment_business_score.append(df_bus['Sentiment'].mean())
        date_list.append(current_day)

        current_day += timedelta(days=1)

    if not date_list:
        return existing_df if not existing_df.empty else None

    new_sentiments = pd.DataFrame({
        'Date': pd.to_datetime(date_list),
        'All_sources_sentiment': sentiment_all_score,
        'Business_sources_sentiment': sentiment_business_score
    }).set_index('Date')

    df_price = load_market_data("AAPL", start_date)

    # 🔧 Solución al error: quitar MultiIndex si existe
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)

    df_price = df_price[['Close']].copy()
    df_price.columns = ['closing_Price']

    merged_new = new_sentiments.join(df_price, how='inner').dropna()

    # Juntar con lo que ya había
    full_df = pd.concat([existing_df, merged_new])
    full_df = full_df[~full_df.index.duplicated(keep='last')]  # evitar duplicados

    # Guardar
    full_df.to_csv(cache_file)

    print(f"Último día: {current_day - timedelta(days=1)}")
    print("df_all rows:", len(df_all))
    print("df_bus rows:", len(df_bus))

    return full_df, df_all, df_bus, business_sources



# -------------------------------------------------------------------
#  2) Función principal de Streamlit
# -------------------------------------------------------------------
def show_news_feed():
    st.header("Noticias y Análisis de Sentimiento")

    # Parámetros interactivos: se asume que el usuario tiene una API key
    st.write("Para usar esta sección necesitas una API key de NewsAPI.")
    api_key = st.sidebar.text_input("NewsAPI Key", type="password", value="")

    if not api_key:
        st.warning("Introduce tu clave de NewsAPI para obtener datos.")
        return

    # Palabras clave configurables
    keyword_all = st.sidebar.text_input("Keyword para noticias generales", "Crypto")
    keyword_business = st.sidebar.text_input("Keyword para fuentes de negocio", "Bitcoin")

    st.write("Obteniendo noticias y calculando sentimiento (últimos 28 días aprox.)")


    # Cargamos y procesamos datos (cacheado)
    #merged_df = load_news_data(api_key, keyword_all, keyword_business)
    merged_df, df_all, df_bus, business_sources = load_news_data(api_key, keyword_all, keyword_business)

    st.write("Noticias de business:")
    st.dataframe(df_bus)
    st.write("Noticias generales:")
    st.dataframe(df_all[['Title', 'Sentiment']])

    if merged_df is None or merged_df.empty:
        st.warning("No se pudieron obtener datos de noticias o precios. Revisa tu API key o rango de fechas.")
        return


    # Mostramos un vistazo del DataFrame
    st.subheader("DataFrame combinando Sentimiento y Precio")
    st.dataframe(merged_df.tail(10))

    # ---------------------------------------------------
    # Gráfico interactivo con dos ejes Y
    # ---------------------------------------------------

    # Gráfico con dos filas: precio/sentimiento y barras de sentimiento
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{}]]
    )

    # --- Fila 1: Precio y líneas de sentimiento
    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df['closing_Price'],
            mode='lines',
            name='Closing Price',
            line=dict(color='gold', width=3)
        ),
        row=1, col=1,
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df['All_sources_sentiment'],
            mode='lines+markers',
            name='All Sources Sentiment',
        ),
        row=1, col=1,
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=merged_df.index,
            y=merged_df['Business_sources_sentiment'],
            mode='lines+markers',
            name='Business Sources Sentiment',
        ),
        row=1, col=1,
        secondary_y=True
    )

    # --- Fila 2: Barras de sentimiento general
    colors = ['green' if s >= 0 else 'red' for s in merged_df['All_sources_sentiment']]
    fig.add_trace(
        go.Bar(
            x=merged_df.index,
            y=merged_df['All_sources_sentiment'],
            marker_color=colors,
            name='Sentimiento (Barras)'
        ),
        row=2, col=1
    )

    # --- Ajustes de layout
    fig.update_layout(
        title="Sentimiento vs. Precio (últimas 4 semanas)",
        hovermode='x unified',
        height=700
    )
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="Precio (USD)", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Score Diario", row=2, col=1)

    # Mostrar el gráfico
    st.plotly_chart(fig, use_container_width=True)


    # ---------------------------------------------------
    # Correlación
    # ---------------------------------------------------
    st.subheader("Correlación entre Sentimiento y Precio")
    corr = merged_df[['All_sources_sentiment','Business_sources_sentiment','closing_Price']].corr()
    st.write(corr)

    st.markdown("""
    **Interpretación**:  
    - Valores cercanos a 1 implican correlación positiva.  
    - Valores cercanos a -1 implican correlación negativa.  
    - Valores cercanos a 0 implican poca correlación.
    """)

    st.write("¡Listo! Aquí tienes una vista rápida de los sentimientos en noticias y el precio de.")
    st.write("Puedes ajustar los parámetros en la barra lateral para ver cómo afectan a los resultados.")