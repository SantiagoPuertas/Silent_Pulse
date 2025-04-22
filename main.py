import streamlit as st

from modules.price_prediction import show_price_prediction
from modules.sentiment_analysis import show_sentiment_analysis
from modules.technical_indicators import show_technical_indicators
from modules.news_feed import show_news_feed

st.set_page_config(page_title="Silent Pulse", layout="wide")

def main():
    st.title("Silent Pulse Patrimonio")

    # Inicializar el estado si no existe
    if "selected_section" not in st.session_state:
        st.session_state.selected_section = None
    if "selected_subsection" not in st.session_state:
        st.session_state.selected_subsection = None

    # Sidebar con secciones y subsecciones
    with st.sidebar:
        st.image("static/SilentPulsePatrimonio_logo.png", width=200)  # Tu logo

        # Sección: Predicción
        with st.expander("📈 Predicción"):
            if st.button("Predicción de precios"):
                st.session_state.selected_section = "Predicción"
                st.session_state.selected_subsection = "Predicción de precios"

        # Sección: Sentimiento
        with st.expander("💬 Sentimiento"):
            if st.button("Análisis de sentimiento"):
                st.session_state.selected_section = "Sentimiento"
                st.session_state.selected_subsection = "Análisis de sentimiento"
            if st.button("Sentimiento de noticias"):
                st.session_state.selected_section = "Sentimiento"
                st.session_state.selected_subsection = "Sentimiento de noticias"

        # Sección: Indicadores Técnicos
        with st.expander("📉 Indicadores Técnicos"):
            if st.button("Indicadores técnicos"):
                st.session_state.selected_section = "Indicadores Técnicos"
                st.session_state.selected_subsection = "Indicadores técnicos"

    # Renderizado principal según la selección
    if st.session_state.selected_section == "Predicción":
        if st.session_state.selected_subsection == "Predicción de precios":
            show_price_prediction()
    elif st.session_state.selected_section == "Sentimiento":
        if st.session_state.selected_subsection == "Análisis de sentimiento":
            show_sentiment_analysis()
        elif st.session_state.selected_subsection == "Sentimiento de noticias":
            show_news_feed()
    elif st.session_state.selected_section == "Indicadores Técnicos":
        if st.session_state.selected_subsection == "Indicadores técnicos":
            show_technical_indicators()
    else:
        st.info("Selecciona una sección desde el menú lateral.")

if __name__ == "__main__":
    main()
