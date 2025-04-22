# 📈 Plataforma de Inversión Automatizada

Esta es una aplicación desarrollada en **Streamlit** que combina **análisis técnico**, **predicción de precios con LSTM**, **análisis de sentimiento en noticias** y **visualización de indicadores de mercado**. Está pensada para ayudarte a tomar decisiones informadas al invertir en activos como **Bitcoin (BTC)**.

---

## 🚀 Funcionalidades

🔹 **Predicción de Precios**:  
Entrena un modelo LSTM con datos de Yahoo Finance para prever precios futuros.

🔹 **Análisis de Sentimiento**:  
Obtiene noticias del último mes desde NewsAPI y analiza el sentimiento con VADER (NLTK).

🔹 **Indicadores Técnicos**:  
Calcula SMA, RSI, OBV, soportes y resistencias sobre datos recientes del mercado.

🔹 **Noticias y Gráfico Interactivo**:  
Relaciona sentimiento del mercado con precio del BTC en un gráfico con doble eje.

---

## 🛠️ Instalación local

1. Clona este repositorio:
```bash
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta la app:
```bash
streamlit run main.py
```

---

## ☁️ Despliegue en Streamlit Cloud

1. Sube este proyecto a un repositorio de GitHub.
2. Ve a [Streamlit Cloud](https://streamlit.io/cloud) y conecta tu cuenta.
3. Crea una nueva app:
   - **Main file path**: `main.py`
4. (Opcional pero recomendado) Configura tu API key de NewsAPI:
   - Ve a **Settings > Secrets** e ingresa:
     ```toml
     NEWS_API_KEY = "tu_clave_de_newsapi"
     ```
   - Luego en el código usa `os.getenv("NEWS_API_KEY")`.

---

## 🔐 Dependencias principales

- `streamlit`
- `yfinance`
- `tensorflow`
- `scikit-learn`
- `nltk` (con `vader_lexicon`)
- `plotly`
- `newsapi-python`
- `mplfinance`

---

## 📸 Captura

![demo](https://user-images.githubusercontent.com/tu_usuario/demo.gif)

---

## 📩 Licencia

Este proyecto es de código abierto bajo la licencia MIT.

---

> Desarrollado con ❤️ por [tu_nombre]
