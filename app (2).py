# -*- coding: utf-8 -*-
"""
Despliegue: Predicción de inversión en una tienda de videojuegos

Pasos del despliegue:
1. Cargar el modelo entrenado (modelo, escalador y lista de variables).
2. Capturar los datos nuevos desde la interfaz de Streamlit.
3. Preparar los datos igual que en el entrenamiento (dummies y normalización).
4. Aplicar el modelo y mostrar la predicción.

Ejecutar en local:  streamlit run app.py
"""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de la página (debe ser la primera instrucción de Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predicción de inversión - Videojuegos", page_icon="🎮")

# El .pkl se busca en la misma carpeta que app.py, así funciona igual en tu
# computador y en Streamlit Cloud.
RUTA_MODELO = Path(__file__).parent / "modelo-reg.pkl"


# ---------------------------------------------------------------------------
# 1. Cargar el modelo (una sola vez, no en cada clic)
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    with open(RUTA_MODELO, "rb") as f:
        modelo, min_max_scaler, variables = pickle.load(f)
    return modelo, min_max_scaler, list(variables)


try:
    modelo, min_max_scaler, variables = cargar_modelo()
except FileNotFoundError:
    st.error(f"No se encontró el archivo del modelo: {RUTA_MODELO.name}. "
             "Súbelo a la misma carpeta que app.py.")
    st.stop()


def es_modelo_de_arboles(m):
    """Los árboles (Tree, Random Forest, etc.) no necesitan normalización.
    KNN, Red Neuronal, SVM y Regresión sí."""
    return hasattr(m, "tree_") or hasattr(m, "estimators_")


# ---------------------------------------------------------------------------
# 2. Interfaz gráfica para capturar los datos
# ---------------------------------------------------------------------------
st.title("🎮 Predicción de inversión en una tienda de videojuegos")
st.write("Ingrese las características del cliente y presione **Predecir**.")

# Los valores deben escribirse EXACTAMENTE como aparecen en el dataset de
# entrenamiento (incluidas las comillas simples), porque de ellos salen los
# nombres de las columnas dummies.
with st.form("formulario"):
    col1, col2 = st.columns(2)
    with col1:
        Edad = st.slider("Edad", min_value=14, max_value=52, value=20, step=1)
        videojuego = st.selectbox("Videojuego", [
            "'Mass Effect'", "'Battlefield'", "'Fifa'", "'KOA: Reckoning'",
            "'Crysis'", "'Sim City'", "'Dead Space'", "'F1'"])
        Plataforma = st.selectbox("Plataforma", ["'Play Station'", "'Xbox'", "PC", "Otros"])
    with col2:
        Sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
        Consumidor_habitual = st.selectbox("Consumidor habitual", ["True", "False"])
    predecir = st.form_submit_button("Predecir")

if predecir:
    # Dataframe con los mismos nombres de variables del entrenamiento
    data = pd.DataFrame(
        [[Edad, videojuego, Plataforma, Sexo, Consumidor_habitual]],
        columns=["Edad", "videojuego", "Plataforma", "Sexo", "Consumidor_habitual"],
    )

    # -----------------------------------------------------------------------
    # 3. Preparación de los datos
    # -----------------------------------------------------------------------
    data_preparada = data.copy()

    # En despliegue drop_first=False: con un solo registro, drop_first=True
    # eliminaría justamente la categoría elegida.
    data_preparada = pd.get_dummies(
        data_preparada,
        columns=["videojuego", "Plataforma", "Sexo", "Consumidor_habitual"],
        drop_first=False, dtype=int,
    )

    # Se agregan las columnas faltantes (en 0) y se dejan en el mismo orden
    # del entrenamiento; las que el modelo no conoce se descartan.
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    # Normalización de la edad (solo transform, nunca fit en el despliegue).
    # Solo se aplica a KNN, Red, SVM y Regresión, no a modelos de árboles.
    if min_max_scaler is not None and not es_modelo_de_arboles(modelo):
        data_preparada[["Edad"]] = min_max_scaler.transform(data_preparada[["Edad"]])

    # -----------------------------------------------------------------------
    # 4. Predicción
    # -----------------------------------------------------------------------
    prediccion = float(modelo.predict(data_preparada)[0])
    data["Predicción"] = prediccion

    st.subheader("Resultado")
    st.metric("Inversión estimada", f"{prediccion:,.2f}")
    st.dataframe(data, hide_index=True)

    with st.expander("Ver los datos preparados que recibe el modelo"):
        st.dataframe(data_preparada, hide_index=True)

# Recordar la medida de error del modelo
st.warning("El modelo tiene un error del 21% (MAPE: error porcentual absoluto medio).")
