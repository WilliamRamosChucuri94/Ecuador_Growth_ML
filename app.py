import streamlit as st
import pandas as pd
import joblib

# =====================================================
# 1. CONFIGURACIÓN DE LA APLICACIÓN
# =====================================================

st.set_page_config(
    page_title="Ecuador Growth ML",
    page_icon="📈",
    layout="wide"
)

st.title("Ecuador Growth ML")
st.subheader("Nowcasting y proyecciones del PIB de Ecuador")

st.info(
    "Aplicación académica de Machine Learning. "
    "Todos los indicadores utilizados son simulados."
)

# =====================================================
# 2. CARGAR EL MODELO ENTRENADO
# =====================================================

modelo = joblib.load("modelo_pib.joblib")

# Recuperamos las variables utilizadas en Colab
FEATURES = modelo.named_steps[
    "preprocesamiento"
].transformers_[0][2]

st.success("Modelo Random Forest cargado correctamente")

# =====================================================
# 3. INFORMACIÓN DEL MODELO
# =====================================================

st.header("Modelo predictivo")

st.write("Algoritmo: Random Forest Regressor")

st.write("Variables utilizadas:", FEATURES)

# =====================================================
# 4. FORMAS DE GENERAR PREDICCIONES
# =====================================================
# =====================================================
# HISTORIAL DE PREDICCIONES
# =====================================================

if "historial" not in st.session_state:
    st.session_state.historial = []

tab1, tab2 = st.tabs([
    "Ingreso manual",
    "Cargar archivo CSV"
])

# =====================================================
# 5. INGRESO MANUAL DE INDICADORES
# =====================================================

with tab1:

    st.subheader("Ingresar indicadores económicos")

    st.write(
        "Introduzca los valores correspondientes "
        "al trimestre que desea estimar."
    )

    valores = {}

    for variable in FEATURES:

        valores[variable] = st.number_input(
            variable,
            value=0.0,
            format="%.3f"
        )

    if st.button("Generar estimación"):

        datos_nuevos = pd.DataFrame([valores])

        prediccion = modelo.predict(datos_nuevos)

        st.metric(
            "Crecimiento estimado del PIB",
            f"{prediccion[0]:.3f}%"
        )

# =====================================================
# 6. CARGAR Y PROCESAR NUEVOS INDICADORES
# =====================================================

with tab2:

    st.subheader("Actualización automática de estimaciones")

    archivo = st.file_uploader(
        "Seleccione un archivo CSV",
        type=["csv"]
    )

    if archivo is not None:

        # Leemos los datos nuevos
        nuevos_datos = pd.read_csv(archivo)

        st.write("Información recibida:")
        st.dataframe(nuevos_datos)

        # Comprobamos las variables del modelo
        faltantes = [
            variable
            for variable in FEATURES
            if variable not in nuevos_datos.columns
        ]

        if faltantes:

            st.error(
                "Faltan variables: " + ", ".join(faltantes)
            )

        else:

            # Seleccionamos los predictores
            X_nuevo = nuevos_datos[FEATURES].copy()

            # Verificamos que sean numéricos
            X_nuevo = X_nuevo.apply(
                pd.to_numeric, errors="coerce"
            )

            # El pipeline genera la predicción
            predicciones = modelo.predict(X_nuevo)

            # Construimos la tabla de resultados
            resultados = nuevos_datos.copy()
            resultados["PIB_estimado"] = predicciones

            st.success("Predicciones generadas automáticamente")

            st.dataframe(resultados)

            # Guardamos el resultado en el historial
            if st.button("Registrar estimaciones"):

                registro = pd.DataFrame({
                    "trimestre": nuevos_datos["trimestre"],
                    "escenario": (
                        nuevos_datos["escenario"]
                        if "escenario" in nuevos_datos.columns
                        else "Sin escenario"
                    ),
                    "PIB_estimado": predicciones
                })

                st.session_state.historial.extend(
                    registro.to_dict("records")
                )

                st.success("Estimaciones registradas")

            # Descargamos las predicciones actuales
            st.download_button(
                "Descargar predicciones",
                data=resultados.to_csv(index=False),
                file_name="predicciones_pib.csv",
                mime="text/csv"
            )

    # =================================================
    # 7. HISTORIAL DE PREDICCIONES
    # =================================================

    st.subheader("Historial de estimaciones")

    if st.session_state.historial:

        historial = pd.DataFrame(
            st.session_state.historial
        )

        st.dataframe(historial)

        st.download_button(
            "Descargar historial",
            data=historial.to_csv(index=False),
            file_name="historial_predicciones.csv",
            mime="text/csv"
        )

    else:

        st.info("Todavía no existen predicciones registradas.")
