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
# 6. CARGAR NUEVOS INDICADORES DESDE CSV
# =====================================================

with tab2:

    st.subheader("Cargar nuevos indicadores económicos")

    archivo = st.file_uploader(
        "Seleccione un archivo CSV",
        type=["csv"]
    )

    if archivo is not None:

        nuevos_datos = pd.read_csv(archivo)

        st.write("Información recibida:")
        st.dataframe(nuevos_datos)

        # Verificamos las columnas necesarias
        faltantes = [
            variable
            for variable in FEATURES
            if variable not in nuevos_datos.columns
        ]

        if faltantes:

            st.error(
                "Faltan las siguientes variables: "
                + ", ".join(faltantes)
            )

        else:

            if st.button("Predecir con CSV"):

                X_nuevo = nuevos_datos[FEATURES]

                predicciones = modelo.predict(X_nuevo)

                resultados = nuevos_datos.copy()

                resultados["PIB_estimado"] = predicciones

                st.subheader("Resultados de la estimación")

                st.dataframe(resultados)

                st.download_button(
                    "Descargar resultados",
                    data=resultados.to_csv(index=False),
                    file_name="estimaciones_pib.csv",
                    mime="text/csv"
                )
