import joblib
import pandas as pd
import sklearn
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Estado nutricional - ENSIN",
    page_icon=":hospital:",
    layout="wide"
)

BASE = Path(__file__).parent
ETIQUETAS = {1: "Delgadez", 2: "Normal", 3: "Sobrepeso", 4: "Obesidad"}

st.title("Estado nutricional estimado a partir de características sociales")
st.subheader("Proyecto Integrador · ENSIN 2015 · Colombia")
st.markdown("""
Esta herramienta es académica y está pensada para **priorizar grupos de población**
(por ejemplo, decidir dónde enfocar un programa de prevención del exceso de peso),
no para clasificar a una persona. Para saber el estado nutricional de alguien hay que
medir su peso y su talla.
""")


# El pipeline guardado ya trae el preprocesamiento, así que solo hay que cargarlo
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(BASE / "modelo_final_estado_nutricional.joblib")
    metadata = joblib.load(BASE / "metadata_modelo.joblib")
    return modelo, metadata


try:
    modelo, metadata = cargar_modelo()
except FileNotFoundError:
    st.error("No se encontraron 'modelo_final_estado_nutricional.joblib' y 'metadata_modelo.joblib'. "
             "Hay que correr el notebook de la entrega final y dejar ambos archivos junto a esta aplicación.")
    st.stop()

if sklearn.__version__ != metadata["sklearn_version"]:
    st.warning(f"El modelo se entrenó con scikit-learn {metadata['sklearn_version']} y aquí está instalado "
               f"{sklearn.__version__}. Instale la versión de requirements.txt para obtener los mismos resultados.")

st.success(f"Modelo cargado: {metadata['nombre_modelo']}")
with st.expander("¿Qué tan bien funciona? (conjunto de prueba)"):
    m = metadata["metricas_test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro-F1", f"{m['macro_f1']:.3f}", help="Línea base (siempre 'normal'): " f"{m.get('macro_f1_base', 0):.3f}")
    c2.metric("Accuracy", f"{m['accuracy']:.3f}")
    c3.metric("Aciertos o error de 1 categoría", f"{m.get('hasta_1_categoria', 0):.1%}")
    c4.metric("Errores de 2 o más categorías", f"{m.get('errores_graves', 0):.1%}")
    if "recall_por_clase" in m:
        st.markdown("Porcentaje de personas de cada categoría que el modelo reconoce:")
        st.dataframe(pd.DataFrame({"Categoría": list(m["recall_por_clase"].keys()),
                                   "Detectadas": [f"{v:.0%}" for v in m["recall_por_clase"].values()]}),
                     hide_index=True)

# Solo pedimos las variables que más pesan; las demás toman el valor típico del
# entrenamiento (mediana o moda), que es lo mismo que haría el imputador
st.header("Características del perfil")

valores_defecto = metadata["valores_defecto"]
entradas = {}
col_izq, col_der = st.columns(2)

for i, col in enumerate(metadata["vars_formulario"]):
    with (col_izq if i % 2 == 0 else col_der):
        if col in metadata["cols_categoricas"]:
            opciones = metadata["opciones_categoricas"][col]
            etiquetas = metadata["etiquetas_valores"][col]
            entradas[col] = st.selectbox(col, opciones, index=opciones.index(valores_defecto[col]),
                                         format_func=lambda v, e=etiquetas: e.get(v, str(v)), key=col)
        else:
            minimo, maximo = metadata["rangos_numericos"][col]
            valor = min(max(float(valores_defecto[col]), minimo), maximo)
            entradas[col] = st.number_input(col, min_value=minimo, max_value=maximo, value=valor, key=col)

restantes = [c for c in metadata["cols_final"] if c not in entradas]
st.caption(f"{len(restantes)} variables más se completan con valores típicos del entrenamiento.")
with st.expander("Ver esas variables"):
    st.dataframe(pd.DataFrame({
        "Variable": restantes,
        "Valor usado": [metadata["etiquetas_valores"].get(c, {}).get(valores_defecto[c], str(valores_defecto[c]))
                        for c in restantes]}),
        hide_index=True)

if st.button("Estimar categoría", type="primary"):
    fila = dict(valores_defecto)
    fila.update(entradas)
    # mismas columnas y en el mismo orden que en el entrenamiento
    df_input = pd.DataFrame([fila])[metadata["cols_final"]]

    prediccion = int(modelo.predict(df_input)[0])
    st.header("Resultado")
    st.markdown(f"### Categoría más probable para este perfil: **{ETIQUETAS[prediccion]}**")

    recall = metadata["metricas_test"].get("recall_por_clase", {})
    if ETIQUETAS[prediccion] in recall:
        st.write(f"En el conjunto de prueba, el modelo reconoció al {recall[ETIQUETAS[prediccion]]:.0%} "
                 f"de las personas que realmente estaban en {ETIQUETAS[prediccion].lower()}.")

    if hasattr(modelo, "predict_proba"):
        puntajes = modelo.predict_proba(df_input)[0]
        df_p = pd.DataFrame({"Clase": [ETIQUETAS[int(c)] for c in modelo.classes_], "Puntaje": puntajes})
        st.subheader("Puntaje por clase (no calibrado, sirve para comparar, no es un riesgo)")
        st.bar_chart(df_p.set_index("Clase"))

    st.caption("El resultado describe asociaciones observadas en la muestra de la ENSIN 2015; no es un "
               "diagnóstico ni reemplaza la medición de peso y talla.")

st.markdown("---")
st.markdown("""
**Proyecto Integrador - Machine Learning I** · Universidad Externado de Colombia
David Duarte Mendieta y Kenner David Lobo Plata
Basado en: Okwechime et al. (2015), *PLOS ONE* 10(12): e0145781.
""")
