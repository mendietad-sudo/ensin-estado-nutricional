# Estado nutricional estimado a partir de características sociales (ENSIN 2015)

Aplicación de Streamlit del Proyecto Integrador de Machine Learning I
(Universidad Externado de Colombia), de David Duarte Mendieta y Kenner David Lobo Plata.

Usa un modelo LDA entrenado con la ENSIN 2015 para estimar la categoría de estado
nutricional más probable de un perfil (delgadez, normal, sobrepeso u obesidad) a partir
de características sociodemográficas, socioeconómicas, del hogar y de salud.

Es una herramienta académica de **priorización poblacional**: no sirve para clasificar
a una persona ni reemplaza medir su peso y su talla.

Ejecutar en local:

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```
