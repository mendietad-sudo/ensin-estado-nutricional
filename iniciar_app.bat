@echo off
REM Doble clic para instalar las librerias (la primera vez) y abrir la aplicacion
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo No se encontro Python.
    echo Instalelo desde https://www.python.org/downloads/ marcando "Add Python to PATH" y vuelva a abrir este archivo.
    pause
    exit /b
)

echo Instalando librerias (solo tarda la primera vez)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo No se pudieron instalar las versiones exactas; probando con versiones compatibles...
    python -m pip install scikit-learn==1.9.1 pandas joblib streamlit
)

echo Abriendo la aplicacion en el navegador (para cerrarla, cierre esta ventana)...
python -m streamlit run app_streamlit.py
pause
