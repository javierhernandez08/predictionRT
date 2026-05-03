# RT Predictor con Gemini AI

Predicción de Retention Time (RT) en cromatografía utilizando Google Gemini AI.

## 📋 Descripción

Esta aplicación web basada en Streamlit utiliza modelos de lenguaje de Google Gemini para predecir el tiempo de retención (Retention Time) de compuestos químicos a partir de su fórmula SMILES. El sistema aprende patrones de un conjunto de datos de ejemplo y luego aplica ese conocimiento para hacer predicciones.

### Flujo de trabajo:

1. **Carga de datos**: Sube un archivo Excel con columnas de SMILES y RT
2. **Contexto (20%)**: El sistema envía automáticamente el 20% de los datos a Gemini como ejemplo
3. **Predicción**: Eliges una molécula del 80% restante y Gemini predice su RT
4. **Resultados**: Comparación entre RT real y predicho

## 🚀 Características

- Soporte para archivos Excel con múltiples hojas
- Integración con Google Gemini API
- Soporte opcional para subir papers científicos como contexto adicional
- Sistema automático de reintentos ante errores de rate limiting
- Interfaz simple y limpia

## 📦 Requisitos

- Python 3.8 o superior
- Cuenta de Google AI Studio con API key de Gemini
- Dependencias listadas en `requirements.txt`

## 🔧 Instalación

1. **Clona el repositorio:**
```bash
git clone https://github.com/javierhernandez08/predictionRT.git
cd predictionRT
