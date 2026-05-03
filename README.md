# RT Predictor con Gemini AI

Prediccion de Retention Time (RT) en cromatografia utilizando Google Gemini AI.

## Descripcion

Esta aplicacion web basada en Streamlit utiliza modelos de lenguaje de Google Gemini para predecir el tiempo de retencion (Retention Time) de compuestos quimicos a partir de su formula SMILES. El sistema aprende patrones de un conjunto de datos de ejemplo y luego aplica ese conocimiento para hacer predicciones.

### Flujo de trabajo:

1. Carga de datos: Sube un archivo Excel con columnas de SMILES y RT
2. Contexto (20%): El sistema envia automaticamente el 20% de los datos a Gemini como ejemplo
3. Prediccion: Eliges una molecula del 80% restante y Gemini predice su RT
4. Resultados: Comparacion entre RT real y predicho

## Caracteristicas

- Soporte para archivos Excel con multiples hojas
- Integracion con Google Gemini API
- Soporte opcional para subir papers cientificos como contexto adicional
- Sistema automatico de reintentos ante errores de rate limiting
- Interfaz simple y limpia

## Requisitos

- Python 3.8 o superior
- Cuenta de Google AI Studio con API key de Gemini
- Dependencias listadas en requirements.txt

## Instalacion

1. Clona el repositorio:
```bash
git clone https://github.com/javierhernandez08/predictionRT.git
cd predictionRT
