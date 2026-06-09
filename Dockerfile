# 1. Imagen base oficial y ligera de Python
FROM python:3.10-slim

# 2. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalamos herramientas necesarias por si acaso (limpiando caché al final)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiamos los requerimientos primero para aprovechar la caché de Docker
COPY requirements.txt .

# 5. Instalamos todas las librerías de Python necesarias
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiamos tu código, tu Excel y tu README al contenedor
COPY . .

# 7. Exponemos el puerto por el que escucha Streamlit por defecto
EXPOSE 8501

# 8. Comando para arrancar tu app apuntando a tu archivo exacto
CMD ["streamlit", "run", "streamlit_codigo.py", "--server.port=8501", "--server.address=0.0.0.0"]
