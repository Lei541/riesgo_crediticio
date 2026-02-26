# Imagen base simple de Python 3.10
FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos requirements
COPY requirements.txt .

# Instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Exponemos el puerto
EXPOSE 8000

# Comando para levantar la API
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]