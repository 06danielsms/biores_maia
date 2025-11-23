FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copiar archivos de configuración de dependencias primero (para cache)
COPY pyproject.toml /app/
COPY streamlit/requirements.txt /app/streamlit/requirements.txt

# Copiar todo el proyecto biores_maia
COPY . /app/

# Instalar el proyecto con uv
RUN uv pip install --system --no-cache -e .

# Crear directorios necesarios
RUN mkdir -p /app/data /app/metrics

# Exponer el puerto de Streamlit
EXPOSE 8501

# Cambiar al directorio de streamlit
WORKDIR /app/streamlit

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
