# BioRes MAIA

Pipeline de preprocesamiento, entrenamiento y validación para clasificación de textos médicos (PLS vs NO_PLS).

## 📋 Requisitos

- Python 3.10+
- Docker & Docker Compose (para despliegue con contenedores)
- UV (opcional, para instalación rápida de dependencias)

## 🚀 Instalación

### Opción 1: Instalación local con UV (recomendado)

```bash
# Instalar UV si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar el repositorio
git clone https://github.com/06danielsms/biores_maia.git
cd biores_maia

# Instalar el proyecto y todas sus dependencias
uv pip install -e .
```

### Opción 2: Instalación local con pip

```bash
# Clonar el repositorio
git clone https://github.com/06danielsms/biores_maia.git
cd biores_maia

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar el proyecto
pip install -e .
```

## 🐳 Despliegue con Docker

### Construcción y ejecución

```bash
# Construir y levantar el servicio Streamlit
docker-compose up --build

# O en modo detached (segundo plano)
docker-compose up -d --build
```

### Acceso a la aplicación

Una vez levantado, abre tu navegador en: **http://localhost:8501**

### Comandos útiles

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Detener el servicio
docker-compose down

# Reconstruir la imagen (después de cambiar dependencias)
docker-compose up --build --force-recreate

# Ver estado del contenedor
docker-compose ps
```

## 📊 Uso

### 1. Preprocesamiento

Ejecuta el script de limpieza de textos:

```bash
python scripts/clean_en.py --config config/config.yaml
```

### 2. Aplicación Streamlit

La aplicación incluye tres secciones principales:

- **Preprocesamiento**: Limpieza y normalización de textos médicos
- **Entrenamiento**: Entrenamiento de clasificadores con visualización de métricas
- **Validación**: Evaluación con ROUGE, BLEU y análisis de resultados

### 3. Notebooks

Explora los notebooks en `jupyter/` para análisis más detallados:

- `project.ipynb`: Pipeline completo del proyecto
- `CodeLlama_7B_Finetuning.ipynb`: Fine-tuning con CodeLlama
- `Qwen3_8B_Finetuning.ipynb`: Fine-tuning con Qwen3

## 📁 Estructura del Proyecto

```
biores_maia/
├── streamlit/          # Aplicación web Streamlit
├── scripts/            # Scripts de preprocesamiento y análisis
├── config/             # Archivos de configuración
├── data/               # Datasets (montados como volumen en Docker)
├── metrics/            # Resultados de métricas (montados como volumen)
├── jupyter/            # Notebooks de análisis
├── pyproject.toml      # Dependencias del proyecto
├── Dockerfile          # Imagen Docker
└── docker-compose.yml  # Orquestación de servicios
```

## 🔧 Configuración

Edita `config/config.yaml` para ajustar:

- Rutas de datos de entrada/salida
- Parámetros de preprocesamiento
- Configuración de métricas y visualizaciones

## 🛠️ Desarrollo

### Instalación de dependencias adicionales

```bash
# Con UV
uv pip install <paquete>

# Con pip
pip install <paquete>
```

### Hot-reload en Docker

El `docker-compose.yml` monta el directorio `streamlit/` como volumen, permitiendo hot-reload durante el desarrollo. Los cambios en el código se reflejan automáticamente sin necesidad de reconstruir la imagen.



