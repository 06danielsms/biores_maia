# BIORES Maia

Plataforma integral para la ingesta, traducción, análisis y generación de resúmenes biomédicos en español coloquial. Este repositorio unifica los artefactos de ciencia de datos con el nuevo stack de aplicaciones (FastAPI + React) y herramientas de soporte (MongoDB, MLflow).

## Arquitectura

```
.
├── backend/                # API FastAPI, configuración y dependencias
│   ├── app/                # Código fuente de la API
│   ├── tests/              # Pruebas automáticas (pytest)
│   ├── Dockerfile          # Imagen Python 3.12 + Uvicorn
│   └── requirements.txt
├── frontend/               # SPA React + Ant Design (Vite)
│   ├── src/                # Componentes, páginas y hooks
│   ├── Dockerfile          # Imagen Node 22 para desarrollo
│   └── package.json
├── data/, scripts/, results/  # Artefactos existentes de pipelines y análisis
├── seed/                  # Datos de ejemplo para inicializar Mongo rápidamente
├── docker-compose.yml     # Orquestación completa del stack
├── docs/                  # Documentación técnica adicional
└── README.md
```

Componentes principales:
- **Backend (FastAPI)**: expone endpoints REST versionados, maneja configuración vía `pydantic-settings` y se integra con MongoDB, MLflow y S3.
- **Frontend (React + AntD)**: panel operativo inspirado en el diseño Figma con vistas para KPIs, corpus, traducciones, resúmenes, evaluación y operaciones.
- **Infraestructura auxiliar**: MongoDB 7 para persistencia documental y MLflow 2.14 para experiment tracking. Listos para integrarse con pipelines DVC/DAGs existentes.

## Variables de entorno

El backend utiliza un archivo `.env`. Parte de los valores esperados se encuentran en `backend/.env.example`:

```
APP_NAME=Biores Maia API
ENVIRONMENT=local
API_DEBUG=true
MONGO_DSN=mongodb://mongo:27017
MONGO_DATABASE=health_literacy_db
MLFLOW_TRACKING_URI=http://mlflow:5000
S3_BUCKET_RESULTS=biores-maia-results
CORS_ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:5173","http://localhost:5174"]
```

Para personalizar la configuración cree un archivo `backend/.env` y ajuste los valores sensibles (credenciales S3, secretos, URLs administradas, etc.).
Consulte `docs/configuration.md` para una guía detallada de credenciales y buenas prácticas.

El frontend toma sus endpoints desde variables `VITE_*` definidas en `docker-compose.yml`. Para desarrollo local sin Docker puede crear un archivo `frontend/.env` con:

```
VITE_API_URL=http://localhost:8001/api/v1
VITE_ENVIRONMENT=local
VITE_MLFLOW_URL=http://localhost:5500
```

## Requisitos

- Docker y Docker Compose v2
- Opcional: Python 3.12, Node.js ≥ 20 si desea ejecutar sin contenedores

## Puesta en marcha con Docker

```bash
# 1. Construir y levantar todo el stack
docker compose up --build

# 2. Backend disponible en http://localhost:8001/docs
#    Frontend disponible en http://localhost:5174
#    MongoDB en mongodb://localhost:27017
#    Redis en redis://localhost:6379
#    MLflow UI en http://localhost:5500
```

El backend utiliza `uvicorn --reload` con montaje del código local, por lo que los cambios se reflejan en caliente. El frontend ejecuta `npm run dev -- --host 0.0.0.0` dentro del contenedor para habilitar hot-reload desde la máquina anfitriona.

Para detener los servicios:

```bash
docker compose down
```

## Pruebas

```bash
# Backend
docker compose run --rm backend pytest

# Frontend
docker compose run --rm frontend npm run test -- --run
```

También puede ejecutar las pruebas desde la máquina anfitriona:

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest

# Frontend
cd frontend
npm install
npm run test -- --run
```

## Seeds y datos de ejemplo

```bash
pip install -r backend/requirements.txt
./scripts/seed_local_dev.sh
```

El wrapper acepta variables `MONGO_DSN` y `MONGO_DATABASE` para apuntar a una instancia personalizada. Los archivos en `seed/` se pueden reemplazar por snapshots propios cuando estén disponibles.

## Endpoints mock disponibles

La API expone un conjunto de endpoints de referencia que permiten al frontend consumir datos sin depender todavía de los pipelines productivos:

- `GET /api/v1/datasets/state` — estado de sincronización DVC/S3.
- `GET /api/v1/corpus/documents` y `GET /api/v1/corpus/{doc_id}` — documentos, filtros y detalle.
- `POST /api/v1/translation/recompute` / `GET /api/v1/translation/metrics/{doc_id}` — recomputo y métricas.
- `POST /api/v1/summaries/jobs` / `GET /api/v1/summaries/jobs/{job_id}` — jobs del laboratorio de resúmenes.
- `GET /api/v1/evaluation/overview`, `/evaluation/alignment/{summary_id}`, `/evaluation/rag/metrics` — métricas de QA y RAG.
- `GET /api/v1/jobs/status` / `GET /api/v1/jobs/costs` — monitoreo operativo.

## Próximos pasos sugeridos

- Conectar los endpoints reales (MongoDB, MLflow y pipelines de generación/evaluación).
- Instrumentar autenticación y control de roles (FastAPI Depends + JWT/OAuth2).
- Añadir CI/CD (GitHub Actions) para pruebas, lint y build de imágenes.
- Integrar observabilidad (Sentry, Prometheus) y seguridad (rate limiting, API keys).

Para más detalle sobre decisiones de arquitectura y fases de entrega consulte `docs/architecture.md`.
