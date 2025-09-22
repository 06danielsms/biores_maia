# Onboarding de datos reales — BIORES Maia

Este documento describe cómo preparar el entorno local para cargar datasets reales en MongoDB, ejecutar el backend y validar el frontend con datos actualizados. Sigue estos pasos al pie de la letra para evitar inconsistencias.

## 1. Preparar entorno de Python

1. Crea un virtualenv o activa tu entorno de preferencia.
2. Instala dependencias del backend:
   ```bash
   pip install -r backend/requirements.txt
   ```
   Esto incluye `pydantic` 2.x, `motor`, `fastapi` y herramientas de pruebas.

> Si usas Anaconda, asegúrate de crear un environment separado con `python>=3.10` para evitar conflictos con `pydantic`.

## 2. Variables de entorno

Crea un archivo `backend/.env` (o exporta las variables) con al menos:

```
MONGO_DSN=mongodb://localhost:27017
MONGO_DATABASE=health_literacy_db
S3_BUCKET_RESULTS=biores-maia-results
MLFLOW_TRACKING_URI=http://localhost:5000
SLACK_WEBHOOK_URL=
```

- Ajusta `MONGO_DSN` si usas Mongo Atlas o Docker.
- `SLACK_WEBHOOK_URL` es opcional; si se deja vacío, las notificaciones se omiten.

## 3. Poblar MongoDB

El proyecto incluye scripts para cargar datos históricos. Ejecuta los siguientes comandos en orden (cada uno acepta `--mongo`, `--database` y `--drop` por si trabajas con otros DSN):

```bash
./scripts/seed_local_dev.sh
```

Este wrapper permite sobreescribir el DSN de Mongo mediante variables de entorno `MONGO_DSN` y `MONGO_DATABASE` si lo necesitas. También puedes ejecutar los scripts individuales manualmente para cargas parciales.

### 3.1 Archivos de entrada

- `data/datasets_state.json`: snapshots DVC/S3 con `name`, `version`, `last_sync`, `status`, `storage`, etc.
- `data/corpus_sample.json`: lista de documentos con texto original, traducción, métricas y comentarios.
- `results/metrics_sample_es.csv`: métricas calculadas por los notebooks (BLEU, chrF2, FKGL, etc.).

Asegúrate de que los campos de fecha estén en ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`). Los scripts normalizan timestamps y convierten métricas numéricas automáticamente.

### 3.2 Columnas/Claves mínimas requeridas

| Script                         | Claves obligatorias                                 |
|--------------------------------|-----------------------------------------------------|
| `import_dataset_state.py`     | `name`, (`last_sync` opcional), `status`, `storage` |
| `import_corpus_documents.py`  | `id`, `source`, `language`, `original`, `translation`|
| `import_translation_metrics.py`| `file`/`document_id`, métricas numéricas             |
| `import_jobs_status.py`       | `job`, timestamps (`started_at` opcional)           |
| `import_evaluation_overview.py`| `metrics` (dict con KPIs), `generated_at`           |


## 4. Ejecutar backend y pruebas

1. Levanta MongoDB (local o Docker).
2. Lanza el backend:
   ```bash
   uvicorn app.main:app --reload --factory --port 8000
   ```
3. Verifica los endpoints con `curl` o `httpie`:
   ```bash
   curl http://localhost:8000/api/v1/datasets/state
   curl http://localhost:8000/api/v1/corpus/documents?limit=5
   curl http://localhost:8000/api/v1/translation/metrics/CT-2024-0001
   ```
4. Ejecuta los tests:
   ```bash
   cd backend
   pytest
   ```

> Si ves `ModuleNotFoundError: app`, confirma que ejecutaste `pip install -r backend/requirements.txt` en el mismo entorno donde corres pytest.

## 5. Arrancar el frontend

1. Copia `.env.local.example` a `.env.local` en `frontend/` y ajusta `VITE_API_URL` a `http://localhost:8000`.
2. Activa mocks opcionales con `VITE_ENABLE_MOCKS=true` si quieres trabajar sin backend.
3. Ejecuta:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
4. Abre `http://localhost:5173` y confirma que las vistas muestran los datasets y documentos importados.

## 6. Checklist de validación

- [ ] `/api/v1/datasets/state` devuelve registros con `last_sync` y `status` correctos.
- [ ] `/api/v1/corpus/documents` lista al menos un documento del lote cargado.
- [ ] `/api/v1/corpus/{id}` incluye `updated_at`, métricas y comentarios.
- [ ] `/api/v1/translation/metrics/{id}` entrega métricas importadas (`bleu`, `chrf2`, etc.).
- [ ] `/api/v1/summaries/jobs/{job_id}` responde con datos reales si hay jobs cargados.
- [ ] React Query deja de mostrar el banner “offline” (salvo que detengas el backend). 

## 7. Siguientes pasos recomendados

- Programar una tarea recurrente (Celery/Cron) que ejecute los scripts de importación a partir de nuevas corridas de DVC/MLflow.
- Exponer endpoints adicionales (`/reports`, `/metrics-history`, `/summaries/jobs/{id}/artifacts`) conforme se vayan populando S3.
- Configurar autenticación para proteger rutas de escritura y operaciones costosas.

---
Con esta guía, cualquier miembro del equipo puede preparar el entorno y validar el frontend con datos productivos en menos de 30 minutos.
