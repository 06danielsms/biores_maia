# Contratos de API y mapa de datos — BIORES Maia

Este documento describe, endpoint por endpoint, los contratos esperados por el frontend, la información que debe persistir cada servicio y los artefactos auxiliares que debemos exponer para trabajar con datos reales. Está pensado como guía para los científicos de datos y el equipo de backend al momento de integrar corpus, métricas y resultados provenientes de notebooks o pipelines batch.

- **Base URL**: `https://<host>/api/v1` (en local `http://localhost:8000/api/v1`).
- **Formato**: JSON UTF-8, `Content-Type: application/json` salvo descargas puntuales.
- **Autenticación**: no implementada aún; planificar JWT/OAuth2 en siguiente iteración.

## Estado actual de los datos

| Endpoint | Estado de los datos | Fuente actual | Comentario |
|----------|---------------------|---------------|------------|
| `/datasets/state` | Real | `metrics.csv` + agregaciones (`local_metrics`) | KPIs de conteo, cobertura y legibilidad calculados desde artefactos tokenizados. |
| `/datasets/tokenized/beto` | Real | Archivos `.arrow` (`data/tokenized/` o S3) | Muestra ejemplos BETO paginados con texto y labels. |
| `/corpus/documents` | Real | Colección `corpus_documents` (Mongo) | Utiliza los documentos cargados por `scripts/seed_local_dev.sh` / pipelines ETL. |
| `/corpus/{document_id}` | Real | `corpus_documents` + anotaciones `comments` | Drawer de detalle con texto original, traducción y metadatos reales. |
| `/translation/metrics/{id}` | Real | Colección `translation_metrics` (Mongo) | Métricas e historial por documento, cargados desde `results/metrics_sample_es.csv` + `seed/translation_history.json`. |
| `/translation/recompute` | Semi-real | Colección `translation_jobs` (Mongo) | Registra el job y notifica; ejecución del worker todavía manual. |
| `/summaries/jobs` | Semi-real | Colección `summary_jobs` (Mongo) | Persiste parámetros del job; pipeline de generación se orquesta externamente. |
| `/summaries/jobs/{job_id}` | Real | Colección `summary_jobs` (Mongo) | Panel de progreso y métricas leídos desde Mongo (seeds/ETL). |
| `/evaluation/overview` | Real | `local_metrics.get_evaluation_summary()` | BERTScore/AlignScore/FKGL/Cobertura desde artefactos locales. |
| `/evaluation/alignment/{summary}` | Real | Colección `alignment_findings` (Mongo) | Hallazgos de QA cargados desde `seed/alignment_findings.json`. |
| `/evaluation/rag/metrics` | Real | `local_metrics.get_rag_summary()` | Métricas RAG derivadas de `metrics.csv`. |
| `/jobs/status` | Real | Colección `jobs_status` (Mongo) | Snapshot operativo (semillas + futuras integraciones con Celery/Step Functions). |
| `/jobs/costs` | Real | Agregados de `metrics.csv` (`local_metrics`) | Estimación determinística basada en volumen de documentos; ajustar con Cost Explorer.


- **Datos reales conectados**: `/datasets/state`, `/datasets/tokenized/beto`, `/corpus/*`, `/translation/metrics/{id}`, `/evaluation/overview`, `/evaluation/alignment/{summary}`, `/evaluation/rag/metrics`, `/jobs/status`, `/jobs/costs`. Todos se alimentan desde Mongo o agregaciones calculadas sobre artefactos del repositorio.
- **Datos semi-reales**: `POST /translation/recompute` y `POST /summaries/jobs` registran los jobs (Mongo + Slack) pero la ejecución del worker depende todavía de procesos externos.
- **Semillas locales**: `./scripts/seed_local_dev.sh` importa los JSON de `seed/` y construye las colecciones necesarias para desarrollo sin pipelines activos.

## 1. Resumen de endpoints activos

| Método | Ruta                               | Uso principal en UI                                       |
|--------|------------------------------------|-----------------------------------------------------------|
| GET    | `/health/`                         | Banner de disponibilidad y tag de entorno en header.      |
| GET    | `/datasets/state`                  | KPIs de datasets en Dashboard y tabla lateral en Corpus.  |
| GET    | `/datasets/tokenized/beto`        | Feed UI con ejemplos BETO (texto + etiquetas).          |
| GET    | `/corpus/documents`                | Tabla de exploración del corpus con filtros.              |
| GET    | `/corpus/{document_id}`            | Drawer con texto original, traducción y métricas.         |
| GET    | `/translation/metrics/{document}`  | Panel de métricas en Translation Lab.                     |
| POST   | `/translation/recompute`           | CTA “Recomputar traducción” en Translation Lab.           |
| POST   | `/summaries/jobs`                  | Formulario para lanzar nuevos resúmenes.                  |
| GET    | `/summaries/jobs/{job_id}`         | Panel de progreso y métricas en Summary Studio.           |
| GET    | `/evaluation/overview`             | KPIs globales (BERTScore, AlignScore, FKGL, cobertura).   |
| GET    | `/evaluation/alignment/{summary}`  | Tabla de alucinaciones y acciones correctivas.            |
| GET    | `/evaluation/rag/metrics`          | Tabla comparativa de métricas RAG en Evaluation Board.    |
| GET    | `/jobs/status`                     | Tabla de jobs activos en Dashboard y Operations Center.   |
| GET    | `/jobs/costs`                      | Breakdown de costos AWS en Operations Center.             |

## 2. Contratos por dominio

### 2.1 Salud y configuración

**Endpoint**: `GET /health/`

- **Request**: sin parámetros.
- **Response**:
  ```json
  {
    "status": "ok",
    "service": "Biores Maia API",
    "timestamp": "2024-11-22T10:32:15.204Z",
    "environment": "staging"
  }
  ```
- **Campos**:
  | Campo        | Tipo   | Descripción                                                    | Fuente real                                 |
  |--------------|--------|----------------------------------------------------------------|---------------------------------------------|
  | `status`     | str    | `ok` si los subservicios críticos responden.                   | Auto-generado, agregar chequeos de dependencias. |
  | `service`    | str    | Nombre de la app (configurado).                                | `settings.app_name`                         |
  | `timestamp`  | str    | ISO8601 con zona `Z`.                                           | `datetime.utcnow()`                         |
  | `environment`| str    | Entorno (`local`, `staging`, `prod`).                           | `settings.environment`                      |
- **Notas**: extender para incluir `dependencies` (Mongo, Redis, S3) y campos de versión git.

### 2.2 Datasets

**Endpoint**: `GET /datasets/state`

- **Query params**: ninguno.
- **Response**:
  ```json
  {
    "datasets": [
      {
        "name": "ClinicalTrials Highlights",
        "version": "v2025.09.21",
        "last_sync": "2025-09-21T19:01:06.273786Z",
        "storage": "s3://biores-maia-data-clean/processed/ClinicalTrials.gov",
        "status": "synced",
        "document_count": 449,
        "translated_pct": 100.0,
        "metrics_ready_pct": 97.6,
        "short_sentence_ratio": 0.212,
        "readability_ratio": 0.595
      }
    ]
  }
  ```
- **Campos obligatorios**:
  | Campo        | Tipo   | Descripción                                                         | Fuente real                                   |
  |--------------|--------|---------------------------------------------------------------------|-----------------------------------------------|
  | `name`       | str    | Nombre legible del dataset/lote.                                    | Catalogar en Mongo colección `datasets_meta`. |
  | `version`    | str    | Commit/tag DVC o hash de snapshot.                                  | DVC (`dvc exp show`) o tabla `dvc_runs`.      |
  | `last_sync`  | str    | Última sincronización con S3/DVC.                                   | Pipelines ingestion.                          |
  | `storage`    | str    | Ubicación física (S3, GDrive, etc.).                                | Configuración infra.                          |
  | `status`     | str    | `synced`, `resync_required`, `queued`, `error`.                     | Calculado con pipelines o heurísticas de cobertura. |
  | `document_count` | int   | Total de documentos disponibles en el dataset.                          | Agregado desde `metrics.csv` (S3 `processed/`). |
  | `translated_pct` | float | Porcentaje de registros etiquetados como PLS.                           | Agregado desde `metrics.csv`.                  |
  | `metrics_ready_pct` | float | Cobertura de evidencia (entidades ≥ 5).                              | Agregado desde `metrics.csv`.                  |
  | `short_sentence_ratio` | float | Fracción con longitud media de oración ≤ 25 tokens.                 | Agregado desde `metrics.csv`.                  |
  | `readability_ratio` | float | Fracción con Gunning Fog ≤ 18.                                        | Agregado desde `metrics.csv`.                  |
- **Estado de los datos**: real — se calcula leyendo `metrics.csv` (local o S3) y aplicando heurísticas locales.
- **UI**: Dashboard (listado horizontal) y Corpus Explorer (descripciones laterales). Los porcentajes ya se alimentan desde los artefactos tokenizados/metrics.

#### 2.2.1 Dataset tokenizado BETO

**Endpoint**: `GET /datasets/tokenized/beto`

- **Query params**:
  | Param | Tipo | Opcional | Descripción |
  |-------|------|----------|-------------|
  | `split` | enum | Sí | `train` (default) o `test`. |
  | `limit` | int ≤200 | Sí | Número de ejemplos por página (default 20). |
  | `offset` | int ≥0 | Sí | Desplazamiento para paginación. |
  | `preview_chars` | int 40-1000 | Sí | Tamaño máximo del preview textual devuelto. |

- **Response**:
  ```json
  {
    "dataset": "beto_dataset",
    "split": "train",
    "total": 1000,
    "limit": 20,
    "offset": 0,
    "has_more": true,
    "info": {
      "feature_columns": ["text", "label", "input_ids", "token_type_ids", "attention_mask"],
      "label_map": {"0": "pls", "1": "tech"},
      "sequence_length": 256
    },
    "items": [
      {
        "index": 0,
        "text": "Resultados de ensayo clínico...",
        "preview": "Resultados de ensayo clínico...",
        "label": 0,
        "label_name": "pls",
        "token_length": 256,
        "attention_span": 243
      }
    ]
  }
  ```
- **Campos relevantes**: `text`, `label`, `label_name`, `token_length`, `attention_span`. El frontend puede usar `preview` para tablas y `text` para paneles de detalle.
- **Fuente real**: archivos `.arrow` y `dataset_info_*.json` sincronizados vía DVC (`data/tokenized/`) o S3 (`s3://biores-maia-data-clean/tokenized/beto_dataset/`).
- **Notas**: el backend cachea el dataset en memoria; considerar un job que derive CSV/JSON amigable si la cardinalidad crece. `label_map` puede sobreescribirse con `TOKENIZED_LABEL_MAP` en `.env` (formato `0=pls,1=tech`).
- **Estado de los datos**: real — proviene de los archivos `.arrow` versionados en `data/tokenized/` o S3 `tokenized/beto_dataset/`.

### 2.3 Corpus

#### 2.3.1 Listado

**Endpoint**: `GET /corpus/documents`

- **Query params**:
  | Param   | Tipo    | Opcional | Descripción                                     |
  |---------|---------|----------|-------------------------------------------------|
  | `source`| str     | Sí       | Filtra por origen (ClinicalTrials, Cochrane…).  |
  | `status`| enum    | Sí       | `translated` o `pending`.                       |
  | `metrics`| enum   | Sí       | `published` o `processing`.                     |
  | `limit` | int ≤200| Sí       | Tamaño de página (default 50).                  |
  | `offset`| int ≥0  | Sí       | Desplazamiento de página.                       |

- **Response**:
  ```json
  {
    "items": [
      {
        "id": "CT-2024-0001",
        "file_name": "CT-2024-0001.json",
        "source": "ClinicalTrials.gov",
        "language": "en",
        "readability_fkgl": 46.2,
        "translated": true,
        "metrics_ready": true,
        "tokens": 1874,
        "domain": "Cardiología",
        "alignment_risk": "bajo",
        "updated_at": "2024-11-21T22:14:00Z"
      }
    ],
    "total": 1234,
    "limit": 50,
    "offset": 0
  }
  ```
- **Campos obligatorios**: `id`, `file_name`, `source`, `language`, `translated`, `metrics_ready`, `updated_at`. Opcionales usados por la UI: `readability_fkgl`, `tokens`, `domain`, `alignment_risk`.
- **Fuente real**: colección `health_literacy_db.corpus_documents` (Mongo), alimentada por `scripts/seed_local_dev.sh` y los ETL del pipeline. Se proyectan los campos usados por la UI y se ordena por `updated_at desc`.
- **Notas**: devolver `file_name`, `alignment_risk`, `metrics_ready` y `tokens` para evitar mapeos manuales; considerar incluir `summary_status` y `tags` clínicos en futuras iteraciones.
- **Estado de los datos**: real — la UI consume exactamente los documentos disponibles en Mongo (seed o ingestados por pipeline).

#### 2.3.2 Detalle de documento

**Endpoint**: `GET /corpus/{document_id}`

- **Response**:
  ```json
  {
    "id": "CT-2024-0001",
    "source": "ClinicalTrials.gov",
    "language": "en",
    "original": "Phase III randomized study...",
    "translation": "Estudio fase III aleatorizado...",
    "metrics": {
      "bleu": 54.9,
      "chrf2": 0.721,
      "fkgl": 6.3,
      "bertscore": 0.89
    },
    "comments": [
      {
        "author": "María Ortega",
        "role": "Especialista clínica",
        "content": "Validar terminología cardiológica.",
        "timestamp": "2024-11-22T10:30:00Z"
      }
    ]
  }
  ```
- **Persistencia**:
  - `original` / `translation`: campos `original` y `translation` (o `translated_content`) en `corpus_documents`.
  - `metrics`: colección `translation_metrics` (Mongo) enlazada por `document_id`.
  - `comments`: almacenados en el mismo documento (`comments` array). Evaluar moverlos a `document_annotations` para historizar cambios.
- **Ampliaciones**: añadir `audit_log`, versiones de traducción, enlaces a archivos en S3 y botón de descarga (`/corpus/{id}/download`).
- **Estado de los datos**: real — los detalles provienen de Mongo y se actualizan al volver a ejecutar `seed_local_dev.sh` o los pipelines de ingesta.

### 2.4 Traducción

#### 2.4.1 Métricas

**Endpoint**: `GET /translation/metrics/{document_id}`

- **Response**:
  ```json
  {
    "document_id": "CT-2024-0001",
    "source": "ClinicalTrials.gov",
    "generated_at": "2024-11-22T09:14:33Z",
    "metrics": {
      "bleu": 54.9,
      "chrf2": 0.721,
      "fkgl": 6.3,
      "length_ratio": 0.98,
      "medical_terms": 38
    },
    "history": [
      {
        "timestamp": "2024-11-22T09:14:00Z",
        "event": "Job Helsinki batch 2024-Q4",
        "detail": "64 documentos traducidos · BLEU promedio 52.4",
        "color": "blue"
      }
    ]
  }
  ```
- **Fuente real**: colección `translation_metrics` en Mongo (alimentada por `results/metrics_sample_es.csv` + `seed/translation_history.json`). Cada entrada incluye métricas numéricas e historial de eventos (`history`).
- **Estado de los datos**: real — el endpoint consulta Mongo y cae a los artefactos locales sólo si la colección está vacía.

#### 2.4.2 Recompute

**Endpoint**: `POST /translation/recompute`

- **Request body**:
  ```json
  {
    "document_id": "CT-2024-0001",
    "model": "helsinki-nlp/opus-mt-en-es",
    "force": true,
    "notify": true,
    "priority": "high",
    "notes": "Recalcular tras actualización de glosario cardiología"
  }
  ```
- **Campos mínimos**: `document_id`. Recomendado capturar `model`, `force`, `notify`, `priority`, `notes`, `target_style`.
- **Response**:
  ```json
  {
    "job_id": "translation-job-20241122-094701",
    "status": "queued",
    "message": "Recomputo encolado. Revisar pipeline celery-worker"
  }
  ```
- **Persistencia**: registra el payload en la colección `translation_jobs` (Mongo) y dispara una notificación a Slack (`notify_slack_async`). El worker de recomputo debe consumir esta colección/cola para materializar la traducción.
- **Estado de los datos**: semi-real — se guarda el job y se notifica, pero la ejecución depende de los workers externos.

### 2.5 Resúmenes

#### 2.5.1 Crear job

**Endpoint**: `POST /summaries/jobs`

- **Request**:
  ```json
  {
    "dataset": "pfizer-2024-q4",
    "model": "ollama-phi3-mini",
    "prompt_template": "templates/pfizer_clinical.j2",
    "max_tokens": 520,
    "tone": "coloquial",
    "citations_format": "ama",
    "include_metrics": true,
    "notify_slack": true,
    "run_tags": ["trial", "beta"],
    "source_documents": ["CT-2024-0001", "CT-2024-0002"]
  }
  ```
- **Response**:
  ```json
  {
    "job_id": "summary-job-20241122-094735",
    "status": "queued",
    "message": "Job recibido. Revisar worker de resúmenes"
  }
  ```
- **Persistencia**: colección `summary_jobs` en Mongo (estado, parámetros, usuario). El script de seeds carga jobs de ejemplo y el endpoint `POST` añade nuevas entradas; planificar integración con MLflow/S3 para salidas.
- **Validaciones**: exigir que `dataset` exista en `/datasets/state`, controlar cuota por usuario.
- **Estado de los datos**: semi-real — los jobs quedan registrados en Mongo y visibles en la UI; la ejecución completa depende de los workers de resúmenes.

#### 2.5.2 Consultar job

**Endpoint**: `GET /summaries/jobs/{job_id}`

- **Response**:
  ```json
  {
    "job_id": "summary-job-20241122-094735",
    "dataset": "pfizer-2024-q4",
    "model": "phi3:mini",
    "status": "streaming",
    "progress": 62,
    "started_at": "2024-11-22T09:47:00Z",
    "metrics": {
      "readability_fkgl": 6.1,
      "coherence": 0.84,
      "alignscore": 0.73
    },
    "current_section": "Recomendaciones",
    "estimated_remaining_seconds": 120,
    "output_path": "s3://biores-maia/results/summaries/summary-job-20241122-094735/output.md"
  }
  ```
- **Ampliaciones**: incluir `stream_url` (SSE/WebSocket), `owner`, `cost_estimate`, `tokens_used`, `error`.
- **Fuente real**: scheduler Celery/AWS + MLflow runs (`mlflow_work/`).
- **Estado de los datos**: real — el detalle proviene de la colección `summary_jobs` (semillas o jobs creados vía API). Campo `summary` y `timeline` se actualizan cuando el worker escribe progreso.

### 2.6 Evaluación

#### 2.6.1 Overview global

**Endpoint**: `GET /evaluation/overview`

- **Response**:
  ```json
  {
    "generated_at": "2025-09-21T19:01:06.273786Z",
    "metrics": {
      "bertscore_f1": 0.553,
      "alignscore": 0.544,
      "fkgl": 21.25,
      "coverage_evidence": 0.885
    },
    "alerts": [
      {
        "summary_id": "DATASET-TRIAL SUMMARIES",
        "level": "warning",
        "detail": "Trial Summaries presenta cobertura de evidencia 79.7%"
      }
    ]
  }
  ```
- **Fuente real**: agregados locales de `metrics.csv` (S3 `processed/`) y coherencias en `results/export/only_metrics_sample.csv`.
- **Estado de los datos**: real — cifras calculadas por `local_metrics.get_evaluation_summary()`.
- **Notas**: los valores se recalculan en caliente; añadir `baseline` (BioLaySumm), `delta`, `samples_evaluated`, `metrics_version` para historizar comparaciones.

#### 2.6.2 Hallazgos de alineación

**Endpoint**: `GET /evaluation/alignment/{summary_id}`

- **Response**:
  ```json
  {
    "summary_id": "TECH-332",
    "alignscore": 0.73,
    "hallucinations": [
      {
        "severity": "critical",
        "excerpt": "El resumen menciona pacientes pediátricos...",
        "evidence": "ClinicalTrials.gov NCT04122111",
        "anchor_start": 125,
        "anchor_end": 198,
        "evidence_url": "s3://.../documents/CT-2024-0001.pdf"
      }
    ],
    "actions": ["Solicitar nueva generación", "Adjuntar evidencia"]
  }
  ```
- **Persistencia**: colección `alignment_findings` en Mongo con referencias a evidencia (S3, Mongo `corpus`). Considerar almacenar `explanation`, `resolved_at`, `resolved_by`.
- **Estado de los datos**: real — el endpoint lee los hallazgos almacenados en Mongo (cargados vía seeds o pipelines de QA).

#### 2.6.3 Métricas RAG

**Endpoint**: `GET /evaluation/rag/metrics`

- **Response**:
  ```json
  {
    "datasets": [
      {
        "name": "ClinicalTrials Highlights",
        "precision_at_5": 0.82,
        "recall_at_5": 0.76,
        "ndcg": 0.88,
        "latency_ms": 320
      }
    ]
  }
  ```
- **Fuente real**: Resultados de evaluación RAG almacenados en `results/` (CSV/JSON) o MLflow tags. Añadir `run_id`, `retriever_version`, `chunk_size`.
- **Estado de los datos**: real — derivado de `get_rag_summary()` sobre `metrics.csv`.

### 2.7 Jobs y costos

#### 2.7.1 Estado

**Endpoint**: `GET /jobs/status`

- **Response**:
  ```json
  {
    "jobs": [
      {
        "job": "Traducción Helsinki batch 2024-Q4",
        "owner": "worker-03",
        "started_at": "2024-11-22T09:01:00Z",
        "duration": "17m",
        "state": "in_progress",
        "type": "translation",
        "queue": "celery:translation",
        "last_heartbeat": "2024-11-22T09:10:12Z"
      }
    ]
  }
  ```
- **Fuente real**: colección `jobs_status` en Mongo (poblada por seeds o integraciones con Celery/Step Functions). Añadir `progress`, `error`, `log_url` para seguimiento completo.
- **Estado de los datos**: real — la UI consume directamente la colección `jobs_status`; al conectar los workers bastará con mantener la misma estructura.

#### 2.7.2 Costos

**Endpoint**: `GET /jobs/costs`

- **Response**:
  ```json
  {
    "services": [
      {
        "service": "EC2 (GPU/CPU)",
        "monthly_cost": 1240,
        "trend": "+8.2%",
        "currency": "USD",
        "source": "aws-cost-explorer"
      }
    ]
  }
  ```
- **Fuente real**: agregados deterministas calculados por `local_metrics` a partir de `metrics.csv` (número de documentos * factor 0.42). Sustituir por Cost Explorer cuando esté disponible.
- **Estado de los datos**: real (determinístico) — mientras no llegue Cost Explorer, el valor se deriva siempre del dataset actual y se refleja igual en la UI.

## 3. Datos y artefactos fuera de la API

| Repositorio/archivo                 | Contenido                             | Acción recomendada                                                   |
|------------------------------------|---------------------------------------|---------------------------------------------------------------------|
| `data/`                            | Corpus crudo y traducido.             | Mantener ETL que sincronice con Mongo y registre metadatos.         |
| `results/metrics_*.csv`            | Métricas de traducción/evaluación.    | Cargar en colección `metrics_history` para consultas por API.       |
| `results/report/`                  | Reportes Markdown/PDF.                | Exponer endpoint `/reports/{report_id}` o vincular en UI.           |
| `mlflow_wor/`                      | Runs de MLflow locales.               | Configurar servidor MLflow y capturar `run_id` en respuestas API.   |
| `scripts/*.py`                     | Pipelines de cómputo (BLEU, stats).   | Convertir en jobs reproducibles (Celery/AWS) y versionar outputs.   |
| `Clasificador_de_alfabetización...`| Modelos de clasificación PLS vs TECH. | Serializar artefactos (`.pkl`) y planificar endpoints de inferencia. |

Para que el frontend pueda “leer resultados pasados y documentos reales” necesitamos endpoints adicionales:

1. `GET /corpus/{id}/versions` — historial de traducciones y ediciones.
2. `GET /summaries/jobs/{job_id}/artifacts` — enlaces a Markdown/PDF, métricas extendidas.
3. `GET /evaluation/runs` — catálogo de ejecuciones MLflow (con filtros por modelo, fecha, dataset).
4. `GET /analytics/metrics-history` — series temporales para KPIs (BLEU, AlignScore, costos).
5. `GET /notebooks/{slug}` — metadatos de notebooks convertidos a reportes (usando Papermill o nbconvert).

> Tip: `./scripts/seed_local_dev.sh` ejecuta todos los importadores (`datasets_state`, `corpus_documents`, `translation_metrics`, `jobs_status`, `evaluation_overview`) usando los JSON de `seed/`. Útil para preparar demos o desarrollos sin pipelines conectados.

## 4. Integración con notebooks y pipelines

- **Traducción y métricas**: los notebooks/scrips en `scripts/` generan CSV con métricas por documento. Planificar ingestión automática a Mongo (`translation_metrics`) y versionado por `run_id`. Cada fila debe incluir: `document_id`, `metric_name`, `metric_value`, `model_version`, `run_id`, `computed_at`.
- **Evaluación (BERTScore, AlignScore)**: almacenar en MLflow con artefactos JSON. El backend debe leerlos mediante `mlflow_client` o mantener un proceso ETL que copie a Mongo.
- **Clasificador PLS vs TECH**: publicar endpoint `POST /analytics/health-literacy-score` que tome texto y responda `prob_pls`, `prob_tech`; persistir inferencias para auditoría.
- **Resultados AWS/Ollama**: los logs `maia-run.log` y artefactos en S3 deben registrarse en `jobs` para que el frontend pueda descargarlos.

## 5. Checklist para datos reales

1. **Configurar conexiones**: completar `.env` con `MONGO_DSN`, `MLFLOW_TRACKING_URI`, `S3_BUCKET_RESULTS`, `SLACK_WEBHOOK_URL`, `REDIS_DSN`.
2. **Normalizar esquemas** en Mongo:
   - `corpus_documents` (metadatos, enlaces a contenido en GridFS/S3).
   - `translations` (historial, métricas, responsable).
   - `summary_jobs` y `summary_outputs` (texto generado, métricas, aprobaciones).
   - `evaluation_runs` (resultado de pipelines, baseline, run_id MLflow).
   - `jobs_runtime` (estado Celery/AWS, logs).
3. **ETL inicial**: poblar endpoints con datos reales extrayendo de `data/` y `results/`.
4. **Sincronizar con DVC/S3**: exponer commit/experimento actual en `/datasets/state` y relacionarlo con documentos.
5. **Auditoría y seguridad**: añadir `owner`, `created_at`, `updated_at` en todas las respuestas; definir roles y filtros por usuario.
6. **Versionar contratos**: si se agregarán campos nuevos, mantener compatibilidad (agregar, no remover) y actualizar MSW + tests.
7. **Monitorizar**: extender `/health/` para comprobar conectividad a Mongo, Redis, Slack, MLflow y S3.

## 6. Próximos pasos sugeridos

1. Orquestar los workers de traducción y resúmenes (Celery/Step Functions) para que los jobs encolados completen estados y métricas en Mongo.
2. Conectar AWS Cost Explorer / FinOps para reemplazar el cálculo determinístico de `/jobs/costs` y añadir trazabilidad por `job_id`.
3. Exponer endpoints de artefactos (descarga de traducciones, resúmenes, logs) utilizando S3/MLflow y enriquecer `/summaries/jobs/{id}` con enlaces.
4. Integrar MLflow como fuente oficial de métricas históricas (`run_id`, `experiment`, `metrics_version`) y reflejarlo en `/evaluation/*`.
5. Diseñar endpoints para notebooks/reportes históricos y para el clasificador PLS/TECH, incluyendo auditoría de inferencias.
6. Añadir autenticación y controles de permisos para operaciones sensibles (recomputos, lanzamientos en AWS) y extender `/health/` con verificaciones de dependencias externas.

Con este contrato alineado, el equipo de ciencia de datos puede comenzar a volcar resultados reales y validar la UI sin romper la compatibilidad del frontend.
