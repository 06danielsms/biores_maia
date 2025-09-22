# Arquitectura técnica BIORES Maia

## Visión general

El sistema se compone de cuatro bloques principales:

1. **Backend FastAPI (`backend/app`)**
   - API REST versión `v1` organizada en routers (`app/api/v1`).
   - Configuración y secretos gestionados con `pydantic-settings` (`app/core/config.py`).
   - Logging estructurado con `logging.config` (`app/core/logging.py`).
   - Integración opcional con Sentry (`app/core/observability.py`) habilitada vía variable `SENTRY_DSN`.
   - Dependencias listas para MongoDB (Motor) y MLflow (`app/dependencies.py`).
   - Pruebas asíncronas con `pytest` + `httpx.AsyncClient` (`backend/tests`).

2. **Frontend React (`frontend/src`)**
   - SPA creada con Vite, React 19 y Ant Design.
   - Layout principal (`layouts/MainLayout.jsx`) que compone navegación lateral, encabezado y breadcrumb.
   - Páginas iniciales alineadas al roadmap del mockup (dashboard, corpus, traducción, resúmenes, evaluación, operaciones).
   - Estado global ligero vía Zustand (`store/useSettingsStore.js`).
   - Capa de datos con Axios + React Query (`services/apiClient.js`, `hooks/useApi.js`).
   - Tests con Vitest + Testing Library (`src/pages/__tests__`).

3. **Infraestructura en contenedores (`docker-compose.yml`)**
   - `backend`: imagen Python 3.12 con auto-reload y variables desde `.env`.
   - `frontend`: imagen Node 22 ejecutando Vite en modo desarrollo.
   - `mongo`: base documental 7.0 con volumen persistente `mongo-data`.
   - `redis`: broker/cache para workers de traducción y trabajos programados.
   - `mlflow`: servidor oficial 2.14 con backend SQLite y artefactos en `mlflow-data`.

4. **Pipelines de datos existentes**
   - Scripts, notebooks y resultados previos se preservan en `scripts/`, `data/`, `results/` y `mlflow_wor/` para integración futura.

## Capas del backend

| Capa            | Descripción                                                                                 |
|-----------------|---------------------------------------------------------------------------------------------|
| `core`          | Configuración, logging, constantes y utilidades comunes.                                    |
| `api.v1`        | Routers con endpoints versionados; ejemplo actual: `/health` (liveness + status detallado). |
| `services`      | Lugar para lógica de negocio: traducción, evaluación, ingesta, etc.                         |
| `models`        | Esquemas pydantic y ODM para documentos y respuestas.                                       |
| `dependencies`  | Factories para inyección de MongoDB, MLflow y recursos externos.                            |

Pruebas: se emplea `pytest` con configuración asíncrona (`backend/pytest.ini`) y tests de humo sobre los endpoints de salud.

## Capas del frontend

| Módulo                     | Rol principal                                                                    |
|---------------------------|-----------------------------------------------------------------------------------|
| `layouts/MainLayout.jsx`  | Orquestación de estructura (sidebar, header, breadcrumb y outlet).                |
| `components/navigation`   | Menú lateral reutilizable alimentado por `config/navigation.js`.                  |
| `components/overview`     | Tarjetas de KPIs, timeline de actividad y widgets para el dashboard.              |
| `pages/*`                 | Vistas de alto nivel según roadmap (Dashboard, Corpus, Traducción, etc.).         |
| `hooks/useApi.js`         | Abstracción de React Query para consumo del backend.                              |
| `services/apiClient.js`   | Cliente Axios con interceptores de error y base URL dinámica.                     |
| `store/useSettingsStore`  | Estado global con URLs, entorno y configuración operativa.                        |

Testing: `vitest` + Testing Library con proveedor de Ant Design para validar renderizado de componentes clave.

## Flujos esperados

1. **Ingesta y traducción**
   - Backend expone endpoints para listar documentos y lanzar jobs de traducción (en próximos sprints).
   - Frontend consume `/corpus` y `/translation` para mostrar estado y permitir recomputos.

2. **Generación de resúmenes**
   - FastAPI orquesta pipelines (Celery o AWS) y expone SSE/WebSockets con métricas en vivo.
   - Frontend muestra viewer estructurado, stream token a token y aprobaciones.

3. **Evaluación y QA**
   - Integración con MLflow para runs, AlignScore, BERTScore y métricas propias.
   - Tablas y charts comparativos vs BioLaySumm 2023 y umbrales internos.

4. **Operaciones y gobernanza**
   - Registro de auditoría, costos AWS, logs de pipelines y alertas críticas.
   - Integración con S3 para evidencia y descargas.

## Estrategia de pruebas

- **Backend**: tests unitarios y de contrato usando `httpx.AsyncClient`. Futuro: mock de Mongo (`mongomock_motor`), pruebas de servicios y esquemas (`pydantic`).
- **Frontend**: tests de componentes críticos (cards, tablas, formularios) con Vitest. Futuro: pruebas de integración con `msw` para simular API.
- **End-to-end**: planificado con Playwright/Cypress una vez estabilizados los endpoints.

## Automatización CI/CD (pendiente)

Sugerencias:
- GitHub Actions con matrices (`python`, `node`) para ejecutar `pytest` y `npm run test`.
- Build y push de imágenes Docker a un registro (ECR, GHCR).
- Despliegue automático a entornos de staging (Kubernetes, ECS o EC2 + docker compose).

## Observabilidad

- Hooks para OpenTelemetry (`fastapi-instrumentation`, `react-use-telemetry`).
- Integración con Sentry (JS y Python) y alertas Slack/Email.
- Health checks ya implementados (`/api/v1/health/`, `/health/live`).

## Roadmap

1. Implementar endpoints reales para corpus, traducciones y resúmenes.
2. Conectar base Mongo existente y modelar esquemas compartidos.
3. Integrar pipelines DVC/MLflow desde backend.
4. Añadir autenticación (Keycloak, Auth0 o JWT propio) y RBAC.
5. Desplegar infraestructura IaC (Terraform/CloudFormation) alineada a pipelines actuales.
