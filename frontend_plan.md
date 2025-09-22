# Plan Frontend y FastAPI - BIORES

## 1. Resumen ampliado del estado actual
- **Datos y versionado**: los corpus biomédicos (ClinicalTrials.gov, Cochrane, Pfizer, Trial Summaries) están versionados con DVC apuntando a un remoto S3 (`s3://biores-maia/dvc`). Hay scripts y notebooks para descargar, limpiar y balancear subconjuntos (60 PLS / 60 TECH) y experimentos adicionales con ~60k documentos.
- **Almacenamiento operativo**: una instancia local de MongoDB (localhost:27017) aloja la colección `health_literacy_db.data_sources_processing` con campos `_id`, `file_path`, `content`, `translated_content`, `file_name` y `source`. Aquí se preservan los documentos originales y sus traducciones al español generadas con el modelo Helsinki-NLP/opus-mt-en-es (BLEU 54.9, chrF2 0.721).
- **Pipelines y métricas**: scripts en `scripts/` traducen, calculan 64 métricas de legibilidad (spaCy + TextDescriptives) y publican resultados en `results/` y MLflow (`mlflow_work/`). Se documentan análisis estadísticos (Mann–Whitney U, Cliff’s d, Cohen’s d, overlap KDE) y un reporte `results/report/top6_metrics.md` con métricas más discriminantes.
- **Modelos**: notebook `Clasificador_de_alfabetizacion_en_salud.ipynb` entrena Random Forest y Gradient Boosting (mejor desempeño RF con ROC AUC 0.9006 usando `class_weight=balanced`) para distinguir PLS vs TECH. Se conservan métricas de validación por fuente (Cochrane, ClinicalTrials.gov, Pfizer).
- **Generación y evaluación**: hay prompts para LLMs, scripts de generación con Ollama (local y en AWS), y pipeline de evaluación (`evaluation/run_eval.py`) que calcula BERTScore, legibilidad y prepara integración de AlignScore. Ejecuta comparaciones con BioLaySumm 2023 y publica reportes en S3 (`s3://biores-maia/results/...`).
- **Infraestructura**: despliegues automatizados via CloudFormation (`infra/aws/cfn/ollama-eval.yaml`) levantan EC2 que ejecutan DVC pull, generan resúmenes (fallback phi3:mini) y suben artefactos a S3; logs se almacenan en `maia-run.log`.
- **Diseño UX**: el Figma (`https://biores.figma.site`) ya describe tres paneles clave: (1) Inicio con KPIs (documentos procesados, % aprobado, tiempos, costos), (2) Laboratorio de resúmenes (configuración LLM, viewer estructurado, stream en vivo de métricas de calidad), y (3) Evaluación & Alucinaciones (métricas RAG, monitoreo de calidad, categorización de alucinaciones).

## 2. Visión de producto
Crear una plataforma única que articule ingesta, traducción, análisis y generación de resúmenes biomédicos en español coloquial. El frontend debe permitir:
- Supervisar el estado del corpus, estadísticas de legibilidad y avances de experimentos.
- Explorar documentos individuales (texto original y traducción) y validar la calidad de la traducción automática.
- Configurar, lanzar y monitorear la generación de resúmenes con distintos LLMs (locales y en la nube), respetando los campos definidos en el mockup.
- Evaluar resúmenes con métricas de relevancia, legibilidad, factualidad, RAG y alucinaciones, integrando hallazgos cuantitativos y cualitativos.
- Gestionar flujos de trabajo orquestados (jobs batch, pipelines AWS, recomputos) y proporcionar trazabilidad para auditorías académicas y clínicas.

## 3. Roles y necesidades
| Rol | Necesidades clave |
| --- | --- |
| Analista de datos | Gestionar corpus desde Mongo/DVC, filtrar documentos, ejecutar recomputos y descargar conjuntos procesados. |
| Especialista médico | Revisar traducciones y resúmenes en lenguaje claro, interpretar métricas de calidad y anotar hallazgos clínicos. |
| Ingeniero ML / LLM | Configurar experimentos (modelos, prompts), monitorizar runs de MLflow y AWS, inspeccionar métricas de evaluación avanzada. |
| PM / Stakeholder | Visualizar KPIs globales, costos, progreso de investigación y comparar con benchmarks externos (BioLaySumm). |
| Auditor / Gobernanza | Consultar historial de cambios, descargas, lanzamientos de jobs y evidencias de calidad para cumplimiento regulatorio. |

## 4. Arquitectura propuesta

### 4.1 Frontend (React + Vite en JavaScript con Ant Design)
Se implementará la UI con React 18 en JavaScript (sin TypeScript en la primera etapa) usando Vite como bundler y la librería Ant Design para componentes. React Router gestionará la navegación, React Query (TanStack Query) el fetching/cache y Zustand o Context API + reducers el estado global ligero.

#### Vistas principales
- **Panel Inicio (Dashboard)**: KPIs del mockup (documentos procesados, % aprobado, tiempos, costos), actividad reciente (traducciones, resúmenes, experimentos) y alertas de pipelines AWS/MLflow. Uso de `Layout`, `Statistic`, `Card`, `Timeline` y `Alert` de Ant Design.
- **Explorador de Corpus**: tabla conectada a MongoDB/DVC con filtros por fuente, idioma, estado de traducción, métricas clave; preview lateral con tabs `Original`, `Traducción`, `Metadatos`, `Métricas destacadas` y gráfica POS vs promedio de grupo. Componentes `Table`, `Form`, `Drawer`, `Descriptions`, `Tabs` y `Tag`.
- **Dataset BETO Tokenizado**: vista enfocada en ejemplos etiquetados (PLS vs TECH) consumiendo `/api/v1/datasets/tokenized/beto`, con tabla paginada (`Table` + `Tag`) y panel de detalle (`Drawer`) que muestra `preview`, longitud de token y métricas auxiliares; base para calibrar el clasificador BETO y explicar resultados en la UI.
- **Comparador de Traducciones**: vista de métricas opus-mt (BLEU, chrF2, brevity penalty) y evaluación manual vs automática, con posibilidad de recalcular. Se apoya en `Card`, `Statistic`, `Segmented` y `Diff`/`Highlight` custom.
- **Laboratorio de Resúmenes (Figma Panel 2)**:
  - Formulario de configuración (`Form`, `Select`, `Slider`, `Radio`, `InputNumber`) para documento, modelo LLM (DeepSeek-R1, Qwen3, Phi3, GPT-4o, etc.), longitud, coloquialidad, tono.
  - Visualizador estructurado (`Tabs`, `Typography`, `Collapse`) con secciones: contexto, metodología, resultados, seguridad, conclusiones; incluye edición colaborativa controlada.
  - Stream en vivo de métricas (Legibilidad, Coherencia, Fidelidad) con `Statistic`, `Progress`, `Result` alimentados vía SSE/WebSocket.
- **Evaluación & Alucinaciones (Figma Panel 3)**:
  - Tablero de métricas RAG y generación (Precision, NDCG, MRR, Recall, Context Precision, Faithfulness, Answer Relevancy, BLEU, Semantic Similarity, AlignScore) mostrados con `Card`, `Tabs`, `Table`, `Radar/Line charts` (Ant Design Charts o Recharts).
  - Vista de alucinaciones con `Alert`, `Timeline`, resaltado de texto y filtros de gravedad (Crítico, Alto, Medio, Bajo).
  - Sección de comentarios clínicos y acciones recomendadas (`Comment`, `List`).
- **Experiment Tracking**: listado de experimentos MLflow y runs AWS con `Table`, `Tag`, `Drawer` para detalle (parámetros, métricas, artefactos S3).
- **Panel de Operaciones**: monitor de jobs (traducción, recomputo, generación) con `Steps`, `Timeline`, `Progress`, estado de workers Celery y salud de servicios (`Result`, `Alert`).

#### Componentes y patrones clave
- Layout responsivo basado en `Layout.Sider` + `Layout.Content` con `Menu` lateral y `PageHeader` por vista.
- Tarjetas KPI reutilizables (`Card` + `Statistic`) para valores, variaciones y objetivos.
- `Table` con paginación server-side, filtros, ordenamiento y export (CSV) mediante botones (`Dropdown`, `Button`). Para datasets grandes se utilizará virtualización con `rc-virtual-list`.
- Componentes de gráficos: Ant Design Charts (basado en G2Plot) o Recharts para líneas, barras, boxplots y densidades encapsulados como `<MetricChart />`.
- Visor de texto con resaltado (diferencias original vs traducido, coloreado por POS) usando bibliotecas como `diff2html` o `react-diff-viewer` embebidas en `Card`/`Collapse`.
- Modales y `Drawer` para comentarios, historial de ediciones y adjuntos.

#### Consideraciones técnicas
- Autenticación con JWT; integración del token vía interceptores (`axios`) y `React Query` para reintentos/invalidación. Guardas de ruta con `React Router`.
- Internacionalización con `react-intl` o `@formatjs` para ES/EN, aprovechando la configuración de `ConfigProvider` de Ant Design.
- Theming: usar `ConfigProvider` y design tokens personalizados para identidad BIORES (colores, tipografías, dark/light mode).
- Manejo de errores centralizado con `message`/`notification` y reporte opcional a Sentry u OTEL.
- SSE/WebSocket para métricas en vivo (laboratorio y jobs) con hooks personalizados.

### 4.2 FastAPI y backend operativos
Diseñar FastAPI modular (`/api/v1`) con servicios desacoplados y conectores a MongoDB, DVC/S3, MLflow y AWS.

#### Módulos sugeridos
- `auth`: emisión y validación de JWT, gestión de roles y auditoría de acciones.
- `datasets`: consulta de metadata DVC/S3, sincronización de fuentes, verificación de versiones.
- `corpus`: endpoints para listar documentos, recuperar original/traducción, actualizar anotaciones y extraer métricas calculadas.
- `translation`: control del pipeline de traducción (opus-mt y modelos futuros), re-cálculo de métricas BLEU/chrF, comparación manual.
- `summaries`: operaciones del laboratorio (configurar job, obtener resumen, versionado, comentarios, descarga de plantilla estructurada).
- `evaluation`: cálculo y consulta de métricas BERTScore, legibilidad, AlignScore (cuando se integre), agregados numéricos y comparaciones vs BioLaySumm.
- `rag`: ingreso y consulta de métricas de recuperación/generación, ingestión de scores desde pipelines RAG.
- `experiments`: wrapper hacia `mlflow.tracking.MlflowClient`, descarga de artefactos y publicaciones de reportes.
- `jobs`: integración con Celery/Redis o Dramatiq para lanzar traducción batch, recomputos, generación en AWS; expone estado, logs, y resultados.
- `infra`: interacción con pipelines AWS (CloudFormation, S3), lectura de logs `maia-run.log`, control de costos estimados.

#### Endpoints clave
| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/api/v1/datasets/state` | Resumen DVC/S3: versiones, hashes, fecha de actualización. |
| GET | `/api/v1/corpus/documents` | Paginación y filtros (fuente, tipo, status traducción, métricas). |
| GET | `/api/v1/corpus/{doc_id}` | Detalle completo (original, traducción, métricas, comentarios). |
| POST | `/api/v1/translation/recompute` | Reprocesa traducción y recalcula métricas de calidad. |
| GET | `/api/v1/translation/metrics/{doc_id}` | BLEU, chrF2, longitud, brevedad, timestamp. |
| POST | `/api/v1/summaries/jobs` | Crea job de generación con parámetros del laboratorio. |
| GET | `/api/v1/summaries/jobs/{job_id}` | Estado, salida parcial, métricas registradas. |
| GET | `/api/v1/evaluation/overview` | KPIs combinados (BERTScore, legibilidad, AlignScore, comparativas). |
| GET | `/api/v1/evaluation/alignment/{summary_id}` | Detalle de alucinaciones categorizadas y referencias. |
| GET | `/api/v1/rag/metrics` | Recupera métricas de pipeline RAG (Precision, NDCG, etc.). |
| GET | `/api/v1/experiments/runs` | Runs MLflow con filtros (experimento, fecha, tags). |
| GET | `/api/v1/experiments/runs/{run_id}` | Parámetros, métricas, artefactos (descarga desde S3). |
| POST | `/api/v1/jobs/aws-launch` | Dispara plantilla CloudFormation preconfigurada y devuelve id de stack/logs. |
| GET | `/api/v1/jobs/{job_id}` | Estado unificado (Celery/AWS) y eventos para SSE. |
| GET | `/api/v1/system/health` | Salud de servicios (Mongo, S3, MLflow, Redis, AWS). |

#### Integraciones
- **MongoDB**: acceso a colecciones para corpus y resúmenes generados; indexación por `source`, `doc_id`, `language`.
- **DVC/S3**: consultas de archivos grandes (usar `dvc.api.open` o lectura directa S3 pre-signed URLs).
- **MLflow**: lectura de runs, métricas y artefactos (`mlflow_work` local o tracking remoto).
- **AWS**: interacción con CloudFormation/EC2 mediante boto3, subida/descarga de reportes desde `s3://biores-maia/results/...`.
- **Herramientas NLP**: wrappers para spaCy, TextDescriptives, BERTScore, AlignScore (cuando se instale), y pipelines LLM (Ollama, OpenAI).

### 4.3 Flujo de datos end-to-end
```text
Fuentes DVC/S3 ──► FastAPI.datasets/corpus ──► MongoDB ──► Frontend Explorador
         │                          │
         │                          └──► FastAPI.translation ─┬──► métricas BLEU/chrF
         │                                                   └──► actualiza Mongo

Usuario configura resumen en Frontend ─► FastAPI.summaries/jobs ─► Celery/AWS (Ollama)
        └─ recibe stream métricas (legibilidad, coherencia, fidelidad) ◄── FastAPI.evaluation

Pipelines RAG/Evaluación ─► FastAPI.rag/evaluation ─► Frontend Evaluación & Alucinaciones
MLflow / AWS logs ─► FastAPI.experiments/infra ─► Panel Operaciones & Tracking
```

### 4.4 Observabilidad y seguridad
- Logging estructurado (JSON) y trazas OTEL para correlacionar jobs entre FastAPI, Celery y AWS.
- Auditoría de acciones (quién generó resumen, descargó datos, lanzó job) almacenada en Mongo o PostgreSQL dedicado.
- Controles de acceso basados en roles con scopes específicos (p.ej. `summaries:launch`, `aws:deploy`, `datasets:write`).
- Mecanismos de rate limiting y validación de payloads para proteger endpoints costosos.

## 5. Historias de usuario

### Fase A — Fundacional (Ingesta y corpus)
- Como analista quiero ver el estado de sincronización del corpus (DVC/S3 vs Mongo) para asegurarme de que trabajamos con la versión correcta.
- Como analista quiero filtrar documentos por fuente, estado de traducción y rango de métricas para preparar datasets de entrenamiento.
- Como especialista médico quiero abrir un documento y visualizar en paralelo el texto original y la traducción para validar fidelidad.
- Como stakeholder quiero conocer los KPIs globales (documentos procesados, tiempo promedio, costo estimado) para reportes ejecutivos.

### Fase B — Traducción y métricas
- Como ingeniero ML quiero volver a ejecutar la traducción de un documento y obtener métricas BLEU/chrF actualizadas para evaluar mejoras del modelo.
- Como analista quiero detectar rápidamente traducciones con baja calidad (BLEU < umbral) y marcarlas para revisión manual.
- Como especialista médico quiero dejar comentarios sobre la traducción y asignar un estado (aprobado, pendiente, requiere revisión).

### Fase C — Laboratorio de resúmenes
- Como ingeniero ML quiero configurar un resumen seleccionando modelo, longitud, nivel de coloquialidad y tono para ejecutar un job en AWS.
- Como especialista médico quiero revisar el resumen estructurado por secciones y editar la redacción antes de aprobarlo.
- Como PM quiero monitorear en vivo la generación (legibilidad, coherencia, fidelidad) para detectar desviaciones o fallos del LLM.

### Fase D — Evaluación y QA
- Como analista quiero visualizar métricas RAG y comparar con benchmarks (BioLaySumm 2023) para comunicar avances del proyecto.
- Como especialista médico quiero identificar fragmentos marcados como alucinaciones críticas y revisarlos con evidencia de soporte.
- Como auditor quiero descargar los reportes de evaluación (BERTScore, legibilidad, AlignScore) junto con logs de ejecución para documentar el proceso.

### Fase E — Operaciones y gobernanza
- Como ingeniero ML quiero monitorear el estado de stacks AWS y acceder a los logs `maia-run.log` desde la UI para depurar ejecuciones.
- Como administrador quiero controlar quién puede lanzar jobs costosos (AWS/Celery) mediante permisos y ver historial de acciones.
- Como stakeholder quiero recibir notificaciones cuando un job finaliza o falla para informar al equipo académico.

## 6. Checklist de tareas

### Fase 0 — Preparación técnica
- [x] Inicializar aplicación React + Vite (JavaScript) con Ant Design y configurar estructura base (`src/pages`, `src/components`, `src/hooks`).
- [x] Integrar React Router, React Query y estado global (Zustand o Context + reducer) con patrones compartidos.
- [x] Configurar FastAPI base con routers, configuración (`pydantic-settings`), logging estructurado, pruebas smoke.
- [x] Definir conexiones seguras a MongoDB, MLflow, S3, AWS (gestión de credenciales y `.env`).

### Fase 1 — Ingesta y corpus
- [x] Enriquecer `/api/v1/datasets/state` con conteos reales y porcentajes derivados de `metrics.csv`.
- [ ] Conectar vista Dataset BETO con `/api/v1/datasets/tokenized/beto` (paginación, filtros básicos, preview detallado).
- [x] Implementar endpoints `datasets.state` y `corpus.documents` con soporte de filtros y paginación.
- [x] Construir vista Explorador de Corpus con `Table`, `Form`, filtros y preview en `Drawer`/`Tabs` (original, traducción, métricas).
- [x] Habilitar KPIs iniciales en Panel Inicio usando `Card` + `Statistic` y timeline de actividad.
- [x] Añadir sistema de comentarios básicos por documento (uso de `Comment`) y registro de auditoría.

### Fase 2 — Traducción y métricas de calidad
- [x] Implementar `translation.recompute` y `translation.metrics` con cálculo BLEU/chrF, longitud y timestamp.
- [x] Diseñar vista Comparador de Traducciones con `Card`, `Statistic`, resaltado de diferencias y alertas de calidad.
- [x] Incorporar alertas en Panel Inicio para traducciones con baja calidad o pendientes de revisión (`Alert`, `Badge`).
- [x] Permitir edición controlada de traducción con versionado y justificaciones (`Form`, `Modal`).

### Fase 3 — Laboratorio de resúmenes
- [x] Implementar `summaries.jobs` (creación/consulta) y flujo con Celery o integración AWS (stack existente).
- [x] Construir formulario de configuración y viewer estructurado según Figma usando componentes Ant Design.
- [x] Establecer stream SSE/WebSocket para métricas en vivo y mostrarlas con `Progress`, `Statistic`, `Timeline`.
- [x] Versionar resúmenes (historial de ejecuciones, aprobaciones, descargas en PDF/Markdown).

### Fase 4 — Evaluación avanzada y RAG
- [x] Calcular métricas reales (`bertscore`, `alignscore`, `fkgl`, cobertura) desde artefactos locales para `/api/v1/evaluation/overview`.
- [x] Exponer endpoints `evaluation.overview`, `evaluation.alignment`, `rag.metrics` conectados a pipelines existentes.
- [x] Visualizar métricas RAG y comparaciones vs BioLaySumm con gráficas y tablas interactivas.
- [x] Implementar vista de alucinaciones con navegación por texto resaltado y categorías (`Tag`, `List`, `Alert`).
- [x] Integrar AlignScore y métricas adicionales (BERTScore, fact-check) cuando esté disponible el repositorio externo.

### Fase 5 — Operaciones y gobernanza
- [x] Agregar módulo `jobs` unificado con estados Celery/AWS y descargas de logs desde S3 (`Table`, `Timeline`).
- [x] Implementar vistas de Experiment Tracking (MLflow) y costos estimados de ejecuciones AWS con tablas y charts.
- [x] Añadir auditoría completa (acciones, descargas, ejecuciones) y controles de roles/permisos finos (guardas en frontend + backend).
- [x] Preparar despliegue Docker Compose (frontend, FastAPI, Mongo, Redis, MLflow) y pipelines CI/CD.
- [x] Integrar observabilidad (Sentry/OTEL, health checks, alertas por Slack/email).

## 7. Preguntas abiertas y riesgos
- ¿Se mantendrá MongoDB como fuente principal o se migrará a un servicio gestionado (Atlas, DocumentDB) para producción?
- ¿Cómo se expondrán los datos confidenciales o clínicos? Requiere políticas de anonimización y acceso restringido.
- Decidir estrategia de cómputo para generación (solo AWS EC2 + Ollama vs. mezcla con servicios externos como OpenAI).
- Validar costo y latencia aceptable de las ejecuciones AWS (instancias CPU vs GPU) y establecer cuotas por rol.
- Confirmar disponibilidad e instalación de AlignScore y otras métricas propietarias dentro del entorno controlado.
- Planificar sincronización bidireccional: cuando un resumen/traducción se edita manualmente en la UI, ¿cómo se versiona en Mongo o S3?
- Evaluar compatibilidad con futuras integraciones (Opik, agentes autónomos) y definir estrategia de versionado de la API.

## 8. Próximos pasos sugeridos
- Socializar este plan con el equipo del proyecto de grado (datos, clínico, ML, UX) y recoger ajustes prioritarios.
- Derivar épicas e issues en el repositorio organizadas por fases/checklist y asignarlas según roles del equipo.
- Elaborar wireframes detallados (Figma) para vistas adicionales (Comparador, Tracking, Operaciones) y validar con usuarios clave.
- Preparar datasets de prueba anonimizados y scripts de seed para apoyar desarrollo local sin depender de producción.
- Definir cronograma de integración continua (sprints) alineado con hitos académicos y entregables del proyecto de grado.

### Fase F — Datos reales y verificación
- Como científico de datos quiero cargar datasets y métricas reales sin romper el front.
- Como QA necesito scripts reproducibles para poblar Mongo con snapshots DVC/MLflow.
- Como equipo de producto queremos una guía clara para preparar entornos (env vars, imports, pruebas).

Checklist (ver `docs/onboarding_data_pipeline.md`):
- [ ] Instalar dependencias de backend (`pip install -r backend/requirements.txt`).
- [ ] Ejecutar `./scripts/seed_local_dev.sh` (acepta `MONGO_DSN` y `MONGO_DATABASE` si trabajas con otra instancia).
- [ ] Ejecutar `uvicorn app.main:app --reload --factory` y validar endpoints (`curl /datasets/state`, etc.).
- [ ] Lanzar `npm run dev` y confirmar que React muestra datos reales sin banner offline.
