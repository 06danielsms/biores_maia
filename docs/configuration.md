# Configuración y seguridad

Este documento resume las variables de entorno disponibles y recomendaciones para proteger credenciales cuando se despliega BIORES Maia.

## Variables backend (`backend/.env`)

| Variable | Descripción | Ejemplo |
| --- | --- | --- |
| `APP_NAME` | Nombre para encabezados y logs. | `Biores Maia API` |
| `ENVIRONMENT` | Identificador de entorno (`local`, `staging`, `prod`). | `prod` |
| `API_DEBUG` | Activa modo debug (solo entornos locales). | `false` |
| `MONGO_DSN` | Cadena de conexión MongoDB (usar usuario/contraseña o SRV seguro). | `mongodb+srv://user:pass@cluster.mongodb.net/health` |
| `MONGO_DATABASE` | Base de datos usada por el backend. | `health_literacy_db` |
| `REDIS_DSN` | Broker/cache para workers. Recomendado Redis con TLS. | `rediss://:<pass>@redis.example.com:6379/0` |
| `MLFLOW_TRACKING_URI` | URL del servidor MLflow. | `https://mlflow.example.com` |
| `S3_BUCKET_RESULTS` | Bucket donde se guardan reportes/resultados. | `biores-maia-results` |
| `SENTRY_DSN` | DSN de Sentry para monitoreo. | `https://<key>@sentry.io/<project>` |
| `SLACK_WEBHOOK_URL` | Webhook entrante usado por notificaciones. | `https://hooks.slack.com/services/...` |

**Recomendaciones**
- Nunca commitear `.env` reales. Utilizar gestores de secretos (AWS Secrets Manager, Vault, Doppler) o variables protegidas en el sistema CI/CD.
- Para conexiones a MongoDB o Redis, preferir TLS y usuarios con privilegios limitados.
- Al usar S3, otorgar políticas específicas por bucket y carpeta.

## Frontend (`frontend/.env`)

| Variable | Descripción |
| --- | --- |
| `VITE_API_URL` | Base URL de la API (`https://api.example.com/api/v1`). |
| `VITE_ENVIRONMENT` | Nombre del entorno mostrado en UI. |
| `VITE_MLFLOW_URL` | URL pública del dashboard MLflow. |

## Certificados y secretos
- En producción usar HTTPS (TLS) para la API, MLflow y Redis.
- Gestionar certificados con ACM/LetsEncrypt o proxies (ALB, Nginx, Traefik).

## Observabilidad
- Con `SENTRY_DSN` el backend reporta errores automáticamente.
- Para expandir observabilidad: integrar OpenTelemetry con exporters (OTLP, Jaeger) y health checks (`/health/live`, `/api/v1/health/`).

## Slack / Alertas
- El backend envía notificaciones cuando se recomputa una traducción o se crea un job de resumen si `SLACK_WEBHOOK_URL` está configurada.
- Ajustar el canal y las políticas de rate-limit directamente en Slack.

## CI/CD
- El workflow `.github/workflows/ci.yml` ejecuta pruebas backend/frontend. Añadir jobs adicionales para despliegues según repositorio/infraestructura.
