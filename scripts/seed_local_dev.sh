#!/usr/bin/env bash
set -euo pipefail

MONGO_DSN=${MONGO_DSN:-mongodb://localhost:27017}
MONGO_DB=${MONGO_DATABASE:-health_literacy_db}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SEED_DIR="$PROJECT_ROOT/seed"

# If USE_DOCKER=1 is set, we'll attempt to run import scripts inside the backend container
USE_DOCKER=${USE_DOCKER:-0}

if ! command -v python >/dev/null 2>&1; then
  echo "Python no está disponible en PATH" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

run_import() {
  local script=$1
  shift
  echo "→ Ejecutando ${script} $*"
  python "${PROJECT_ROOT}/scripts/${script}" "$@"
}

# Quick runtime check for required Python package 'motor'
check_python_pkg() {
  local pkg=$1
  python - <<PY
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("$pkg") is not None else 1)
PY
}

if [ "$USE_DOCKER" = "1" ]; then
  echo "USE_DOCKER=1 -> intentaremos ejecutar los importadores dentro del contenedor backend (docker compose)."
  echo "Asegúrate de tener docker y docker compose disponibles y el servicio 'mongo' levantado."
  echo "Ejecutando: docker compose run --rm backend ./scripts/seed_local_dev.sh"
  docker compose run --rm backend ./scripts/seed_local_dev.sh
  exit $?
fi

if ! check_python_pkg motor; then
  cat <<MSG
La dependencia 'motor' no está disponible en el entorno Python que invoca este script.

Opciones para solucionarlo:

1) Instalar dependencias localmente (recomendado si trabajas fuera de Docker):
   cd backend
   pip install -r requirements.txt
   cd ..
   ./scripts/seed_local_dev.sh

2) Usar Docker Compose (ejecuta los importadores dentro del contenedor backend):
   # Si usas Docker Compose, primero levanta los servicios necesarios (mongo, etc.):
   docker compose up -d mongo mlflow redis
   # Luego ejecuta los importadores dentro del contenedor backend:
   docker compose run --rm backend ./scripts/seed_local_dev.sh

Si ya tienes 'motor' instalado y sigues viendo este mensaje, revisa que 'python' en PATH sea el que corresponde a tu entorno virtual.
MSG
  exit 1
fi

run_import import_dataset_state.py --input "$SEED_DIR/datasets_state.json" --mongo "$MONGO_DSN" --database "$MONGO_DB" --drop
run_import import_corpus_documents.py --input "$SEED_DIR/corpus_sample.json" --mongo "$MONGO_DSN" --database "$MONGO_DB" --drop
run_import import_jobs_status.py --input "$SEED_DIR/jobs_status.json" --mongo "$MONGO_DSN" --database "$MONGO_DB" --drop
run_import import_evaluation_overview.py --input "$SEED_DIR/evaluation_overview.json" --mongo "$MONGO_DSN" --database "$MONGO_DB" --drop

python "$PROJECT_ROOT/scripts/import_seed_insights.py" --mongo "$MONGO_DSN" --database "$MONGO_DB"

echo "Seeds cargados en ${MONGO_DB} (${MONGO_DSN})"
