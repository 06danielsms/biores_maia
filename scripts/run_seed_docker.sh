#!/usr/bin/env bash
set -euo pipefail

# Helper para levantar mongo y ejecutar los importadores dentro del contenedor backend.
# Uso: ./scripts/run_seed_docker.sh

# Ajustes
COMPOSE_CMD=${COMPOSE_CMD:-docker compose}
SERVICES_TO_UP=${SERVICES_TO_UP:-mongo}
BACKEND_SERVICE=${BACKEND_SERVICE:-backend}
DB_NAME=${MONGO_DATABASE:-health_literacy_db}

echo "1) Levantando servicios necesarios: ${SERVICES_TO_UP}"
$COMPOSE_CMD up -d $SERVICES_TO_UP

echo "2) (Opcional) construir la imagen backend para asegurar dependencias"
$COMPOSE_CMD build $BACKEND_SERVICE || true

echo "3) Ejecutando el seeding dentro del contenedor backend"
$COMPOSE_CMD run --rm $BACKEND_SERVICE ./scripts/seed_local_dev.sh

# Pequeña comprobación: listar colecciones en la BD
echo "4) Comprobando colecciones en MongoDB (${DB_NAME})"
$COMPOSE_CMD exec mongo mongosh --eval "db.getSiblingDB('${DB_NAME}').getCollectionNames().forEach(c => print(c))"

echo "Seeding completado. Si quieres, levanta backend y frontend con docker compose up -d backend frontend"
