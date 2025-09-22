#!/usr/bin/env python
"""Carga métricas de traducción desde CSV a MongoDB.

Uso:
    python scripts/import_translation_metrics.py \
        --csv results/metrics_sample_es.csv \
        --mongo "mongodb://localhost:27017" \
        --database health_literacy_db
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa métricas de traducción a MongoDB")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Ruta al CSV con métricas (resultado de scripts de análisis)",
    )
    parser.add_argument(
        "--mongo",
        type=str,
        default="mongodb://localhost:27017",
        help="DSN de MongoDB",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="health_literacy_db",
        help="Nombre de la base de datos",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="translation_metrics",
        help="Nombre de la colección destino",
    )
    return parser.parse_args()


def build_metric_document(row: dict[str, str]) -> dict[str, Any]:
    document_id = row.get("file") or row.get("document_id")
    if not document_id:
        raise ValueError("column 'file' o 'document_id' es obligatoria en el CSV")

    metrics = {}
    for key, value in row.items():
        if key in {"file", "document_id", "source", "group"}:
            continue
        if value is None or value == "":
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value

    return {
        "document_id": document_id,
        "source": row.get("source"),
        "group": row.get("group"),
        "metrics": metrics,
        "generated_at": datetime.utcnow(),
    }


async def import_metrics(args: argparse.Namespace) -> None:
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    client = AsyncIOMotorClient(args.mongo)
    db = client[args.database]
    collection = db[args.collection]

    documents = []
    with args.csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                documents.append(build_metric_document(row))
            except ValueError as exc:
                logger.warning("Fila omitida: %s", exc)

    if not documents:
        logger.info("No se encontraron filas válidas en %s", args.csv)
        return

    await collection.create_index("document_id")
    await collection.insert_many(documents)
    logger.info("Importadas %s métricas en %s.%s", len(documents), args.database, args.collection)

    client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(import_metrics(args))


if __name__ == "__main__":
    main()
