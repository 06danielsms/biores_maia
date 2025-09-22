#!/usr/bin/env python
"""Carga el estado de datasets en MongoDB.

Permite transformar archivos JSON/CSV con metadatos de datasets y 
subirlos a la colección `datasets_state` para que los endpoints 
`/datasets/state` sirvan datos reales.

Ejemplo:
    python scripts/import_dataset_state.py \
        --input data/datasets_state.json \
        --mongo "mongodb://localhost:27017" \
        --database health_literacy_db
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "La dependencia 'motor' es requerida. Instala con 'pip install motor'."
    ) from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa datasets a MongoDB")
    parser.add_argument("--input", type=Path, required=True, help="Archivo JSON/JSONL/CSV con datasets")
    parser.add_argument("--mongo", type=str, default="mongodb://localhost:27017", help="DSN de MongoDB")
    parser.add_argument("--database", type=str, default="health_literacy_db", help="Base de datos destino")
    parser.add_argument("--collection", type=str, default="datasets_state", help="Colección destino")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Elimina la colección antes de insertar (útil para cargas completas)",
    )
    return parser.parse_args()


def _load_json_records(path: Path) -> List[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "datasets" in data and isinstance(data["datasets"], list):
            return list(data["datasets"])
        return [data]
    if isinstance(data, list):
        return list(data)
    raise ValueError("El JSON debe contener una lista o la clave 'datasets'.")


def _load_jsonl_records(path: Path) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_csv_records(path: Path) -> List[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_records(path: Path) -> List[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl_records(path)
    if suffix == ".csv":
        return _load_csv_records(path)
    raise ValueError(f"Formato no soportado: {suffix}. Usa JSON, JSONL o CSV.")


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            pass
    return datetime.utcnow()


def normalise(record: dict[str, Any]) -> dict[str, Any]:
    data = dict(record)
    name = data.get("name") or data.get("dataset")
    if not name:
        raise ValueError("Cada dataset debe tener campo 'name'.")

    data["name"] = str(name)
    data["status"] = str(data.get("status", "synced"))
    if "last_sync" in data:
        data["last_sync"] = _to_datetime(data["last_sync"])
    else:
        data["last_sync"] = datetime.utcnow()

    if "updated_at" in data:
        data["updated_at"] = _to_datetime(data["updated_at"])
    else:
        data["updated_at"] = datetime.utcnow()

    for numeric_key in ("document_count", "translated_pct", "metrics_ready_pct"):
        if numeric_key in data and data[numeric_key] is not None:
            try:
                data[numeric_key] = float(data[numeric_key])
            except (TypeError, ValueError):
                logger.debug("Campo %s no convertible a float: %s", numeric_key, data[numeric_key])

    return data


async def import_datasets(args: argparse.Namespace) -> None:
    records = load_records(args.input)
    logger.info("Registros detectados: %s", len(records))
    if not records:
        logger.warning("No se encontraron datasets en %s", args.input)
        return

    datasets = []
    for raw in records:
        try:
            datasets.append(normalise(raw))
        except ValueError as exc:
            logger.warning("Dataset omitido: %s", exc)

    client = AsyncIOMotorClient(args.mongo)
    collection = client[args.database][args.collection]

    if args.drop:
        await collection.delete_many({})

    for dataset in datasets:
        await collection.replace_one({"name": dataset["name"]}, dataset, upsert=True)

    await collection.create_index("name", unique=True)
    logger.info("Se importaron %s datasets en %s.%s", len(datasets), args.database, args.collection)
    client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(import_datasets(args))


if __name__ == "__main__":
    main()
