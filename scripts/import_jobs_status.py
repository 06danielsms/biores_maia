#!/usr/bin/env python
"""Importa estados de jobs (Celery/AWS) a MongoDB."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Instala 'motor' para usar este script") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa jobs activos a MongoDB")
    parser.add_argument("--input", type=Path, required=True, help="Archivo JSON/JSONL con jobs")
    parser.add_argument("--mongo", type=str, default="mongodb://localhost:27017")
    parser.add_argument("--database", type=str, default="health_literacy_db")
    parser.add_argument("--collection", type=str, default="jobs_status")
    parser.add_argument("--drop", action="store_true", help="Elimina la colección antes de insertar")
    return parser.parse_args()


def load_records(path: Path) -> List[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.get("jobs", []))
        if isinstance(data, list):
            return list(data)
    elif suffix in {".jsonl", ".ndjson"}:
        records: List[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    raise ValueError(f"Formato no soportado: {suffix}. Usa JSON o JSONL.")


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


def normalise(job: dict[str, Any]) -> dict[str, Any]:
    data = dict(job)
    if "job" not in data:
        raise ValueError("Cada registro debe tener una clave 'job'.")

    data["job"] = str(data["job"])
    data["state"] = str(data.get("state", "in_progress"))

    if "started_at" in data:
        data["started_at"] = _to_datetime(data["started_at"])
    else:
        data["started_at"] = datetime.utcnow()

    if "last_heartbeat" in data:
        data["last_heartbeat"] = _to_datetime(data["last_heartbeat"])

    return data


async def import_jobs(args: argparse.Namespace) -> None:
    records = load_records(args.input)
    jobs = []
    for raw in records:
        try:
            jobs.append(normalise(raw))
        except ValueError as exc:
            logger.warning("Job omitido: %s", exc)

    client = AsyncIOMotorClient(args.mongo)
    collection = client[args.database][args.collection]

    if args.drop:
        await collection.delete_many({})

    if jobs:
        await collection.insert_many(jobs)
        await collection.create_index("job")
        logger.info("Se importaron %s jobs", len(jobs))
    else:
        logger.info("No se importaron jobs.")

    client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(import_jobs(args))


if __name__ == "__main__":
    main()
