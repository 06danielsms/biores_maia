#!/usr/bin/env python
"""Importa métricas agregadas de evaluación a MongoDB."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Instala 'motor' para usar este script") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa overview de evaluación")
    parser.add_argument("--input", type=Path, required=True, help="Archivo JSON con métricas")
    parser.add_argument("--mongo", type=str, default="mongodb://localhost:27017")
    parser.add_argument("--database", type=str, default="health_literacy_db")
    parser.add_argument("--collection", type=str, default="evaluation_overview")
    parser.add_argument("--drop", action="store_true", help="Eliminar colección previa")
    return parser.parse_args()


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


def normalise(data: dict[str, Any]) -> dict[str, Any]:
    record = dict(data)
    record["generated_at"] = _to_datetime(record.get("generated_at"))
    alerts = record.get("alerts")
    if isinstance(alerts, list):
        normalised_alerts = []
        for alert in alerts:
            if not isinstance(alert, dict):
                continue
            normal = dict(alert)
            if "detected_at" in normal:
                normal["detected_at"] = _to_datetime(normal["detected_at"])
            normalised_alerts.append(normal)
        record["alerts"] = normalised_alerts
    return record


async def import_overview(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "metrics" in payload:
        payloads = [payload]
    elif isinstance(payload, list):
        payloads = payload
    else:
        raise ValueError("El JSON debe ser un objeto con 'metrics' o una lista de ellos")

    client = AsyncIOMotorClient(args.mongo)
    collection = client[args.database][args.collection]

    if args.drop:
        await collection.delete_many({})

    normalised = [normalise(item) for item in payloads]
    if normalised:
        await collection.insert_many(normalised)
        logger.info("Se importaron %s registros en %s.%s", len(normalised), args.database, args.collection)
    else:
        logger.info("No se importó ningún overview")

    client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(import_overview(args))


if __name__ == "__main__":
    main()
