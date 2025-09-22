#!/usr/bin/env python
"""Ingresa documentos del corpus en MongoDB.

Permite cargar archivos JSON o JSONL con metadatos, textos originales,
traducciones y métricas para que el endpoint `/corpus/documents`
recupere información real.

Formato esperado (JSON):
[
  {
    "id": "CT-2024-0001",
    "file_name": "CT-2024-0001.json",
    "source": "ClinicalTrials.gov",
    "language": "en",
    "translated": true,
    "metrics_ready": true,
    "tokens": 1874,
    "domain": "Cardiología",
    "alignment_risk": "bajo",
    "updated_at": "2024-11-21T22:14:00Z",
    "original": "...",
    "translation": "...",
    "metrics": {"bleu": 54.9},
    "comments": [{"author": "María", "timestamp": "2024-11-22T10:30:00Z"}]
  }
]
"""
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
    raise SystemExit("Instala 'motor' para ejecutar este script (pip install motor)") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa documentos del corpus a MongoDB")
    parser.add_argument("--input", type=Path, required=True, help="Archivo JSON/JSONL con documentos")
    parser.add_argument("--mongo", type=str, default="mongodb://localhost:27017", help="DSN de MongoDB")
    parser.add_argument("--database", type=str, default="health_literacy_db", help="Base de datos destino")
    parser.add_argument("--collection", type=str, default="corpus_documents", help="Colección destino")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Elimina la colección antes de insertar (para recargas completas)",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                return list(data["items"])
            return [data]
        if isinstance(data, list):
            return list(data)
        raise ValueError("El JSON debe contener una lista o un objeto con clave 'items'.")
    if suffix in {".jsonl", ".ndjson"}:
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


def normalise(document: dict[str, Any]) -> dict[str, Any]:
    data = dict(document)
    doc_id = data.get("id") or data.get("document_id")
    if not doc_id:
        raise ValueError("Cada documento debe tener campo 'id'.")

    data["id"] = str(doc_id)
    data.setdefault("file_name", f"{doc_id}.json")
    data.setdefault("translated", bool(data.get("translation")))
    data.setdefault("metrics_ready", bool(data.get("metrics")))
    data.setdefault("alignment_risk", "pendiente")
    data.setdefault("tokens", int(data.get("tokens", 0)))

    if "updated_at" in data:
        data["updated_at"] = _to_datetime(data["updated_at"])
    else:
        data["updated_at"] = datetime.utcnow()

    if "comments" in data and isinstance(data["comments"], list):
        comments = []
        for comment in data["comments"]:
            if not isinstance(comment, dict):
                continue
            normalised = dict(comment)
            if "timestamp" in normalised:
                normalised["timestamp"] = _to_datetime(normalised["timestamp"])
            comments.append(normalised)
        data["comments"] = comments

    if "metrics" in data and isinstance(data["metrics"], dict):
        metrics = {}
        for key, value in data["metrics"].items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                metrics[key] = value
        data["metrics"] = metrics

    return data


async def import_documents(args: argparse.Namespace) -> None:
    records = load_records(args.input)
    logger.info("Documentos detectados: %s", len(records))
    if not records:
        logger.warning("No se encontraron documentos en %s", args.input)
        return

    documents = []
    for raw in records:
        try:
            documents.append(normalise(raw))
        except ValueError as exc:
            logger.warning("Documento omitido: %s", exc)

    client = AsyncIOMotorClient(args.mongo)
    collection = client[args.database][args.collection]

    if args.drop:
        await collection.delete_many({})
        await client[args.database]["document_comments"].delete_many({})

    for document in documents:
        comments = document.pop("comments", [])
        await collection.replace_one({"id": document["id"]}, document, upsert=True)
        if comments:
            for comment in comments:
                comment_doc = dict(comment)
                comment_doc["document_id"] = document["id"]
                await client[args.database]["document_comments"].insert_one(comment_doc)

    await collection.create_index("id", unique=True)
    logger.info(
        "Se importaron %s documentos en %s.%s",
        len(documents),
        args.database,
        args.collection,
    )
    client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(import_documents(args))


if __name__ == "__main__":
    main()
