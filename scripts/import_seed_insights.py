#!/usr/bin/env python
"""Carga artefactos de `seed/` y métricas locales en MongoDB.

Popula colecciones usadas por el frontend cuando no existe un backend
productivo: métricas de traducción por documento, jobs de resúmenes,
hallazgos de alineación, costos de jobs y métricas RAG agregadas.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import sys

from motor.motor_asyncio import AsyncIOMotorClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = REPO_ROOT / "seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Importa artefactos seed a MongoDB")
    parser.add_argument("--mongo", type=str, default="mongodb://localhost:27017")
    parser.add_argument("--database", type=str, default="health_literacy_db")
    parser.add_argument(
        "--corpus-file",
        type=Path,
        default=SEED_DIR / "corpus_sample.json",
        help="Corpus de documentos con métricas integradas",
    )
    parser.add_argument(
        "--translation-history",
        type=Path,
        default=SEED_DIR / "translation_history.json",
        help="Historial de jobs de traducción por documento",
    )
    parser.add_argument(
        "--summary-jobs",
        type=Path,
        default=SEED_DIR / "summary_jobs.json",
        help="Estados históricos de summary jobs",
    )
    parser.add_argument(
        "--alignment-findings",
        type=Path,
        default=SEED_DIR / "alignment_findings.json",
        help="Hallazgos de alineación para dashboards de QA",
    )
    return parser.parse_args()


def _parse_timestamp(value: Any) -> Any:
    if isinstance(value, datetime) or value in (None, ""):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return value
    return value


def load_translation_history(path: Path) -> dict[str, list[dict[str, Any]]]:
    history: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return history
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, Iterable):  # type: ignore[truthy-function]
        return history
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        doc_id = str(raw.get("document_id") or "").strip()
        if not doc_id:
            continue
        entry = dict(raw)
        entry["timestamp"] = _parse_timestamp(entry.get("timestamp"))
        history.setdefault(doc_id, []).append(entry)
    return history


def build_translation_docs(
    corpus_path: Path, history_by_doc: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    if not corpus_path.exists():
        return []
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    docs: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        doc_id = entry.get("id")
        if not doc_id:
            continue
        metrics = dict(entry.get("metrics") or {})
        if "fkgl" not in metrics and entry.get("readability_fkgl") is not None:
            metrics["fkgl"] = entry["readability_fkgl"]
        docs.append(
            {
                "document_id": doc_id,
                "source": entry.get("source"),
                "metrics": metrics,
                "history": history_by_doc.get(doc_id, []),
                "generated_at": datetime.utcnow(),
            }
        )
    return docs


def load_summary_jobs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record["started_at"] = _parse_timestamp(record.get("started_at"))
        timeline = []
        for raw_event in record.get("timeline") or []:
            if not isinstance(raw_event, dict):
                continue
            event = dict(raw_event)
            event["timestamp"] = _parse_timestamp(event.get("timestamp"))
            timeline.append(event)
        record["timeline"] = timeline
        jobs.append(record)
    return jobs


def load_alignment_findings(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        record["detected_at"] = _parse_timestamp(record.get("detected_at"))
        events = []
        for raw in record.get("hallucinations") or []:
            if not isinstance(raw, dict):
                continue
            entry = dict(raw)
            entry["timestamp"] = _parse_timestamp(entry.get("timestamp"))
            events.append(entry)
        record["hallucinations"] = events
        findings.append(record)
    return findings


def compute_jobs_costs_and_rag() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sys_path_added = False
    backend_path = REPO_ROOT / "backend"
    if str(backend_path) not in sys.path:
        sys.path.append(str(backend_path))
        sys_path_added = True
    try:
        from app.services.local_metrics import _load_metrics_summary, get_rag_summary

        summary = _load_metrics_summary()
        rag_payload = get_rag_summary() or {"datasets": []}
        if summary is None:
            return [], rag_payload
        rag_payload.setdefault("generated_at", summary.generated_at.isoformat())

        costs: list[dict[str, Any]] = []
        for snapshot in summary.datasets:
            monthly_cost = round(snapshot.total * 0.42, 2)
            trend_delta = snapshot.translated_pct() - 50.0
            costs.append(
                {
                    "service": snapshot.name,
                    "monthly_cost": monthly_cost,
                    "trend": f"{trend_delta:+.1f}%",
                }
            )
        return costs, rag_payload
    finally:
        if sys_path_added:
            sys.path.remove(str(backend_path))


async def main() -> None:
    args = parse_args()

    history_by_doc = load_translation_history(args.translation_history)
    translation_docs = build_translation_docs(args.corpus_file, history_by_doc)
    summary_jobs = load_summary_jobs(args.summary_jobs)
    alignment_findings = load_alignment_findings(args.alignment_findings)
    jobs_costs, rag_payload = compute_jobs_costs_and_rag()

    client = AsyncIOMotorClient(args.mongo)
    db = client[args.database]

    if translation_docs:
        coll = db["translation_metrics"]
        await coll.drop()
        await coll.insert_many(translation_docs)
        await coll.create_index("document_id", unique=True)

    if summary_jobs:
        coll = db["summary_jobs"]
        await coll.drop()
        await coll.insert_many(summary_jobs)
        await coll.create_index("job_id", unique=True)

    if alignment_findings:
        coll = db["alignment_findings"]
        await coll.drop()
        await coll.insert_many(alignment_findings)
        await coll.create_index("summary_id", unique=True)

    if jobs_costs:
        coll = db["jobs_costs"]
        await coll.drop()
        await coll.insert_many(jobs_costs)

    if rag_payload:
        coll = db["rag_metrics"]
        await coll.drop()
        await coll.insert_one(rag_payload)

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
