"""Pruebas básicas para los endpoints de la API."""
import asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


def test_datasets_state_returns_list() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/api/v1/datasets/state")

        assert response.status_code == 200
        body = response.json()
        assert "datasets" in body
        assert isinstance(body["datasets"], list)

    asyncio.run(_run())


def test_tokenized_dataset_slice() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get(
                "/api/v1/datasets/tokenized/beto", params={"split": "train", "limit": 3}
            )

        assert response.status_code == 200
        body = response.json()
        assert body["items"]
        first = body["items"][0]
        assert "text" in first
        assert "label" in first
        assert "token_length" in first

    asyncio.run(_run())


def test_corpus_documents_filters() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/api/v1/corpus/documents", params={"status": "translated"})

        assert response.status_code == 200
        body = response.json()
        assert "items" in body
        assert body["items"]

    asyncio.run(_run())


def test_translation_metrics_and_recompute() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            metrics_response = await client.get("/api/v1/translation/metrics/CT-2024-0001")
            recompute_response = await client.post(
                "/api/v1/translation/recompute",
                json={"document_id": "CT-2024-0001", "model": "helsinki"},
            )

        assert metrics_response.status_code == 200
        assert "metrics" in metrics_response.json()
        assert recompute_response.status_code == 202

    asyncio.run(_run())


def test_summaries_job() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/api/v1/summaries/jobs/job-123")

        assert response.status_code == 200
        assert response.json()["status"] == "streaming"

    asyncio.run(_run())


def test_evaluation_endpoints() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            overview = await client.get("/api/v1/evaluation/overview")
            alignment = await client.get("/api/v1/evaluation/alignment/TECH-332")
            rag = await client.get("/api/v1/evaluation/rag/metrics")

        assert overview.status_code == 200
        assert "metrics" in overview.json()
        assert alignment.status_code == 200
        assert rag.status_code == 200

    asyncio.run(_run())


def test_jobs_status_and_costs() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            status_response = await client.get("/api/v1/jobs/status")
            costs_response = await client.get("/api/v1/jobs/costs")

        assert status_response.status_code == 200
        assert costs_response.status_code == 200
        assert "jobs" in status_response.json()
        assert "services" in costs_response.json()

    asyncio.run(_run())
