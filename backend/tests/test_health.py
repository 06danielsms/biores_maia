"""Pruebas para los endpoints de salud."""
import asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


def test_health_endpoint_returns_ok() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/api/v1/health/")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "Biores Maia API"

    asyncio.run(_run())


def test_live_probe() -> None:
    async def _run() -> None:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get("/health/live")

        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    asyncio.run(_run())
