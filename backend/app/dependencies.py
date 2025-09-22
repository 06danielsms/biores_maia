"""Dependencias compartidas para la API."""
from collections.abc import AsyncGenerator
from typing import Optional

from fastapi import Depends

try:  # pragma: no cover - permite ejecutar sin dependencia instalada
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
except ModuleNotFoundError:  # pragma: no cover - fallback en CI liviano
    AsyncIOMotorClient = None  # type: ignore[assignment]
    AsyncIOMotorDatabase = None  # type: ignore[assignment]

from .core.config import settings


async def get_mongo_client() -> AsyncGenerator[Optional[AsyncIOMotorClient], None]:
    if AsyncIOMotorClient is None:
        yield None
        return

    client: Optional[AsyncIOMotorClient] = None
    try:
        client = AsyncIOMotorClient(
            settings.mongo_dsn,
            serverSelectionTimeoutMS=500,
        )
        yield client
    except Exception:  # pragma: no cover - defensivo ante DSN inválido
        yield None
    finally:
        if client is not None:
            client.close()


async def get_database(
    client: Optional[AsyncIOMotorClient] = Depends(get_mongo_client),
) -> AsyncGenerator[Optional[AsyncIOMotorDatabase], None]:
    if client is None:
        yield None
    else:
        yield client[settings.mongo_database]
