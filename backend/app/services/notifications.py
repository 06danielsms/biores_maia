"""Integraciones de notificación (Slack, etc.)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


async def _post_slack(payload: Dict[str, Any]) -> None:
    if not settings.slack_webhook_url:
        logger.debug("Slack webhook no configurado, se omite notificación")
        return

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(settings.slack_webhook_url, json=payload)
            response.raise_for_status()
    except Exception as exc:  # pragma: no cover - logging preventiva
        logger.warning("Error enviando notificación a Slack: %s", exc)


def notify_slack_async(message: str, *, blocks: list[dict[str, Any]] | None = None) -> None:
    """Envía una notificación a Slack en segundo plano si hay webhook."""

    payload: Dict[str, Any] = {"text": message}
    if blocks:
        payload["blocks"] = blocks

    asyncio.create_task(_post_slack(payload))
