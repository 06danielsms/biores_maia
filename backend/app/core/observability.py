"""Inicialización de observabilidad (Sentry, logging, etc.)."""
from __future__ import annotations

import logging
from typing import Optional

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
except ImportError:  # pragma: no cover - dependencia opcional
    sentry_sdk = None  # type: ignore[assignment]

from .config import settings


def init_observability() -> None:
    """Configura integraciones opcionales como Sentry."""

    if settings.sentry_dsn and sentry_sdk is not None:
        logging_integration = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR,
        )
        sentry_sdk.init(  # type: ignore[attr-defined]
            dsn=settings.sentry_dsn,
            environment=settings.environment,
            integrations=[FastApiIntegration(), logging_integration],
            traces_sample_rate=0.1,
        )
