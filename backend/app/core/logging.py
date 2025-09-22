"""Configuración de logging estructurado para FastAPI."""
import logging.config
from typing import Any, Dict


def setup_logging() -> None:
    """Configura logging básico con estructura uniforme."""

    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": "INFO",
            }
        },
        "root": {
            "handlers": ["console"],
            "level": "INFO",
        },
    }

    logging.config.dictConfig(logging_config)
