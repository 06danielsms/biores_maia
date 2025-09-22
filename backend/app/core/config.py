"""Configuración central de la aplicación FastAPI."""
from functools import lru_cache
from typing import Any, Optional

try:  # pragma: no cover - compatibilidad con Pydantic < 2
    from pydantic import Field, field_validator
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import Field, validator

    _PYDANTIC_V2 = False

    def field_validator(field_name: str, *, mode: str = "after"):
        pre = mode == "before"

        def decorator(func):
            return validator(field_name, pre=pre, allow_reuse=True)(func)

        return decorator

try:  # pragma: no cover - pydantic-settings >=2
    from pydantic_settings import BaseSettings, SettingsConfigDict
    _HAS_SETTINGS_DICT = True
except ImportError:  # pragma: no cover
    from pydantic import BaseSettings

    _HAS_SETTINGS_DICT = False


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Valores de configuración expuestos a la aplicación."""

    if _HAS_SETTINGS_DICT:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )
    else:  # pragma: no cover - compatibilidad Pydantic 1
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"

    app_name: str = Field(default="Biores Maia API", validation_alias="APP_NAME")
    environment: str = Field(default="local", validation_alias="ENVIRONMENT")
    debug: bool = Field(default=True, validation_alias="API_DEBUG")
    api_prefix: str = Field(default="/api/v1", validation_alias="API_PREFIX")

    mongo_dsn: str = Field(default="mongodb://mongo:27017", validation_alias="MONGO_DSN")
    mongo_database: str = Field(default="health_literacy_db", validation_alias="MONGO_DATABASE")
    redis_dsn: str = Field(default="redis://redis:6379/0", validation_alias="REDIS_DSN")

    mlflow_tracking_uri: Optional[str] = Field(
        default=None, validation_alias="MLFLOW_TRACKING_URI"
    )
    s3_bucket_results: Optional[str] = Field(
        default=None, validation_alias="S3_BUCKET_RESULTS"
    )
    sentry_dsn: Optional[str] = Field(default=None, validation_alias="SENTRY_DSN")
    slack_webhook_url: Optional[str] = Field(default=None, validation_alias="SLACK_WEBHOOK_URL")
    allowed_cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5174",
        ],
        validation_alias="CORS_ALLOWED_ORIGINS",
    )

    tokenized_dataset_root: str = Field(
        default="data/tokenized", validation_alias="TOKENIZED_DATA_ROOT"
    )
    tokenized_dataset_name: str = Field(
        default="beto_dataset", validation_alias="TOKENIZED_DATASET_NAME"
    )
    tokenized_dataset_bucket: Optional[str] = Field(
        default="biores-maia-data-clean", validation_alias="TOKENIZED_DATA_BUCKET"
    )
    tokenized_dataset_prefix: str = Field(
        default="tokenized/beto_dataset", validation_alias="TOKENIZED_DATA_PREFIX"
    )
    tokenized_label_map: dict[int, str] = Field(
        default_factory=lambda: {0: "pls", 1: "tech"},
        validation_alias="TOKENIZED_LABEL_MAP",
    )

    corpus_s3_bucket: Optional[str] = Field(
        default="biores-project", validation_alias="CORPUS_S3_BUCKET"
    )
    corpus_s3_prefix: str = Field(
        default="data/files/md5", validation_alias="CORPUS_S3_PREFIX"
    )
    corpus_source_name: str = Field(
        default="biores-project", validation_alias="CORPUS_SOURCE_NAME"
    )
    corpus_default_language: str = Field(
        default="en", validation_alias="CORPUS_DEFAULT_LANGUAGE"
    )
    corpus_default_translation_placeholder: str = Field(
        default="Traducción pendiente de ingestión.",
        validation_alias="CORPUS_TRANSLATION_PLACEHOLDER",
    )

    local_corpus_file: str = Field(
        default="seed/corpus_sample.json", validation_alias="LOCAL_CORPUS_FILE"
    )
    local_translation_metrics_csv: str = Field(
        default="results/metrics_sample_es.csv",
        validation_alias="LOCAL_TRANSLATION_METRICS_CSV",
    )
    local_translation_history_file: str = Field(
        default="seed/translation_history.json",
        validation_alias="LOCAL_TRANSLATION_HISTORY_FILE",
    )
    local_alignment_findings_file: str = Field(
        default="seed/alignment_findings.json",
        validation_alias="LOCAL_ALIGNMENT_FINDINGS_FILE",
    )
    local_summary_jobs_file: str = Field(
        default="seed/summary_jobs.json",
        validation_alias="LOCAL_SUMMARY_JOBS_FILE",
    )

    @field_validator("allowed_cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return _split_csv(value)
        if isinstance(value, list):
            return value
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5174",
        ]


    @field_validator("tokenized_label_map", mode="before")
    @classmethod
    def parse_label_map(cls, value: Any) -> dict[int, str]:
        if value in (None, ""):
            return {0: "pls", 1: "tech"}
        if isinstance(value, dict):
            return {int(key): str(val) for key, val in value.items()}
        if isinstance(value, str):
            mapping: dict[int, str] = {}
            for chunk in value.split(","):
                if not chunk.strip():
                    continue
                if "=" not in chunk:
                    raise ValueError(
                        "Invalid tokenized label map entry; use 'id=label' format"
                    )
                key_str, label = chunk.split("=", 1)
                key_str = key_str.strip()
                label = label.strip()
                if not key_str:
                    raise ValueError("Tokenized label map keys must be numeric")
                mapping[int(key_str)] = label
            return mapping or {0: "pls", 1: "tech"}
        raise ValueError("Unsupported tokenized label map format")


@lru_cache
def get_settings() -> Settings:
    """Devuelve una única instancia de Settings en formato cacheado."""

    return Settings()  # type: ignore[arg-type]


settings = get_settings()
