# config.py
"""Конфигурация оркестратора с поддержкой fallback-моделей."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as exc:
    raise SystemExit("Не установлен pydantic-settings. Выполните: pip install pydantic-settings") from exc


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- API КЛЮЧИ ---
    openrouter_api_key: str = Field(..., min_length=1)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    deepseek_api_key: Optional[str] = Field(default=None)
    deepseek_base_url: str = Field(default="https://api.deepseek.com")

    # --- AGNES CLI -----------------------------------------------------------
    agnes_api_key: str = Field(..., min_length=1, description="API ключ Agnes")
    agnes_cli_path: str = Field(default="agnes", description="Путь к исполняемому файлу Agnes CLI")
    agnes_model: str = Field(default="agnes/agnes-2.5-flash", description="Модель Agnes для выполнения")
    agnes_timeout: int = Field(default=300, gt=0, description="Таймаут Agnes в секундах")

    # --- МОДЕЛИ И FALLBACK (читаем как строку, чтобы избежать JSON-ошибок pydantic-settings) ---
    fallback_planner_models_str: str = Field(
        default="qwen/qwen-2.5-72b-instruct,deepseek/deepseek-chat",
        alias="FALLBACK_PLANNER_MODELS"
    )
    fallback_reviewer_models_str: str = Field(
        default="qwen/qwen-2.5-72b-instruct,deepseek/deepseek-chat",
        alias="FALLBACK_REVIEWER_MODELS"
    )

    @property
    def planner_models(self) -> list[str]:
        return [m.strip() for m in self.fallback_planner_models_str.split(",") if m.strip()]

    @property
    def reviewer_models(self) -> list[str]:
        return [m.strip() for m in self.fallback_reviewer_models_str.split(",") if m.strip()]

    # --- ПРОЕКТ И ЛИМИТЫ ---
    default_project_path: Optional[str] = Field(default=None)
    log_level: str = Field(default="INFO")
    max_steps: int = Field(default=10, gt=0)

    max_retries: int = Field(default=2, gt=0)
    retry_delay: float = Field(default=5.0, gt=0)
    retry_backoff: float = Field(default=2.0, gt=1.0)
    delay_between_calls: float = Field(default=5.0, gt=0)

    enable_backups: bool = Field(default=True)
    backup_dir: str = Field(default="./backups")

    @field_validator("openrouter_api_key")
    @classmethod
    def _validate_openrouter_key(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) < 20:
            raise ValueError("OPENROUTER_API_KEY пустой или является заглушкой.")
        return v

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL '{v}' недопустим. Разрешено: {sorted(allowed)}")
        return v_upper

    @field_validator("agnes_api_key")
    @classmethod
    def _validate_agnes_key(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) < 10:
            raise ValueError("AGNES_API_KEY пустой или слишком короткий.")
        return v


def load_settings(project_path_override: Optional[Path] = None) -> Settings:
    settings = Settings()
    project: Optional[Path] = project_path_override
    if project is None:
        raw = settings.default_project_path
        if raw is None or not raw.strip():
            raise SystemExit("Путь к проекту не задан. Передайте --project или заполните DEFAULT_PROJECT_PATH в .env.")
        project = Path(raw).expanduser().resolve()
    else:
        project = project.expanduser().resolve()

    if not project.exists():
        raise SystemExit(f"Путь к проекту не существует: {project}")
    if not project.is_dir():
        raise SystemExit(f"Путь к проекту не является каталогом: {project}")

    object.__setattr__(settings, "default_project_path", str(project))
    return settings