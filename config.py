# config.py
"""Конфигурация оркестратора: мульти-роутерный fallback (OrcaRouter → OpenRouter)."""
from __future__ import annotations
from pathlib import Path
from typing import Optional

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


    # --- НАСТРОЙКИ РЕВЬЮЕРА ------------------------------------------------
    reviewer_max_output_length: int = Field(
        default=4000,
        gt=500,
        alias="REVIEWER_MAX_OUTPUT_LENGTH",
        description="Максимальная длина вывода stdout/stderr для анализа ревьюером"
    )

    # --- OPENROUTER (fallback-роутер, приоритет 1) ------------------------
    # Используется в Settings.routers_config → ModelResolver. НЕ удалять.
    openrouter_api_key: str = Field(..., min_length=1)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    # --- AGNES CLI -----------------------------------------------------
    agnes_api_key: str = Field(..., min_length=1, description="API ключ Agnes")
    agnes_cli_path: str = Field(default="agnes", description="Путь к исполняемому файлу Agnes CLI")
    agnes_model: str = Field(default="agnes/agnes-2.5-flash", description="Модель Agnes")
    agnes_timeout: int = Field(default=300, gt=0, description="Таймаут Agnes в секундах")

    # --- РОУТЕРЫ И МОДЕЛИ ---------------------------------------------
    priority_keywords_str: str = Field(
        default="qwen3, deepseek-v3, qwen-2.5-72b, nemotron",
        alias="PRIORITY_KEYWORDS",
    )
    models_cache_path: str = Field(
        default="./models_cache.json",
        alias="MODELS_CACHE_PATH",
    )

    # OrcaRouter (приоритет 0)
    orcarouter_api_key: str = Field(default="", alias="ORCAROUTER_API_KEY")
    orcarouter_base_url: str = Field(
        default="https://api.orcarouter.ai/v1", alias="ORCAROUTER_BASE_URL"
    )

    max_replan_attempts: int = Field(default=2, gt=0, description="Максимум попыток перепланирования одного шага")

    @property
    def priority_keywords(self) -> list[str]:
        """Список приоритетных ключевых слов для подбора моделей (разбит по запятой, strip)."""
        return [k.strip() for k in self.priority_keywords_str.split(",") if k.strip()]

    @property
    def routers_config(self) -> list[dict]:
        """Список роутеров в порядке приоритета (0 — высший).

        Включает только те роутеры, для которых задан API-ключ.
        Каждый элемент: {name, base_url, api_key, priority}.
        """
        routers: list[dict] = []
        if self.openrouter_api_key:  # ← сначала OpenRouter
            routers.append({
                "name": "openrouter",
                "base_url": self.openrouter_base_url,
                "api_key": self.openrouter_api_key,
                "priority": 0,  # ← высший приоритет
            })
        if self.orcarouter_api_key:  # ← потом OrcaRouter
            routers.append({
                "name": "orcarouter",
                "base_url": self.orcarouter_base_url,
                "api_key": self.orcarouter_api_key,
                "priority": 1,  # ← резервный
            })
        routers.sort(key=lambda r: r["priority"])
        return routers

    # --- ПРОЕКТ И ЛИМИТЫ -----------------------------------------------
    default_project_path: Optional[str] = Field(default=None)
    log_level: str = Field(default="INFO")
    max_steps: int = Field(default=10, gt=0)
    max_retries: int = Field(default=2, gt=0)
    retry_delay: float = Field(default=5.0, gt=0)
    retry_backoff: float = Field(default=2.0, gt=1.0)
    delay_between_calls: float = Field(default=5.0, gt=0)
    enable_backups: bool = Field(default=True)
    backup_dir: str = Field(default="./backups")

    # --- ВАЛИДАТОРЫ ----------------------------------------------------
    @field_validator("openrouter_api_key", "agnes_api_key")
    @classmethod
    def _validate_keys(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) < 10:
            raise ValueError("API ключ пустой или слишком короткий.")
        return v

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL '{v}' недопустим. Разрешено: {sorted(allowed)}")
        return v_upper


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

    settings.default_project_path = str(project)
    return settings