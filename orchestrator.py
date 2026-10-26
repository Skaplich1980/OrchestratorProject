# logger_setup.py
"""Настройка логирования для оркестратора.

Структура логов для каждого запуска:
    logs/run_2026-10-26_17-49-01/
    ├── run.log                     # общий текстовый лог по всем ролям
    ├── planner.log                 # лог агента-планировщика
    ├── executor.log                # лог агента-исполнителя
    ├── reviewer.log                # лог агента-ревьюера
    ├── structured_trace.jsonl      # машиночитаемые события (JSON Lines)
    └── backups/                    # бэкапы файлов перед правками
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class RunLogPaths:
    """Пути к файлам одного запуска. Создаётся ОДИН раз, передаётся всем агентам."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir: Path = run_dir
        self.run_log: Path = run_dir / "run.log"
        self.jsonl_path: Path = run_dir / "structured_trace.jsonl"
        # Каталог бэкапов текущего запуска (создаётся лениво).
        self.backups_dir: Path = run_dir / "backups"
        self.agent_logs: dict[str, Path] = {}


def _build_role_model_formatter() -> logging.Formatter:
    """Кастомный формат: [ts.ms] [ROLE/MODEL] [LEVEL] message."""
    fmt = "[%(asctime)s.%(msecs)03d] [%(role)s/%(model)s] [%(levelname)s] %(message)s"
    return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")


class _RoleModelFilter(logging.Filter):
    """Добавляет атрибуты role и model в каждый LogRecord."""

    def __init__(self, role: str, model: str) -> None:
        super().__init__()
        self._role = role
        self._model = model

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "role"):
            record.role = self._role  # type: ignore[attr-defined]
        if not hasattr(record, "model"):
            record.model = self._model  # type: ignore[attr-defined]
        return True


def create_run_directory(base_log_dir: Path) -> RunLogPaths:
    """Создаёт папку запуска ОДИН раз. Вызывается один раз в orchestrator.main()."""
    run_dir = base_log_dir / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunLogPaths(run_dir)


def setup_agent_logger(
    name: str,
    role: str,
    model: str,
    run_paths: RunLogPaths,
    level: str = "DEBUG",
) -> logging.Logger:
    """Создаёт логгер агента, привязанный к УЖЕ СУЩЕСТВУЮЩЕЙ папке запуска.

    :param name: Имя логгера и файла (например, 'planner' → planner.log).
    :param role: Роль для метки (planner/executor/reviewer).
    :param model: Имя модели (qwen3-235b / agnes-cli / dynamic).
    :param run_paths: Единые пути запуска (создаются create_run_directory).
    """
    # Общий run.log — сюда пишут все агенты (для сквозного чтения хронологии).
    run_handler = logging.FileHandler(run_paths.run_log, encoding="utf-8")
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(_build_role_model_formatter())

    # Индивидуальный лог агента.
    agent_log = run_paths.run_dir / f"{name}.log"
    run_paths.agent_logs[name] = agent_log
    agent_handler = logging.FileHandler(agent_log, encoding="utf-8")
    agent_handler.setLevel(logging.DEBUG)
    agent_handler.setFormatter(_build_role_model_formatter())

    # Консоль.
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    console.setFormatter(_build_role_model_formatter())

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Если setup вызван повторно — чистим старые хендлеры, чтобы не дублировать строки.
    logger.handlers.clear()
    logger.addHandler(run_handler)
    logger.addHandler(agent_handler)
    logger.addHandler(console)
    logger.propagate = False

    # Вешаем фильтр role/model на КАЖДЫЙ хендлер.
    role_filter = _RoleModelFilter(role, model)
    for h in logger.handlers:
        h.addFilter(role_filter)

    # Прячем пути на логгере, чтобы emit_structured_event мог их найти.
    logger.run_paths = run_paths  # type: ignore[attr-defined]

    logger.debug("Логгер '%s' (role=%s, model=%s) → %s", name, role, model, run_paths.run_dir)
    return logger


def emit_structured_event(
    logger: logging.Logger,
    event_type: str,
    role: str,
    model: str,
    success: bool,
    tokens_used: Optional[dict[str, int]] = None,
    **extra: Any,
) -> None:
    """Дописывает одно JSON-событие в structured_trace.jsonl (JSON Lines)."""
    run_paths: Optional[RunLogPaths] = getattr(logger, "run_paths", None)
    if run_paths is None:
        logger.warning("structured_trace.jsonl недоступен — логгер без run_paths.")
        return

    record: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "event": event_type,
        "role": role,
        "model": model,
        "success": success,
        "tokens_used": tokens_used,
    }
    record.update(extra)

    try:
        with run_paths.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        logger.warning("Не удалось записать JSONL-событие: %s", e)