# agents/executor.py
"""Исполнитель на базе Agnes CLI (subprocess).

Выполняет один шаг плана через команду:
  agnes run "<task>" --model <model> --auto

Ключевые особенности:
1. Передача AGNES_API_KEY через переменную окружения subprocess.
2. Флаг --auto для автоматического подтверждения прав на изменение файлов.
3. Бэкапы целевых файлов перед выполнением (если включено в настройках).
4. Создание .agneshints для контекста проекта.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agents.base import AgentResult, BaseAgent
from config import Settings

# Локальный .agneshints — если его нет в проекте, создадим дефолтный.
_DEFAULT_AGNESHINTS = (
    "This is a Python project under automated orchestration.\n"
    "Follow the project's existing code style.\n"
    "Prefer minimal, atomic changes.\n"
)


class AgnesExecutor(BaseAgent):
    """Исполнитель задач через Agnes CLI."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        super().__init__()
        self.name = "executor"
        self.role = "executor"
        self.model = settings.agnes_model
        self._settings = settings
        self._logger = logger

    async def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        project_path_str: str = context.get("project_path", ".")
        project_path: Path = Path(project_path_str)
        step_number: int = int(context.get("step_number", 0))
        target_files: list[str] = list(context.get("target_files", []) or [])
        backups_dir: str | None = context.get("backups_dir")

        self._logger.info("Шаг #%d: запуск Agnes CLI в '%s'.", step_number, project_path)
        self._logger.debug("--- ЗАДАЧА (PROMPT) ДЛЯ AGNES ---\n%s", task)

        # --- 1. БЕКАПЫ перед правками -----------------------------------
        if self._settings.enable_backups and backups_dir is not None and target_files:
            self._backup_files(project_path, target_files, Path(backups_dir), step_number)

        # --- 2. .agneshints в целевом проекте ---------------------------
        self._ensure_agneshints(project_path)

        # --- 3. Формируем команду ---------------------------------------
        # ИСПРАВЛЕНИЕ: сообщение передаётся как позиционный аргумент [message..],
        # а не через --prompt. Добавляем --model и --auto.
        cmd: list[str] = [
            self._settings.agnes_cli_path,
            "run",
            task,  # <-- Позиционный аргумент (сообщение)
            "--model", self._settings.agnes_model,
            "--auto",  # КРИТИЧЕСКИ ВАЖНО: авто-подтверждение изменений
        ]

        # --- 4. Окружение: инжектируем API ключ -------------------------
        env = os.environ.copy()
        env["AGNES_API_KEY"] = self._settings.agnes_api_key

        # --- 5. Запуск subprocess ---------------------------------------
        try:
            self._logger.debug("Команда: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                cwd=str(project_path),
                capture_output=True,
                text=True,
                env=env,
                timeout=self._settings.agnes_timeout,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            self._logger.error("Agnes не найден по пути '%s'.", self._settings.agnes_cli_path)
            return AgentResult(
                success=False, data=None,
                error_message=f"Agnes binary not found: {self._settings.agnes_cli_path} ({exc})"
            )
        except subprocess.TimeoutExpired as exc:
            partial_out = self._coerce_text(exc.stdout)
            partial_err = self._coerce_text(exc.stderr)
            self._logger.error("Agnes не завершился за %d c. Partial stdout: %s", self._settings.agnes_timeout,
                               partial_out[:500])
            return AgentResult(
                success=False,
                data={"stdout": partial_out, "stderr": partial_err, "returncode": None},
                error_message=f"Agnes CLI timed out after {self._settings.agnes_timeout}s",
            )
        except OSError as exc:
            self._logger.error("OSError при запуске Agnes: %s", exc)
            return AgentResult(success=False, data=None, error_message=f"OS error: {exc}")

        # --- 6. Разбор результата ---------------------------------------
        data = {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

        if result.returncode == 0:
            self._logger.info("Agnes: code=0, stdout=%d симв.", len(result.stdout))
            return AgentResult(success=True, data=data)

        self._logger.error("Agnes: code=%d, stderr=%s", result.returncode, result.stderr[:500])
        return AgentResult(
            success=False, data=data,
            error_message=f"Agnes exited with code {result.returncode}: {result.stderr.strip()[:300]}"
        )

    # ----------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------
    def _backup_files(self, project_path: Path, target_files: list[str], backups_dir: Path, step_number: int) -> None:
        """Копирует целевые файлы в backups/step_N/ перед правками."""
        step_backup = backups_dir / f"step_{step_number:03d}"
        try:
            step_backup.mkdir(parents=True, exist_ok=True)
            copied = 0
            for rel in target_files:
                if not rel or rel in (".", "./"):
                    continue  # Пропускаем корень проекта

                src = project_path / rel
                if not src.exists() or src.is_dir():
                    continue  # Пропускаем несуществующие файлы и директории

                dst = step_backup / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1

            if copied > 0:
                self._logger.info("Шаг #%d: сделан бэкап %d файл(ов) в %s", step_number, copied, step_backup)
        except OSError as e:
            self._logger.warning("Шаг #%d: сбой бэкапа: %s", step_number, e)

    @staticmethod
    def _ensure_agneshints(project_path: Path) -> None:
        """Создаёт <проект>/.agneshints, если его нет."""
        hints = project_path / ".agneshints"
        if not hints.exists():
            try:
                hints.write_text(_DEFAULT_AGNESHINTS, encoding="utf-8")
            except OSError:
                pass

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Нормализует stdout/stderr из TimeoutExpired (bytes|str|None → str)."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)