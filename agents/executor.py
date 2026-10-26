# agents/executor.py
"""Исполнитель через Agnes CLI (subprocess).

Особенности:
  1. Бэкапы файлов из target_files ПЕРЕД запуском (если ENABLE_BACKUPS=true).
  2. Генерация .agneshints в целевом проекте (если отсутствует) —
     это локальный контекст Agnes, он приоритетнее глобального.
  3. AGNES_MOIM_MESSAGE_TEXT — персистентный guardrail, инжектится каждый ход
     (Agnes его не «забудет» при заполнении контекста).
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

# Guardrail — инжектится КАЖДЫЙ ход через AGNES_MOIM_MESSAGE_TEXT.
# Agnes не может его «забыть» при заполнении контекста.
_GUARDRAIL_TEXT = (
    "AUTOMATED ORCHESTRATION MODE.\n"
    "Rules:\n"
    "1. Execute ONLY the task given. Do not refactor unrelated code.\n"
    "2. Keep changes minimal and atomic.\n"
    "3. Do not create new files unless the task explicitly requires it.\n"
    "4. Do not run destructive commands (rm -rf, drop table, etc.).\n"
    "5. If unsure — make the smallest safe change.\n"
)

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
        self.model = "agnes-cli"
        self._settings = settings
        self._logger = logger

    async def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        project_path_str: str = context.get("project_path", ".")
        project_path: Path = Path(project_path_str)
        step_number: int = int(context.get("step_number", 0))
        target_files: list[str] = list(context.get("target_files", []) or [])
        # Каталог бэкапов текущего запуска (может не быть — тогда пропускаем).
        backups_dir: str | None = context.get("backups_dir")

        self._logger.info(
            "Шаг #%d: запуск Agnes CLI в '%s'.",
            step_number,
            project_path,
        )
        # Глубокое логирование промта для Agnes.
        self._logger.debug("--- ЗАДАЧА (PROMPT) ДЛЯ AGNES ---\n%s", task)

        # --- 1. БЕКАПЫ перед правками -----------------------------------
        if self._settings.enable_backups and backups_dir is not None and target_files:
            self._backup_files(project_path, target_files, Path(backups_dir), step_number)

        # --- 2. .agneshints в целевом проекте ---------------------------
        # Agnes Code загружает <корень_проекта>/.agneshints при старте сессии.
        # Создаём, если отсутствует — чтобы Agnes понимал контекст.
        self._ensure_agneshints(project_path)

        # --- 3. Формируем команду ---------------------------------------
        # args-список (без shell=True) — защита от инъекций.
        cmd: list[str] = [
            self._settings.agnes_cli_path,
            "run",
            "-t",
            task,
            "--no-session",  # изолированный контекст, без истории между шагами
        ]

        # --- 4. Окружение: AGNES_MODE=auto + guardrail ------------------
        env = os.environ.copy()
        env["AGNES_MODE"] = "auto"  # полностью автоматический режим
        env["AGNES_MOIM_MESSAGE_TEXT"] = _GUARDRAIL_TEXT  # персистентные правила

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
            self._logger.error(
                "Agnes не найден по пути '%s'. Проверь AGNES_CLI_PATH в .env.",
                self._settings.agnes_cli_path,
            )
            return AgentResult(
                success=False,
                data=None,
                error_message=(
                    f"Agnes binary not found: {self._settings.agnes_cli_path} ({exc})"
                ),
            )
        except subprocess.TimeoutExpired as exc:
            partial_out = self._coerce_text(exc.stdout)
            partial_err = self._coerce_text(exc.stderr)
            self._logger.error(
                "Agnes не завершился за %d c. Partial stdout: %s",
                self._settings.agnes_timeout,
                partial_out[:500],
            )
            return AgentResult(
                success=False,
                data={"stdout": partial_out, "stderr": partial_err, "returncode": None},
                error_message=f"Agnes CLI timed out after {self._settings.agnes_timeout}s",
            )
        except OSError as exc:
            self._logger.error("OSError при запуске Agnes: %s", exc)
            return AgentResult(success=False, data=None, error_message=f"OS error: {exc}")

        # --- 6. Разбор результата ---------------------------------------
        data = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
        if result.returncode == 0:
            self._logger.info("Agnes: code=0, stdout=%d симв.", len(result.stdout))
            return AgentResult(success=True, data=data)

        self._logger.error(
            "Agnes: code=%d, stderr=%s",
            result.returncode,
            result.stderr[:500],
        )
        return AgentResult(
            success=False,
            data=data,
            error_message=(
                f"Agnes exited with code {result.returncode}: "
                f"{result.stderr.strip()[:300]}"
            ),
        )

    # ----------------------------------------------------------------
    # Вспомогательные
    # ----------------------------------------------------------------
    def _backup_files(
        self,
        project_path: Path,
        target_files: list[str],
        backups_dir: Path,
        step_number: int,
    ) -> None:
        """Копирует целевые файлы в backups/step_N/ перед правками."""
        step_backup = backups_dir / f"step_{step_number:03d}"
        try:
            step_backup.mkdir(parents=True, exist_ok=True)
            copied = 0
            for rel in target_files:
                src = project_path / rel
                if not src.exists():
                    # Файл ещё не создан — шаг, вероятно, его создаст.
                    self._logger.debug("Бэкап: '%s' не существует — пропускаем.", rel)
                    continue
                dst = step_backup / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)  # copy2 сохраняет метаданные
                copied += 1
            self._logger.info(
                "Шаг #%d: сделан бэкап %d файл(ов) в %s",
                step_number,
                copied,
                step_backup,
            )
        except OSError as e:
            # Бэкап — не критичен, но логируем явно.
            self._logger.warning("Шаг #%d: сбой бэкапа: %s", step_number, e)

    @staticmethod
    def _ensure_agneshints(project_path: Path) -> None:
        """Создаёт <проект>/.agneshints, если его нет."""
        hints = project_path / ".agneshints"
        if hints.exists():
            return
        try:
            hints.write_text(_DEFAULT_AGNESHINTS, encoding="utf-8")
        except OSError:
            # Если не смогли создать — не страшно, Agnes просто не получит контекст.
            pass

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Нормализует stdout/stderr из TimeoutExpired (bytes|str|None → str)."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)