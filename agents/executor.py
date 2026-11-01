# agents/executor.py
"""Исполнитель на базе Agnes CLI (subprocess).

Выполняет один шаг плана через команду:
  agnes run "<prompt>" --model <model> --auto

Ключевые особенности:
1. Промт собирается через PromptManager (без хардкода в .py).
2. Передача AGNES_API_KEY через переменную окружения subprocess.
3. Флаг --auto для автоматического подтверждения прав на изменение файлов.
4. Бэкапы целевых файлов перед выполнением (если включено в настройках).
5. Финальный промт записывается в <project>/.agneshints, чтобы Agnes
   гарантированно его "увидел" (локальный .agneshints имеет приоритет).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agents.base import AgentResult, BaseAgent
from config import Settings
from prompt_manager import PromptManager


class AgnesExecutor(BaseAgent):
    """Исполнитель задач через Agnes CLI."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        prompt_manager: PromptManager,
    ) -> None:
        super().__init__()
        self.name = "executor"
        self.role = "executor"
        self.model = settings.agnes_model
        self._settings = settings
        self._logger = logger
        self._prompt_manager = prompt_manager

    async def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        project_path_str: str = context.get("project_path", ".")
        project_path: Path = Path(project_path_str)
        step_number: int = int(context.get("step_number", 0))
        target_files: list[str] = list(context.get("target_files", []) or [])
        backups_dir: str | None = context.get("backups_dir")
        step_info: str = str(context.get("step_info", "") or "")

        self._logger.info("Шаг #%d: запуск Agnes CLI в '%s'.", step_number, project_path)

        # --- 1. Сборка финального промта через PromptManager -----------
        final_prompt: str = self._prompt_manager.build_executor_prompt(
            task=task,
            target_files=target_files,
            step_info=step_info,
        )
        # INFO-уровень: в executor.log всегда виден ровно тот текст,
        # который уйдёт в `agnes run`.
        self._logger.info(
            "--- ФИНАЛЬНЫЙ ПРОМТ ДЛЯ AGNES (шаг #%d) ---\n%s",
            step_number,
            final_prompt,
        )

        # --- 2. БЕКАПЫ перед правками -----------------------------------
        if self._settings.enable_backups and backups_dir is not None and target_files:
            self._backup_files(project_path, target_files, Path(backups_dir), step_number)

        # --- 3. .agneshints в целевом проекте ---------------------------
        # Пишем именно финальный промт + глобальные правила,
        # чтобы Agnes гарантированно их учёл.
        self._write_agneshints(project_path, final_prompt, step_number)

        # --- 4. Формируем команду ---------------------------------------
        # Защита от лимита длины CLI в Windows (~8191 символ):
        # если промт длинный, записываем его во временный файл в корне проекта
        # и передаём через stdin, чтобы не превышать ограничение ОС.
        PROMPT_LENGTH_LIMIT = 7000  # С запасом под Windows

        if len(final_prompt) > PROMPT_LENGTH_LIMIT:
            # Записываем промт во временный файл в проекте
            temp_prompt_file = project_path / ".agnes_task.md"
            temp_prompt_file.write_text(final_prompt, encoding="utf-8")
            self._logger.info(
                "Шаг #%d: промт слишком длинный (%d симв.), записан в %s",
                step_number, len(final_prompt), temp_prompt_file,
            )

            # Передаём промт через stdin (Agnes должен поддерживать чтение из stdin)
            cmd: list[str] = [
                self._settings.agnes_cli_path,
                "run",
                "--model", self._settings.agnes_model,
                "--auto",
            ]
            stdin_data: str | None = final_prompt
        else:
            # Короткий промт — передаём как позиционный аргумент
            cmd = [
                self._settings.agnes_cli_path,
                "run",
                final_prompt,
                "--model", self._settings.agnes_model,
                "--auto",
            ]
            stdin_data = None

        # --- 5. Окружение: инжектируем API ключ -------------------------
        env = os.environ.copy()
        env["AGNES_API_KEY"] = self._settings.agnes_api_key

        # --- 6. Запуск subprocess (асинхронно, чтобы не блокировать event loop) ----
        try:
            self._logger.debug("Команда: %s", " ".join(cmd[:3]) + " ...")

            def _run_subprocess() -> subprocess.CompletedProcess:
                return subprocess.run(
                    cmd,
                    cwd=str(project_path),
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=self._settings.agnes_timeout,
                    encoding="utf-8",
                    errors="replace",
                    input=stdin_data,  # Передаём промт через stdin, если он длинный
                )

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _run_subprocess)
        except FileNotFoundError as exc:
            self._logger.error("Agnes не найден по пути '%s'.", self._settings.agnes_cli_path)
            return AgentResult(
                success=False, data=None,
                error_message=f"Agnes binary not found: {self._settings.agnes_cli_path} ({exc})"
            )
        except subprocess.TimeoutExpired as exc:
            partial_out = self._coerce_text(exc.stdout)
            partial_err = self._coerce_text(exc.stderr)
            self._logger.error(
                "Agnes не завершился за %d c. Partial stdout: %s",
                self._settings.agnes_timeout, partial_out[:500],
            )
            return AgentResult(
                success=False,
                data={"stdout": partial_out, "stderr": partial_err, "returncode": None},
                error_message=f"Agnes CLI timed out after {self._settings.agnes_timeout}s",
            )
        except OSError as exc:
            self._logger.error("OSError при запуске Agnes: %s", exc)
            return AgentResult(success=False, data=None, error_message=f"OS error: {exc}")

        # --- 7. Разбор результата ---------------------------------------
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
                if not rel or rel in (".", "./"):
                    continue  # Пропускаем корень проекта

                src = project_path / rel

                # ЗАЩИТА ОТ PATH TRAVERSAL: проверяем, что файл внутри проекта
                try:
                    src_resolved = src.resolve()
                    project_resolved = project_path.resolve()
                    if not src_resolved.is_relative_to(project_resolved):
                        self._logger.warning(
                            "Шаг #%d: пропущен файл '%s' — путь выходит за пределы проекта (защита от path traversal)",
                            step_number, rel,
                        )
                        continue
                except (OSError, ValueError) as e:
                    self._logger.warning(
                        "Шаг #%d: не удалось проверить путь '%s': %s. Файл пропущен.",
                        step_number, rel, e,
                    )
                    continue

                if not src.exists() or src.is_dir():
                    continue  # Пропускаем несуществующие файлы и директории

                dst = step_backup / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1

            if copied > 0:
                self._logger.info(
                    "Шаг #%d: сделан бэкап %d файл(ов) в %s",
                    step_number, copied, step_backup,
                )
        except OSError as e:
            self._logger.warning("Шаг #%d: сбой бэкапа: %s", step_number, e)

    def _write_agneshints(
        self,
        project_path: Path,
        final_prompt: str,
        step_number: int,
    ) -> None:
        """Автоматически создаёт/обновляет .agneshints в целевом проекте.
        Объединяет глобальные правила оркестратора с промтом текущего шага.
        """
        hints_path = project_path / ".agneshints"

        # 1. Получаем глобальные правила через PromptManager (единая точка истины)
        global_rules = self._prompt_manager.get_global_rules()
        if global_rules:
            global_rules = global_rules.strip() + "\n\n"
        else:
            self._logger.warning("Глобальные правила не найдены в PromptManager")

        # 2. Формируем итоговый контент: Глобальные правила + Контекст шага
        content_to_write = (
            f"{global_rules}"
            f"--- КОНТЕКСТ И ЗАДАЧА ТЕКУЩЕГО ШАГА #{step_number} ---\n"
            f"{final_prompt}"
        )

        # 3. Записываем в целевой проект
        try:
            hints_path.write_text(content_to_write, encoding="utf-8")
            self._logger.debug(
                "Шаг #%d: .agneshints обновлён в проекте (%d симв.) по пути %s",
                step_number, len(content_to_write), hints_path,
            )
        except OSError as e:
            self._logger.warning(
                "Шаг #%d: не удалось обновить .agneshints (%s): %s",
                step_number, hints_path, e,
            )

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Нормализует stdout/stderr из TimeoutExpired (bytes|str|None → str)."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)