# orchestrator.py
"""Главный CLI-оркестратор.

Точка входа: ``python orchestrator.py --project <path> --goal <file>``.

Логика:
1. Инициализация настроек (config.py) и логгеров (logger_setup.py).
2. Создание PromptManager (единая точка сборки промтов).
3. Чтение цели из файла, указанного через --goal.
4. Вызов QwenPlanner для получения плана.
5. Цикл по шагам плана (ограничение MAX_STEPS):
   - AgnesExecutor выполняет шаг (промт из PromptManager, бэкап, .agneshints).
   - QwenReviewer проверяет результат.
   - Если success=False — прервать цикл с явной причиной.
6. Печать итогового отчёта через rich.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.base import AgentResult
from agents.executor import AgnesExecutor
from agents.planner import QwenPlanner
from agents.reviewer import QwenReviewer
from config import Settings, load_settings
from logger_setup import (
    RunLogPaths,
    create_run_directory,
    emit_structured_event,
    setup_agent_logger,
)
from prompt_manager import PromptManager

# Папка с шаблонами промтов — рядом с orchestrator.py.
PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"


def _print_summary(
    console: Console,
    total_steps: int,
    completed_steps: int,
    failure_reason: Optional[str],
    run_paths: Optional[RunLogPaths],
) -> None:
    """Печатает итоговый отчёт в консоль через rich."""
    table = Table(title="Оркестратор: итог")
    table.add_column("Метрика", style="bold")
    table.add_column("Значение")
    table.add_row("Всего шагов в плане", str(total_steps))
    table.add_row("Выполнено успешно", str(completed_steps))
    table.add_row("Запущено с ошибкой", "Да" if failure_reason else "Нет")
    if failure_reason:
        table.add_row("Причина остановки", failure_reason[:200])
    console.print(table)

    if run_paths is not None:
        log_lines = [
            f"run.log: {run_paths.run_log}",
            f"planner.log: {run_paths.agent_logs.get('planner', '—')}",
            f"executor.log: {run_paths.agent_logs.get('executor', '—')}",
            f"reviewer.log: {run_paths.agent_logs.get('reviewer', '—')}",
            f"JSONL: {run_paths.jsonl_path}",
        ]
        console.print(Panel("\n".join(log_lines), title="Логи запуска", border_style="cyan"))


async def run_orchestrator(settings: Settings, goal_text: str, project_path: Path) -> int:
    """Выполняет полный цикл оркестрации.
    :returns: Код возврата процесса (0 = успех, 1 = ошибка).
    """
    console = Console()

    # --- 1. Единый контекст логов на весь запуск -----------------------
    run_paths: RunLogPaths = create_run_directory(Path("logs"))
    run_paths.backups_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Инициализация логгеров -------------------------------------
    planner_logger = setup_agent_logger("planner", "planner", "dynamic", run_paths, settings.log_level)
    executor_logger = setup_agent_logger("executor", "executor", "agnes-cli", run_paths, settings.log_level)
    reviewer_logger = setup_agent_logger("reviewer", "reviewer", "dynamic", run_paths, settings.log_level)

    # --- 3. PromptManager: единая точка сборки промтов -----------------
    if not PROMPTS_DIR.exists():
        planner_logger.warning(
            "Папка промтов не найдена: %s. Будут использованы fallback-шаблоны из PromptManager.",
            PROMPTS_DIR,
        )
    prompt_manager = PromptManager(prompts_dir=PROMPTS_DIR)

    # --- 4. Создание агентов с инъекцией PromptManager -----------------
    planner = QwenPlanner(settings, planner_logger, prompt_manager)
    executor = AgnesExecutor(settings, executor_logger, prompt_manager)
    # Reviewer пока без PromptManager — перепишем отдельным шагом.
    reviewer = QwenReviewer(settings, reviewer_logger)

    emit_structured_event(
        planner_logger, "run_started", "orchestrator", "orchestrator", True,
        goal=goal_text[:200], project=str(project_path),
    )

    # --- 5. Планировщик ------------------------------------------------
    console.print("[bold cyan]→ Планировщик:[/bold cyan] формирование плана…")
    planner_result: AgentResult = await planner.execute(
        task=goal_text,
        context={"project_path": str(project_path)},
    )

    if not planner_result.success or not planner_result.data:
        reason = planner_result.error_message or "Unknown planner error"
        console.print(f"[bold red]✗ Ошибка планировщика:[/bold red] {reason}")
        planner_logger.error("Планировщик не смог составить план: %s", reason)
        emit_structured_event(
            planner_logger, "plan_failed", "planner", "dynamic",
            False, error=reason, tokens_used=planner_result.tokens_used,
        )
        _print_summary(console, 0, 0, reason, run_paths)
        return 1

    plan: list[dict[str, Any]] = planner_result.data
    total_steps: int = len(plan)
    console.print(f"[green]✓ План из {total_steps} шагов.[/green] (лимит MAX_STEPS={settings.max_steps})")

    # --- 6. Цикл по шагам с поддержкой Replan ------------------------------
    max_steps: int = settings.max_steps
    completed_steps: int = 0
    failure_reason: Optional[str] = None

    for step_index, step in enumerate(plan, start=1):
        if step_index > max_steps:
            console.print(
                f"[yellow]⚠ Достигнут лимит MAX_STEPS={max_steps}. Остальные шаги пропущены.[/yellow]"
            )
            break

        # Изначальные значения шага
        current_step_desc: str = str(step.get("description", f"Step {step_index}"))
        current_target_files: list[str] = list(step.get("target_files", []) or [])
        replan_attempts: int = 0

        # Цикл выполнения и перепланирования для текущего шага
        while replan_attempts <= settings.max_replan_attempts:
            step_info: str = (
                f"Step {step_index} of {min(len(plan), max_steps)}. "
                f"Original plan description: {current_step_desc}"
            )

            console.print(
                f"[bold]Шаг {step_index} (попытка {replan_attempts + 1}):[/bold] {current_step_desc}\n"
                f"  Целевые файлы: {', '.join(current_target_files) if current_target_files else '—'}"
            )

            emit_structured_event(
                executor_logger, "step_started", "executor", "agnes-cli", True,
                step=step_index, attempt=replan_attempts + 1, description=current_step_desc,
                target_files=current_target_files,
            )

            # --- 6a. Исполнитель -------------------------------------------
            executor_result: AgentResult = await executor.execute(
                task=current_step_desc,
                context={
                    "project_path": str(project_path),
                    "step_number": step_index,
                    "target_files": current_target_files,
                    "backups_dir": str(run_paths.backups_dir),
                    "step_info": step_info,
                },
            )

            if not executor_result.success:
                reason = executor_result.error_message or "Executor failed"
                failure_reason = f"Шаг {step_index}: исполнитель не завершил задачу: {reason}"
                console.print(f"[bold red]✗ {failure_reason}[/bold red]")
                emit_structured_event(
                    executor_logger, "step_failed", "executor", "agnes-cli", False,
                    step=step_index, error=reason,
                )
                break  # Выход из while, переход к завершению оркестратора

            # --- 6b. Ревьюер -----------------------------------------------
            reviewer_result: AgentResult = await reviewer.execute(
                task=current_step_desc,
                context={
                    "project_path": str(project_path),
                    "step_number": step_index,
                    "executor_result": executor_result,
                },
            )

            review_data: dict[str, Any] = reviewer_result.data or {}
            review_success: bool = bool(review_data.get("success", False))
            review_reason: str = str(review_data.get("reason", ""))

            emit_structured_event(
                reviewer_logger, "review_result", "reviewer", "dynamic",
                review_success, step=step_index, attempt=replan_attempts + 1, reason=review_reason,
                tokens_used=reviewer_result.tokens_used,
            )

            if review_success:
                # Успех! Выходим из цикла while и переходим к следующему шагу плана
                completed_steps += 1
                console.print(f"[green]✓ Шаг {step_index} завершён.[/green] {review_reason[:100]}")
                break

            # --- 6c. Неудача: пытаемся перепланировать ---------------------
            replan_attempts += 1
            if replan_attempts > settings.max_replan_attempts:
                failure_reason = (
                    f"Шаг {step_index}: ревьюер не подтвердил выполнение после "
                    f"{settings.max_replan_attempts} попыток исправления. Причина: {review_reason}"
                )
                console.print(f"[bold red]✗ {failure_reason}[/bold red]")
                break  # Выход из while, исчерпаны попытки

            console.print(
                f"[yellow]⚠ Шаг {step_index} не прошёл ревью. "
                f"Попытка перепланирования {replan_attempts}/{settings.max_replan_attempts}...[/yellow]"
            )

            # Формируем задачу для экстренного перепланирования
            replan_task = (
                f"CRITICAL CORRECTION REQUIRED.\n"
                f"Original task: {current_step_desc}\n"
                f"Reviewer feedback: {review_reason}\n"
                f"Generate a new, corrected plan (1-2 steps MAX) to fix this specific issue. "
                f"Do not change unrelated files. Adhere strictly to CLEAN CODE principles."
            )

            replan_result: AgentResult = await planner.execute(
                task=replan_task,
                context={"project_path": str(project_path)}
            )

            if replan_result.success and replan_result.data:
                # Берём первый шаг из нового плана как исправленную задачу
                new_step = replan_result.data[0]
                current_step_desc = new_step.get("description", current_step_desc)
                current_target_files = list(new_step.get("target_files", []) or [])
                console.print(f"[cyan]↻ Получен исправленный план: {current_step_desc}[/cyan]")
            else:
                failure_reason = f"Шаг {step_index}: не удалось сгенерировать исправленный план. {replan_result.error_message}"
                console.print(f"[bold red]✗ {failure_reason}[/bold red]")
                break  # Выход из while, планировщик не смог помочь

        # Если failure_reason установлен внутри while, прерываем весь цикл шагов
        if failure_reason:
            break

        # --- 6d. Задержка между успешными шагами ---------------------------
        if step_index < min(len(plan), max_steps) and not failure_reason:
            console.print(f"[dim]…пауза {settings.delay_between_calls}s (защита от rate limit)[/dim]")
            await asyncio.sleep(settings.delay_between_calls)

    # --- 7. Финальное событие и отчёт ----------------------------------
    emit_structured_event(
        planner_logger, "run_completed", "orchestrator", "orchestrator",
        failure_reason is None,
        completed=completed_steps, total=total_steps, failure_reason=failure_reason,
    )

    _print_summary(console, total_steps, completed_steps, failure_reason, run_paths)
    return 0 if failure_reason is None else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Оркестратор автоматизации задач Python-проектов.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Путь к целевому проекту (дефолт: DEFAULT_PROJECT_PATH из .env)",
    )
    parser.add_argument(
        "--goal",
        type=Path,
        required=True,
        help="Путь к файлу с целевым описанием задачи (goal.txt)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа для ``python orchestrator.py``."""
    args = parse_args(argv)

    # --- 1. Загрузка конфигурации с ПОЛНЫМ перехватом ошибок -----------
    project_override: Optional[Path] = args.project
    try:
        settings = load_settings(project_path_override=project_override)
    except SystemExit as exc:
        print(f"❌ Ошибка конфигурации (SystemExit): {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"❌ Ошибка валидации .env: {exc}", file=sys.stderr)
        print(
            "💡 Проверьте файл .env: ключи не должны быть заглушками (например, 'ТВОЙ_КЛЮЧ').",
            file=sys.stderr,
        )
        return 1

    # --- 2. Чтение цели ------------------------------------------------
    goal_path: Path = args.goal
    try:
        goal_text: str = goal_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        print(
            f"❌ Ошибка кодировки: Файл '{goal_path}' сохранён не в UTF-8.",
            file=sys.stderr,
        )
        print(
            "💡 Решение: Откройте файл в Блокноте -> Файл -> Сохранить как -> Кодировка: UTF-8",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError:
        print(f"❌ Файл цели не найден: {goal_path}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"❌ Ошибка чтения файла цели: {exc}", file=sys.stderr)
        return 1

    if not goal_text:
        print("❌ Файл цели пуст — нечего планировать.", file=sys.stderr)
        return 1

    # --- 3. Запускаем асинхронный цикл ---------------------------------
    try:
        return asyncio.run(
            run_orchestrator(settings, goal_text, Path(settings.default_project_path))
        )
    except KeyboardInterrupt:
        print("\n⏹ Прервано пользователем (Ctrl+C).", file=sys.stderr)
        return 130
    except Exception:  # noqa: BLE001
        print("\n❌ Необработанное исключение:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())