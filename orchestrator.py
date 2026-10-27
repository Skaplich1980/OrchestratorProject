# orchestrator.py
"""Главный CLI-оркестратор.
Точка входа: ``python orchestrator.py --project <path> --goal <file>``.

Логика:
1. Инициализация настроек (config.py) и логгеров (logger_setup.py).
2. Чтение цели из файла, указанного через --goal.
3. Вызов QwenPlanner для получения плана (динамические free-модели + fallback).
4. Цикл по шагам плана (ограничение MAX_STEPS):
   - AgnesExecutor выполняет шаг (с предварительным бэкапом target_files).
   - QwenReviewer проверяет результат.
   - Если success=False — прервать цикл с явной причиной.
5. Печать итогового отчёта через rich.
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
    # Гарантируем существование папки для бэкапов этого запуска
    run_paths.backups_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Инициализация логгеров (все пишут в одну папку run_paths) --
    # Используем "dynamic", так как конкретная модель выбирается внутри агента
    planner_logger = setup_agent_logger("planner", "planner", "dynamic", run_paths, settings.log_level)
    executor_logger = setup_agent_logger("executor", "executor", "agnes-cli", run_paths, settings.log_level)
    reviewer_logger = setup_agent_logger("reviewer", "reviewer", "dynamic", run_paths, settings.log_level)

    planner = QwenPlanner(settings, planner_logger)
    executor = AgnesExecutor(settings, executor_logger)
    reviewer = QwenReviewer(settings, reviewer_logger)

    emit_structured_event(
        planner_logger, "run_started", "orchestrator", "orchestrator", True,
        goal=goal_text[:200], project=str(project_path),
    )

    # --- 3. Планировщик ------------------------------------------------
    console.print("[bold cyan]→ Планировщик:[/bold cyan] формирование плана…")
    planner_result: AgentResult = await planner.execute(
        task=goal_text,
        context={"project_path": str(project_path)},
    )

    if not planner_result.success:
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

    # --- 4. Цикл по шагам ----------------------------------------------
    max_steps: int = settings.max_steps
    completed_steps: int = 0
    failure_reason: Optional[str] = None

    for step_index, step in enumerate(plan, start=1):
        if step_index > max_steps:
            console.print(f"[yellow]⚠ Достигнут лимит MAX_STEPS={max_steps}. Остальные шаги пропущены.[/yellow]")
            break

        step_desc: str = str(step.get("description", f"Step {step_index}"))
        target_files: list[str] = list(step.get("target_files", []) or [])

        console.print(
            f"[bold]Шаг {step_index}/{min(total_steps, max_steps)}:[/bold] {step_desc}\n"
            f"  Целевые файлы: {', '.join(target_files) if target_files else '—'}"
        )

        emit_structured_event(
            executor_logger, "step_started", "executor", "agnes-cli", True,
            step=step_index, description=step_desc, target_files=target_files,
        )

        # --- 4a. Исполнитель (с бэкапами внутри) -----------------------
        executor_result: AgentResult = await executor.execute(
            task=step_desc,
            context={
                "project_path": str(project_path),
                "step_number": step_index,
                "target_files": target_files,
                "backups_dir": str(run_paths.backups_dir),
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
            break

        # --- 4b. Ревьюер -----------------------------------------------
        reviewer_result: AgentResult = await reviewer.execute(
            task=step_desc,
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
            review_success, step=step_index, reason=review_reason,
            tokens_used=reviewer_result.tokens_used,
        )

        if not review_success:
            failure_reason = f"Шаг {step_index}: ревьюер не подтвердил выполнение. Причина: {review_reason}"
            console.print(f"[bold red]✗ {failure_reason}[/bold red]")
            break

        # --- 4c. Успех шага и задержка ---------------------------------
        completed_steps += 1
        console.print(f"[green]✓ Шаг {step_index} завершён.[/green] {review_reason[:100]}")

        if step_index < min(total_steps, max_steps):
            console.print(f"[dim]…пауза {settings.delay_between_calls}s (защита от rate limit)[/dim]")
            await asyncio.sleep(settings.delay_between_calls)

    # --- 5. Финальное событие и отчёт ----------------------------------
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
        description="Оркестратор автоматизации задач Python- projekтов.",
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
    except Exception as exc:  # Перехватываем ValidationError от pydantic
        print(f"❌ Ошибка валидации .env: {exc}", file=sys.stderr)
        print("💡 Проверьте файл .env: ключи не должны быть заглушками (например, 'ТВОЙ_КЛЮЧ').", file=sys.stderr)
        return 1

    # --- 2. Чтение цели ------------------------------------------------
    goal_path: Path = args.goal
    try:
        goal_text: str = goal_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        print(f"❌ Ошибка кодировки: Файл '{goal_path}' сохранён не в UTF-8.", file=sys.stderr)
        print("💡 Решение: Откройте файл в Блокноте -> Файл -> Сохранить как -> Кодировка: UTF-8", file=sys.stderr)
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
    except Exception as exc:  # noqa: BLE001
        print("\n❌ Необработанное исключение:", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())