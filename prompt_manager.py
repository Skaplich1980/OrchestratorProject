# prompt_manager.py
"""Менеджер промтов: собирает финальный промт из шаблонов и правил.

Идея: промты живут в .txt/.md файлах, а не в Python-коде.
PromptManager — единственная точка, которая знает, как их склеить.
"""
from __future__ import annotations

from pathlib import Path


class PromptManager:
    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = prompts_dir
        self._global_rules = self._load_text("global_rules.md")

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------
    def _load_text(self, filename: str) -> str:
        """Загружает текст файла. Если файла нет, возвращает пустую строку."""
        filepath = self.prompts_dir / filename
        if filepath.exists():
            return filepath.read_text(encoding="utf-8").strip()
        return ""

    def _format_safe(self, template: str, **kwargs: object) -> str:
        """Безопасная подстановка: недостающие ключи не роняют .format().

        Защита от ситуации, когда в шаблоне случайно остался плейсхолдер,
        которого нет в kwargs. Не падаем в проде из-за опечатки в .txt.
        """

        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        return template.format_map(_SafeDict(**kwargs))

    # ------------------------------------------------------------------
    # PLANNER
    # ------------------------------------------------------------------
    def build_planner_system_prompt(self, max_steps: int) -> str:
        """Собирает system-промт для планировщика."""
        template = self._load_text("planner_system.txt")
        if not template:
            template = (
                "You are a precise project planning agent for Python codebases.\n"
                "Given the goal and project context, produce a strictly valid JSON array of steps.\n"
                "Each step is an object:\n"
                '{{\n'
                '  "step": <int>,\n'
                '  "description": "<short imperative sentence>",\n'
                '  "target_files": ["<relative/path>", ...]\n'
                '}}\n'
                "Rules:\n"
                "- Return ONLY a JSON array. No markdown, no code fences, no commentary.\n"
                "- Each step is atomic and independently verifiable.\n"
                "- target_files are relative paths from the project root.\n"
                "- Up to {max_steps} steps.\n"
                "\n"
                "## GLOBAL RULES\n"
                "{global_rules}\n"
            )
        return self._format_safe(
            template,
            max_steps=max_steps,
            global_rules=self._global_rules,
        )

    def build_planner_user_prompt(
        self,
        project_path: str | None,
        task: str,
        max_steps: int,
    ) -> str:
        """Собирает user-промт для планировщика."""
        return (
            f"Project path: {project_path}\n"
            f"Goal:\n{task}\n"
            f"Return the plan as a JSON array (up to {max_steps} steps)."
        )

    # ------------------------------------------------------------------
    # EXECUTOR (Agnes CLI)
    # ------------------------------------------------------------------
    def build_executor_prompt(
        self,
        task: str,
        target_files: list[str],
        step_info: str = "",
    ) -> str:
        """Собирает финальный промт-задание для Agnes.

        Важно: Agnes получает задание как позиционный аргумент CLI.
        Этот метод возвращает ровно ту строку, которая уйдёт в `agnes run`.
        Дополнительно она пишется в `<project>/.agneshints`, чтобы
        гарантированно была учтена (локальный .agneshints имеет приоритет).
        """
        template = self._load_text("executor_system.txt")
        if not template:
            template = (
                "{task}\n"
                "\n"
                "Target files: {target_files}\n"
                "{step_info}\n"
                "## GLOBAL RULES\n"
                "{global_rules}\n"
            )
        return self._format_safe(
            template,
            task=task,
            target_files=", ".join(target_files) if target_files else "None",
            step_info=step_info,
            global_rules=self._global_rules,
        )

    # ------------------------------------------------------------------
    # REVIEWER
    # ------------------------------------------------------------------
    def build_reviewer_system_prompt(self) -> str:
        """Загружает системный промт для ревьюера из файла reviewer_system.txt.

        Если файл отсутствует, используется встроенный fallback-шаблон,
        чтобы ревьюер продолжал работать даже без внешних шаблонов.
        """
        template = self._load_text("reviewer_system.txt")
        if not template:
            template = (
                "You are a strict code review agent.\n"
                "Given a task and the execution result (stdout/stderr/returncode), determine\n"
                "whether the task was completed successfully.\n"
                'Return ONLY a JSON object:\n'
                '{"success": true/false, "reason": "<short explanation>"}\n'
                "Rules:\n"
                "- Be strict: errors, missing changes, incomplete work → success=false.\n"
                "- reason must be a single sentence.\n"
                "- No markdown, no commentary, no extra keys.\n"
            )
        return template

    def get_global_rules(self) -> str:
        """Возвращает глобальные правила для Agnes из файла global_agneshints.txt.

        Читает файл из корня оркестратора (parent от prompts_dir).
        Если файл отсутствует, возвращает пустую строку.
        """
        global_hints_path = self.prompts_dir.parent / "global_agneshints.txt"
        if global_hints_path.exists():
            return global_hints_path.read_text(encoding="utf-8")
        return ""