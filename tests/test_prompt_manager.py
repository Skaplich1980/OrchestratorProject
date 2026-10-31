# tests/test_prompt_manager.py
import pytest
from pathlib import Path
from prompt_manager import PromptManager


@pytest.fixture
def temp_prompts_dir(tmp_path: Path) -> Path:
    """Создаёт временную директорию с тестовыми шаблонами."""
    (tmp_path / "global_rules.md").write_text("Rule 1: Be strict.\nRule 2: No dead code.", encoding="utf-8")
    (tmp_path / "executor_system.txt").write_text(
        "Task: {task}\nTarget: {target_files}\nRules:\n{global_rules}", encoding="utf-8"
    )
    return tmp_path


def test_prompt_manager_builds_executor_prompt(temp_prompts_dir: Path):
    manager = PromptManager(prompts_dir=temp_prompts_dir)

    result = manager.build_executor_prompt(
        task="Fix bug in main.py",
        target_files=["main.py", "test_main.py"],
        step_info="Step 1 of 3"
    )

    assert "Fix bug in main.py" in result
    assert "main.py, test_main.py" in result
    assert "Rule 1: Be strict." in result
    assert "Rule 2: No dead code." in result


def test_prompt_manager_handles_missing_files_safely(tmp_path: Path):
    """Если файлов нет, менеджер должен использовать fallback-шаблоны без ошибок."""
    manager = PromptManager(prompts_dir=tmp_path)

    # Не должно быть исключения KeyError из-за отсутствующих файлов
    result = manager.build_planner_system_prompt(max_steps=5)

    assert "{max_steps}" not in result  # Плейсхолдер должен быть заменён
    assert "5" in result
    assert "GLOBAL RULES" in result