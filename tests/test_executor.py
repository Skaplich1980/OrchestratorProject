# tests/ test_executor.py
import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock

from config import load_settings
from logger_setup import create_run_directory, setup_agent_logger
from prompt_manager import PromptManager
from agents.executor import AgnesExecutor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


async def main():
    settings = load_settings(project_path_override=Path("D:\\Project\\garden_helper"))
    run_paths = create_run_directory(Path("logs"))
    logger = setup_agent_logger("executor", "executor", settings.agnes_model, run_paths, "DEBUG")

    prompt_manager = MagicMock(spec=PromptManager)
    prompt_manager.build_executor_prompt.return_value = "test prompt"
    executor = AgnesExecutor(settings, logger, prompt_manager)

    # Простая тестовая задача
    task = "Создай файл test_agnes.txt с текстом 'Agnes работает через оркестратор!'"

    print(f"\n🚀 Отправляем задачу в Agnes: {task}")
    result = await executor.execute(
        task=task,
        context={
            "project_path": str(settings.default_project_path),
            "step_number": 1,
            "target_files": ["test_agnes.txt"],
            "backups_dir": str(run_paths.backups_dir)
        }
    )

    if result.success:
        print("\n✅ УСПЕХ! Agnes выполнил задачу.")
        print(f"Return code: {result.data['returncode']}")
    else:
        print("\n❌ ОШИБКА!")
        print(f"Error: {result.error_message}")
        print(f"Stderr: {result.data.get('stderr', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())