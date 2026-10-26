# test_step2_planner_api.py
"""
ШАГ 2: Проверка реального API-запроса планировщика с динамическим выбором моделей.

Запуск из корня проекта:
    python test_step2_planner_api.py
"""
import sys
import asyncio
import logging
from pathlib import Path

# Добавляем корень проекта в sys.path для корректных импортов
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import load_settings
from logger_setup import create_run_directory, setup_agent_logger
from agents.planner import QwenPlanner

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


async def main() -> None:
    print("\n" + "=" * 70)
    print("ШАГ 2: Тест реального API-запроса планировщика")
    print("=" * 70)

    # 1. Загружаем настройки (используем текущую папку как тестовый проект)
    try:
        settings = load_settings(project_path_override=project_root)
    except SystemExit as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return

    # 2. Инициализируем логгер
    run_paths = create_run_directory(Path("logs"))
    logger = setup_agent_logger("test_planner", "planner", "dynamic", run_paths, settings.log_level)

    # 3. Создаём планировщик
    planner = QwenPlanner(settings, logger)

    # 4. Тестовая задача
    test_goal = "Создай файл hello.py с функцией greet(), которая печатает 'Hello, Garden Helper!'. Добавь типизацию."

    print(f"\n[TEST] Отправка задачи планировщику:\n'{test_goal}'\n")

    # 5. Выполняем запрос
    result = await planner.execute(
        task=test_goal,
        context={"project_path": str(project_root)}
    )

    # 6. Анализ результата
    print("\n" + "-" * 70)
    if result.success:
        print("✅ УСПЕХ! План получен.")
        print(f"Использованная модель: {planner.model}")
        print(f"Потрачено токенов: {result.tokens_used}")
        print("\nСгенерированный план:")
        for step in result.data:
            print(f"  Шаг {step['step']}: {step['description']}")
            print(f"    Файлы: {', '.join(step.get('target_files', []))}")
    else:
        print("❌ ПРОВАЛ! Планировщик не смог сформировать план.")
        print(f"Ошибка: {result.error_message}")

    print("\n" + "=" * 70)
    print(f"Полные логи сохранены в: {run_paths.run_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())