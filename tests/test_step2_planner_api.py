# test_step2_planner_api.py
"""
ШАГ 2: Проверка реального API-запроса планировщика (OpenRouter + DeepSeek).
"""
import sys
import asyncio
import logging
from pathlib import Path

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


async def test_provider(name: str, settings, logger, base_url: str, api_key: str, model: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"ТЕСТ ПЛАНИРОВЩИКА: {name} (модель: {model})")
    print(f"{'=' * 70}")

    # Создаём планировщик с переопределёнными настройками для этого провайдера
    planner = QwenPlanner(
        settings=settings,
        logger=logger,
        override_api_key=api_key,
        override_base_url=base_url,
        override_model=model
    )

    test_goal = "Создай файл hello.py с функцией greet(), которая печатает 'Hello, Garden Helper!'. Добавь типизацию."
    print(f"\n[ЗАДАЧА] {test_goal}\n")

    result = await planner.execute(
        task=test_goal,
        context={"project_path": str(project_root)}
    )

    print("-" * 70)
    if result.success:
        print(f"✅ УСПЕХ ({name})! План получен.")
        print(f"Использованная модель: {planner.model}")
        print(f"Потрачено токенов: {result.tokens_used}")
        print("\nСгенерированный план:")
        for step in result.data:
            print(f"  Шаг {step['step']}: {step['description']}")
            print(f"    Файлы: {', '.join(step.get('target_files', []))}")
    else:
        print(f"❌ ПРОВАЛ ({name})! Планировщик не смог сформировать план.")
        print(f"Ошибка: {result.error_message}")


async def main() -> None:
    try:
        settings = load_settings(project_path_override=project_root)
    except SystemExit as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return

    run_paths = create_run_directory(Path("logs"))
    logger = setup_agent_logger("test_planner", "planner", "dynamic", run_paths, settings.log_level)

    # 1. Тест OpenRouter (бесплатные модели)
    # Берём первую модель из динамического списка или резервную
    await test_provider(
        name="OpenRouter (Free)",
        settings=settings,
        logger=logger,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        model=settings.planner_models[0] if settings.planner_models else "qwen/qwen-2.5-72b-instruct:free"
    )

    # 2. Тест DeepSeek (если есть ключ)
    if settings.deepseek_api_key and len(settings.deepseek_api_key) > 10:
        await test_provider(
            name="DeepSeek API",
            settings=settings,
            logger=logger,
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            model="deepseek-chat"
        )
    else:
        print(f"\n{'=' * 70}")
        print("⏭️ DeepSeek пропущен: DEEPSEEK_API_KEY не указан или пуст в .env")
        print(f"{'=' * 70}")

    print(f"\n📂 Полные логи сохранены в: {run_paths.run_dir}")


if __name__ == "__main__":
    asyncio.run(main())