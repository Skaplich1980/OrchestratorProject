# tests/test_step1_find_models.py
"""
ШАГ 1: Ручная проверка поиска и сортировки бесплатных моделей OpenRouter.
"""
import sys
import asyncio
import logging
from pathlib import Path

# КРИТИЧЕСКИ ВАЖНО: Добавляем корень проекта в sys.path.
# Path(__file__).parent.parent указывает на D:\Project\OrchestratorProject
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Теперь импорт сработает корректно
from openrouter_utils import get_free_models

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


async def main() -> None:
    print("\n" + "=" * 70)
    print("ШАГ 1: Тест поиска бесплатных моделей OpenRouter")
    print("=" * 70)

    print("\n[TEST] Запрос моделей с сортировкой по context_length (desc)...")
    models = await get_free_models(sort_by="context_length", limit=5, min_context_length=32000)

    if models:
        print(f"\n✅ УСПЕХ! Найдено бесплатных моделей: {len(models)}")
        print("Приоритетный список для оркестратора:")
        for i, model_id in enumerate(models, 1):
            print(f"  {i}. {model_id}")
    else:
        print("\n❌ ПРОВАЛ: Не найдено бесплатных моделей с context_length >= 32000.")
        print("Возможные причины:")
        print("  1. OpenRouter API временно недоступен или блокируется.")
        print("  2. Провайдер убрал бесплатные тарифы.")
        print("  3. Файл openrouter_utils.py отсутствует в корне проекта.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())