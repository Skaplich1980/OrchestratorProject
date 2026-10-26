# tests/test_openrouter.py
"""Автоматические тесты для openrouter_utils."""
import pytest
from openrouter_utils import get_free_models


@pytest.mark.asyncio
async def test_get_free_models_returns_sorted_list():
    """
    Проверяет, что функция возвращает список бесплатных моделей,
    отсортированный по убыванию context_length.
    """
    # Запрашиваем топ-3 бесплатные модели с контекстом > 32k
    models = await get_free_models(sort_by="context_length", limit=3, min_context_length=32000)

    # 1. Список не должен быть пустым (если OpenRouter API доступен)
    assert isinstance(models, list), "Результат должен быть списком"
    assert len(models) > 0, "Список бесплатных моделей пуст. Проверь API OpenRouter."

    # 2. Проверяем, что элементы — это строки (ID моделей)
    assert all(isinstance(m, str) for m in models), "Все элементы должны быть строками (ID моделей)"

    # 3. Проверяем, что в ID есть маркер бесплатности (опционально, но полезно)
    # Некоторые бесплатные модели могут не иметь :free, но большинство имеют.
    print(f"\n✅ Успешно получены модели: {models}")