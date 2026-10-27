# openrouter_utils.py
"""Утилиты для работы с OpenRouter API (получение и сортировка бесплатных моделей)."""
from __future__ import annotations
import httpx
import logging
from typing import Optional

logger = logging.getLogger("openrouter_utils")


# Черный список: модели, которые не подходят для генерации кода/текста
# или блокируют прямой API-доступ.
BLACKLIST_KEYWORDS = [
    "thinkingmachines",  # блокируют прямой API (требуют UI)
    "lyria",             # Google Lyria — генерация музыки/аудио
    "image", "video", "audio", "music", "tts", "whisper" # мультимодальные
]


async def get_free_models(
        sort_by: str = "context_length",
        limit: int = 5,
        min_context_length: int = 32000,
) -> list[str]:
    """
    Запрашивает список моделей OpenRouter, фильтрует только полностью бесплатные
    и сортирует по заданному критерию.

    :param sort_by: "context_length" (по убыванию контекста) или "top-weekly" (по популярности)
    :param limit: Максимальное количество моделей для возврата
    :param min_context_length: Минимальный размер контекста (фильтр, по умолчанию 32K)
    :return: Список ID моделей (например, ['qwen/qwen-2.5-72b-instruct:free', ...])
    """
    url = "https://openrouter.ai/api/v1/models"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        # Фильтруем только полностью бесплатные модели
        free_models = []
        for model in data.get("data", []):
            pricing = model.get("pricing", {})
            # Проверяем, что и вход, и выход бесплатны (строка "0")
            if str(pricing.get("prompt", "1")) == "0" and str(pricing.get("completion", "1")) == "0":
                model_id = model.get("id", "")
                model_id_lower = model_id.lower()

                # НОВОЕ: Игнорируем модели, блокирующие прямой API-доступ
                if "thinkingmachines" in model_id.lower():
                    continue
                # Пропускаем модели из черного списка
                if any(kw in model_id_lower for kw in BLACKLIST_KEYWORDS):
                    continue

                context_length = model.get("context_length", 0)

                # Фильтр по минимальному контексту (для планирования нужно >=32K)
                if model_id and context_length >= min_context_length:
                    free_models.append({
                        "id": model_id,
                        "context_length": context_length,
                        "name": model.get("name", ""),
                    })

        if not free_models:
            logger.warning("Не найдено бесплатных моделей с context_length >= %d", min_context_length)
            return []

        # Сортировка
        if sort_by == "context_length":
            # Сортируем по убыванию контекста (больше = лучше для планирования)
            free_models.sort(key=lambda m: m["context_length"], reverse=True)
            logger.info("Сортировка по context_length (desc)")
        elif sort_by == "top-weekly":
            logger.info("Сортировка по top-weekly (используем context_length как fallback)")
            free_models.sort(key=lambda m: m["context_length"], reverse=True)
        else:
            logger.warning("Неизвестный sort_by='%s', используем context_length", sort_by)
            free_models.sort(key=lambda m: m["context_length"], reverse=True)

        # Берём топ-N
        result = [m["id"] for m in free_models[:limit]]

        logger.info("Найдено %d бесплатных моделей (отфильтровано %d):", len(result), len(free_models))
        for m in free_models[:limit]:
            logger.info("  - %s (context: %d)", m["id"], m["context_length"])

        return result

    except httpx.HTTPError as e:
        logger.error("Ошибка HTTP при запросе моделей OpenRouter: %s", e)
        return []
    except Exception as e:
        logger.error("Неожиданная ошибка при запросе моделей OpenRouter: %s", e)
        return []