# openrouter_utils.py
"""Утилиты для работы с OpenRouter API: получение и фильтрация бесплатных моделей.

Используется ModelResolver для безопасного получения ТОЛЬКО бесплатных моделей.
Поддерживает оба формата суффиксов:
  - OpenRouter:  id заканчивается на ':free'
  - OrcaRouter:  id заканчивается на '-free'
"""
from __future__ import annotations

import httpx
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openrouter_utils")

# Черный список: модели, которые не подходят для генерации кода/текста
# или блокируют прямой API-доступ.
BLACKLIST_KEYWORDS: List[str] = [
    "thinkingmachines",   # блокируют прямой API (требуют UI)
    "lyria",              # Google Lyria — генерация музыки/аудио
    "image", "video", "audio", "music", "tts", "whisper",  # мультимодальные
]


def is_free_model(model_id: str, pricing: Optional[Dict[str, Any]] = None) -> bool:
    """Определяет, является ли модель бесплатной.

    Критерии (любой из):
    1. ID заканчивается на ':free' (OpenRouter) или '-free' (OrcaRouter).
    2. pricing.prompt == '0' И pricing.completion == '0'.
    """
    mid = model_id.lower()
    if mid.endswith(":free") or mid.endswith("-free"):
        return True
    if pricing:
        prompt_price = str(pricing.get("prompt", "1"))
        completion_price = str(pricing.get("completion", "1"))
        if prompt_price == "0" and completion_price == "0":
            return True
    return False


def _is_blacklisted(model_id: str) -> bool:
    """Проверяет, что модель не в чёрном списке."""
    mid = model_id.lower()
    return any(kw in mid for kw in BLACKLIST_KEYWORDS)


async def get_free_models(
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
    sort_by: str = "context_length",
    limit: int = 5,
    min_context_length: int = 32000,
) -> List[Dict[str, Any]]:
    """Запрашивает список моделей и возвращает только бесплатные.

    :param base_url: Базовый URL роутера (OpenRouter / OrcaRouter).
    :param api_key: API-ключ (для OrcaRouter обязателен, для OpenRouter — нет).
    :param sort_by: 'context_length' или 'top-weekly'.
    :param limit: Максимальное количество моделей для возврата.
    :param min_context_length: Минимальный размер контекста (фильтр).
    :return: Список словарей {'id', 'context_length', 'name'}.
    """
    url = f"{base_url.rstrip('/')}/models"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.error("HTTP-ошибка при запросе моделей (%s): %s", base_url, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.error("Неожиданная ошибка при запросе моделей: %s", exc)
        return []

    free_models: List[Dict[str, Any]] = []
    for model in data.get("data", []):
        model_id: str = model.get("id", "")
        if not model_id:
            continue

        # Фильтр по чёрному списку
        if _is_blacklisted(model_id):
            continue

        # Фильтр по бесплатности
        pricing = model.get("pricing")
        if not is_free_model(model_id, pricing):
            continue

        context_length: int = int(model.get("context_length", 0) or 0)
        if context_length < min_context_length:
            continue

        free_models.append({
            "id": model_id,
            "context_length": context_length,
            "name": model.get("name", ""),
        })

    if not free_models:
        logger.warning(
            "Не найдено бесплатных моделей с context_length >= %d на %s",
            min_context_length, base_url,
        )
        return []

    # Сортировка
    if sort_by == "context_length":
        free_models.sort(key=lambda m: m["context_length"], reverse=True)
    else:
        free_models.sort(key=lambda m: m["context_length"], reverse=True)

    result = free_models[:limit]
    logger.info(
        "Найдено %d бесплатных моделей на %s (показано топ-%d):",
        len(result), base_url, limit,
    )
    for m in result:
        logger.info("  - %s (context: %d)", m["id"], m["context_length"])

    return result


def get_free_model_ids(
    base_url: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
    sort_by: str = "context_length",
    limit: int = 5,
    min_context_length: int = 32000,
) -> List[str]:
    """Удобная обёртка: возвращает только список ID (для ModelResolver)."""
    models = asyncio_run_if_needed(
        get_free_models(
            base_url=base_url,
            api_key=api_key,
            sort_by=sort_by,
            limit=limit,
            min_context_length=min_context_length,
        )
    )
    return [m["id"] for m in models]


def asyncio_run_if_needed(coro):
    """Запускает корутину, если нет активного event loop (для sync-вызовов)."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Мы внутри async-контекста — вернуть корутину нельзя,
        # но для sync-вызовов из ModelResolver это не случится.
        raise RuntimeError("get_free_model_ids вызван внутри async-контекста")
    return asyncio.run(coro)