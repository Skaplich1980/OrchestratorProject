# model_resolver.py

"""Сервис разрешения и кэширования доступных LLM-моделей по приоритету."""
from __future__ import annotations
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Settings

logger = logging.getLogger(__name__)

CACHE_MAX_AGE_DAYS = 7

# Резервные списки моделей на случай, если эндпоинт /models недоступен или возвращает ошибку
KNOWN_FREE_MODELS = {
    "orcarouter": [
        "orcarouter/deepseek-v3-free",
        "orcarouter/qwen-2.5-72b-free",
        "orcarouter/qwen3-free",
        "orcarouter/deepseek-r1-free",
        "orca/orcacyber-flash-1.0-free",  # ← новая кибермодель
    ],
    "openrouter": [
        "qwen/qwen-2.5-72b-instruct:free",
        "deepseek/deepseek-chat:free",
        "qwen/qwen-3-32b:free",
        "deepseek/deepseek-r1:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",  # ← Nemotron для планирования
        "nvidia/nemotron-3.5-lightning:free",      # ← быстрая модель
    ]
}


def _normalize(s: str) -> str:
    """Нормализация строки для сравнения: убирает '-', '_', '.', пробелы и приводит к нижнему регистру.

    Это позволяет, например, ключевому слову 'qwen3' соответствовать модели 'qwen-3-32b'.
    """
    return re.sub(r"[-_.\s]", "", s.lower())


class ModelResolver:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_path = Path(settings.models_cache_path).expanduser().resolve()

    def get_execution_queue(self) -> List[Dict[str, Any]]:
        """
        Возвращает плоский список конфигураций для выполнения запросов.
        Список отсортирован по приоритету ключевого слова, а затем по приоритету роутера.
        """
        cache = self._load_cache()
        if cache is not None:
            logger.info(f"Использован кэш моделей из {self.cache_path}")
            return cache

        logger.info("Кэш устарел или отсутствует. Выполняется сканирование роутеров...")
        queue = self._build_queue()
        self._save_cache(queue)
        return queue

    def _load_cache(self) -> Optional[List[Dict[str, Any]]]:
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            last_updated_str = data.get("last_updated")
            if not last_updated_str:
                return None

            last_updated = datetime.fromisoformat(last_updated_str)
            if datetime.now() - last_updated > timedelta(days=CACHE_MAX_AGE_DAYS):
                logger.info("Кэш моделей устарел (> 7 дней).")
                return None

            return data.get("execution_queue", [])
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Не удалось прочитать кэш моделей: {e}. Будет выполнено новое сканирование.")
            return None

    def _save_cache(self, queue: List[Dict[str, Any]]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_updated": datetime.now().isoformat(),
                "execution_queue": queue,
            }
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Кэш моделей успешно сохранён в {self.cache_path}")
        except IOError as e:
            logger.error(f"Не удалось сохранить кэш моделей: {e}")

    def _build_queue(self) -> List[Dict[str, Any]]:
        """Сканирует роутеры и формирует приоритетную очередь."""
        raw_models: List[Dict[str, Any]] = []

        for router in self.settings.routers_config:
            models = self._fetch_models_from_router(router)
            for model_id in models:
                raw_models.append({
                    "router": router["name"],
                    "model_id": model_id,
                    "base_url": router["base_url"],
                    "api_key": router["api_key"],
                    "router_priority": router["priority"],
                })

        execution_queue: List[Dict[str, Any]] = []

        # Умное сопоставление: ищем ключевые слова в ID модели (с нормализацией)
        for keyword_rank, keyword in enumerate(self.settings.priority_keywords):
            norm_keyword = _normalize(keyword)
            matched_models = []
            for model_info in raw_models:
                if norm_keyword in _normalize(model_info["model_id"]):
                    matched_models.append(model_info)

            # Сортируем внутри одного приоритета: сначала роутеры с высшим приоритетом (0, затем 1)
            matched_models.sort(key=lambda x: x["router_priority"])

            for model_info in matched_models:
                # Избегаем дубликатов в очереди
                if not any(
                        item["router"] == model_info["router"] and item["model_id"] == model_info["model_id"]
                        for item in execution_queue
                ):
                    execution_queue.append({
                        "router": model_info["router"],
                        "model_id": model_info["model_id"],
                        "base_url": model_info["base_url"],
                        "api_key": model_info["api_key"],
                        "priority_rank": keyword_rank,
                    })

        if not execution_queue:
            logger.warning(
                "Не найдено ни одной модели по приоритетным ключевым словам. Используются запасные варианты.")
            # Фоллбэк: добавляем все известные модели, если умный поиск ничего не дал
            # Храним router_priority во внутреннем списке для корректной финальной сортировки
            fallback_entries: List[tuple[int, int, Dict[str, Any]]] = []
            for router in self.settings.routers_config:
                for model_id in KNOWN_FREE_MODELS.get(router["name"], []):
                    fallback_entries.append((99, router["priority"], {
                        "router": router["name"],
                        "model_id": model_id,
                        "base_url": router["base_url"],
                        "api_key": router["api_key"],
                        "priority_rank": 99,  # Низкий приоритет
                    }))
            fallback_entries.sort(key=lambda t: (t[0], t[1]))
            execution_queue = [t[2] for t in fallback_entries]
            return execution_queue

        # Финальная сортировка: сначала по priority_rank (возрастание), затем по router_priority
        # router_priority извлекается из raw_models для корректной сортировки,
        # но в финальные элементы очереди НЕ включается (тесты сравнивают на равенство без него).
        router_priority_map: Dict[tuple[str, str], int] = {}
        for m in raw_models:
            router_priority_map[(m["router"], m["model_id"])] = m["router_priority"]

        def _sort_key(entry: Dict[str, Any]) -> tuple[int, int]:
            pr = entry["priority_rank"]
            rp = router_priority_map.get((entry["router"], entry["model_id"]), 99)
            return (pr, rp)

        execution_queue.sort(key=_sort_key)
        return execution_queue

    def _fetch_models_from_router(self, router: Dict[str, Any]) -> List[str]:
        """Получает ТОЛЬКО бесплатные модели роутера через openrouter_utils.

        Использует синхронную обёртку get_free_model_ids, которая:
        1. Делает HTTP-запрос к /models эндпоинту роутера
        2. Фильтрует по суффиксам :free и -free
        3. Фильтрует по pricing.prompt == '0' И pricing.completion == '0'
        4. Исключает чёрный список (thinkingmachines, мультимодальные)
        5. Фильтрует по min_context_length
        """
        try:
            from openrouter_utils import get_free_model_ids
            model_ids = get_free_model_ids(
                base_url=router["base_url"],
                api_key=router["api_key"],
                sort_by="context_length",
                limit=50,  # С запасом — дальнейшая фильтрация по keywords
                min_context_length=16000,  # Для ревьюера хватит 16K
            )
            logger.info(
                "Роутер %s: получено %d бесплатных моделей через openrouter_utils.",
                router["name"], len(model_ids),
            )
            return model_ids
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Не удалось получить бесплатные модели от %s (%s). "
                "Используются известные бесплатные модели.",
                router["name"], exc,
            )
            return KNOWN_FREE_MODELS.get(router["name"], [])