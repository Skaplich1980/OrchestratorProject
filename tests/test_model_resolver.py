"""Тесты сервиса ModelResolver: кэширование, умный поиск, приоритизация."""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from model_resolver import ModelResolver, KNOWN_FREE_MODELS


# ---------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------
@pytest.fixture
def env_vars(tmp_path):
    """Набор переменных окружения для тестов."""
    cache_path = tmp_path / "models_cache.json"
    return {
        "PRIORITY_KEYWORDS": "qwen3, deepseek-v3, qwen-2.5-72b",
        "ORCAROUTER_API_KEY": "sk-orca-test-key-1234567890",
        "ORCAROUTER_BASE_URL": "https://api.orcarouter.ai/v1",
        "OPENROUTER_API_KEY": "sk-or-test-key-1234567890",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "AGNES_API_KEY": "sk-agnes-test-key-1234567890",
        "MODELS_CACHE_PATH": str(cache_path),
        "DEFAULT_PROJECT_PATH": str(tmp_path),
    }


@pytest.fixture
def settings(monkeypatch, env_vars, tmp_path):
    """Создаёт Settings с тестовыми значениями."""
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    # Создаём директорию проекта, чтобы load_settings не падал
    from config import Settings
    return Settings()


@pytest.fixture
def resolver(settings):
    return ModelResolver(settings)


def _write_cache(path: Path, queue: list, age_days: int = 0) -> None:
    """Записывает тестовый кэш с заданным возрастом."""
    data = {
        "last_updated": (datetime.now() - timedelta(days=age_days)).isoformat(),
        "execution_queue": queue,
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------
# Тесты загрузки кэша
# ---------------------------------------------------------------------
class TestCacheLoading:
    def test_cache_loaded_when_fresh(self, resolver, env_vars):
        """Свежий кэш (< 7 дней) используется без сканирования."""
        cache_path = Path(env_vars["MODELS_CACHE_PATH"])
        expected_queue = [
            {"router": "orcarouter", "model_id": "qwen/qwen-3-32b:free", "priority_rank": 0}
        ]
        _write_cache(cache_path, expected_queue, age_days=3)

        result = resolver.get_execution_queue()

        assert result == expected_queue

    def test_cache_ignored_when_stale(self, resolver, env_vars):
        """Кэш старше 7 дней игнорируется, запускается сканирование."""
        cache_path = Path(env_vars["MODELS_CACHE_PATH"])
        _write_cache(cache_path, [{"old": "data"}], age_days=10)

        # Мокаем _build_queue, чтобы не делать реальных HTTP-запросов
        with patch.object(resolver, "_build_queue", return_value=[{"new": "data"}]) as mock_build:
            result = resolver.get_execution_queue()
            mock_build.assert_called_once()

        assert result == [{"new": "data"}]

    def test_cache_ignored_when_missing(self, resolver, env_vars):
        """При отсутствии файла кэша запускается сканирование."""
        cache_path = Path(env_vars["MODELS_CACHE_PATH"])
        assert not cache_path.exists()

        with patch.object(resolver, "_build_queue", return_value=[{"scanned": True}]) as mock_build:
            result = resolver.get_execution_queue()
            mock_build.assert_called_once()

        assert result == [{"scanned": True}]

    def test_cache_ignored_when_corrupted(self, resolver, env_vars):
        """Повреждённый JSON игнорируется, запускается сканирование."""
        cache_path = Path(env_vars["MODELS_CACHE_PATH"])
        cache_path.write_text("{invalid json", encoding="utf-8")

        with patch.object(resolver, "_build_queue", return_value=[]):
            result = resolver.get_execution_queue()

        assert isinstance(result, list)


# ---------------------------------------------------------------------
# Тесты умного поиска и приоритизации
# ---------------------------------------------------------------------
class TestSmartMatching:
    def test_smart_matching_by_keywords(self, resolver):
        """Ключевое слово 'qwen3' находит модель 'qwen/qwen-3-32b:free'."""
        fake_models_by_router = {
            "orcarouter": ["orcarouter/deepseek-v3-free", "orcarouter/qwen-2.5-72b-free"],
            "openrouter": ["qwen/qwen-3-32b:free", "deepseek/deepseek-chat:free"],
        }

        def fake_fetch(router):
            return fake_models_by_router.get(router["name"], [])

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch):
            queue = resolver._build_queue()

        # Ключ 'qwen3' должен найти 'qwen/qwen-3-32b:free'
        model_ids = [item["model_id"] for item in queue]
        assert "qwen/qwen-3-32b:free" in model_ids

    def test_priority_sorting(self, resolver):
        """Модели сортируются по приоритету ключевого слова, затем по приоритету роутера."""
        fake_models_by_router = {
            "orcarouter": ["orcarouter/deepseek-v3-free"],  # priority_rank=1
            "openrouter": ["qwen/qwen-3-32b:free"],  # priority_rank=0
        }

        def fake_fetch(router):
            return fake_models_by_router.get(router["name"], [])

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch):
            queue = resolver._build_queue()

        # qwen3 (rank=0) должен идти раньше deepseek-v3 (rank=1)
        assert queue[0]["model_id"] == "qwen/qwen-3-32b:free"
        assert queue[0]["priority_rank"] == 0
        assert queue[1]["model_id"] == "orcarouter/deepseek-v3-free"
        assert queue[1]["priority_rank"] == 1

    def test_router_priority_within_same_keyword(self, resolver):
        """Внутри одного ключевого слова OrcaRouter (priority=0) идёт раньше OpenRouter (priority=1)."""
        fake_models_by_router = {
            "orcarouter": ["orcarouter/qwen-2.5-72b-free"],
            "openrouter": ["qwen/qwen-2.5-72b-instruct:free"],
        }

        def fake_fetch(router):
            return fake_models_by_router.get(router["name"], [])

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch):
            queue = resolver._build_queue()

        # Оба соответствуют ключу 'qwen-2.5-72b' (rank=2)
        # OrcaRouter должен идти первым
        qwen_matches = [item for item in queue if "qwen-2.5-72b" in item["model_id"]]
        assert len(qwen_matches) == 2
        assert qwen_matches[0]["router"] == "orcarouter"
        assert qwen_matches[1]["router"] == "openrouter"

    def test_deduplication(self, resolver):
        """Одинаковые модели не дублируются в очереди."""
        # Обе модели содержат 'qwen3', но с разными ID — дедупликация по (router, model_id)
        fake_models_by_router = {
            "orcarouter": ["orcarouter/qwen3-free", "orcarouter/qwen3-free"],  # дубликат
            "openrouter": [],
        }

        def fake_fetch(router):
            return fake_models_by_router.get(router["name"], [])

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch):
            queue = resolver._build_queue()

        model_ids = [item["model_id"] for item in queue]
        assert model_ids.count("orcarouter/qwen3-free") == 1


# ---------------------------------------------------------------------
# Тесты fallback на KNOWN_FREE_MODELS
# ---------------------------------------------------------------------
class TestFallbackToKnownModels:
    def test_fallback_when_api_unavailable(self, resolver):
        """При недоступности API используется KNOWN_FREE_MODELS."""

        def fake_fetch(router):
            # Имитация недоступности API
            return []

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch):
            queue = resolver._build_queue()

        # Должны использоваться модели из KNOWN_FREE_MODELS
        assert len(queue) > 0
        # Проверяем, что все модели из KNOWN_FREE_MODELS присутствуют
        all_known = []
        for models in KNOWN_FREE_MODELS.values():
            all_known.extend(models)

        queue_model_ids = {item["model_id"] for item in queue}
        # Хотя бы некоторые известные модели должны быть в очереди
        assert len(queue_model_ids & set(all_known)) > 0

    def test_empty_queue_when_no_matches_and_no_known(self, resolver):
        """Если ничего не найдено и KNOWN_FREE_MODELS пуст, очередь пустая."""

        def fake_fetch(router):
            return []

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch):
            with patch("model_resolver.KNOWN_FREE_MODELS", {}):
                queue = resolver._build_queue()

        assert queue == []


# ---------------------------------------------------------------------
# Тесты сохранения кэша
# ---------------------------------------------------------------------
class TestCacheSaving:
    def test_cache_saved_after_build(self, resolver, env_vars):
        """После построения очереди кэш сохраняется в файл."""
        cache_path = Path(env_vars["MODELS_CACHE_PATH"])
        assert not cache_path.exists()

        with patch.object(resolver, "_build_queue", return_value=[{"test": "data"}]):
            resolver.get_execution_queue()

        assert cache_path.exists()
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "last_updated" in data
        assert data["execution_queue"] == [{"test": "data"}]