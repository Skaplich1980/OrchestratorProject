# tests/test_model_resolution.py
"""Интеграционные тесты: Settings + ModelResolver."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch

import pytest

from config import Settings
from model_resolver import ModelResolver


# ---------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------
@pytest.fixture
def env_vars(tmp_path):
    return {
        "PRIORITY_KEYWORDS": "qwen3, deepseek-v3, qwen-2.5-72b",
        "ORCAROUTER_API_KEY": "sk-orca-test-key-1234567890",
        "ORCAROUTER_BASE_URL": "https://api.orcarouter.ai/v1",
        "OPENROUTER_API_KEY": "sk-or-test-key-1234567890",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "AGNES_API_KEY": "sk-agnes-test-key-1234567890",
        "MODELS_CACHE_PATH": str(tmp_path / "cache.json"),
        "DEFAULT_PROJECT_PATH": str(tmp_path),
    }


@pytest.fixture
def settings(monkeypatch, env_vars):
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    return Settings()


# ---------------------------------------------------------------------
# Тесты Settings
# ---------------------------------------------------------------------
class TestSettingsParsing:
    def test_settings_loads_priority_keywords(self, settings):
        """Settings корректно парсит PRIORITY_KEYWORDS."""
        assert settings.priority_keywords == ["qwen3", "deepseek-v3", "qwen-2.5-72b"]

    def test_settings_routers_config_order(self, settings):
        """Порядок роутеров: orcarouter (priority=0), openrouter (priority=1)."""
        routers = settings.routers_config
        assert len(routers) == 2
        assert routers[0]["name"] == "orcarouter"
        assert routers[0]["priority"] == 0
        assert routers[1]["name"] == "openrouter"
        assert routers[1]["priority"] == 1

    def test_settings_routers_config_has_credentials(self, settings):
        """Каждый роутер имеет base_url и api_key."""
        for router in settings.routers_config:
            assert "base_url" in router
            assert "api_key" in router
            assert router["base_url"].startswith("https://")
            assert len(router["api_key"]) > 10


# ---------------------------------------------------------------------
# Интеграционные тесты
# ---------------------------------------------------------------------
class TestEndToEnd:
    def test_end_to_end_cache_creation(self, settings, tmp_path):
        """Полный цикл: Settings → ModelResolver → кэш создаётся."""
        resolver = ModelResolver(settings)
        cache_path = Path(settings.models_cache_path)

        assert not cache_path.exists()

        # Мокаем _fetch_models_from_router, чтобы не делать реальных запросов
        def fake_fetch(router):
            if router["name"] == "orcarouter":
                return ["orcarouter/qwen-3-32b-free"]
            return ["qwen/qwen-2.5-72b-instruct:free"]

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch):
            queue = resolver.get_execution_queue()

        assert cache_path.exists()
        assert len(queue) > 0
        # Проверяем, что qwen3 (priority_rank=0) идёт первым
        assert queue[0]["priority_rank"] == 0

    def test_end_to_end_cache_reuse(self, settings, tmp_path):
        """Повторный вызов использует кэш, не делает сканирование."""
        resolver = ModelResolver(settings)

        # Первый вызов — создаёт кэш
        def fake_fetch(router):
            return ["orcarouter/qwen-3-32b-free"]

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch) as mock_fetch:
            queue1 = resolver.get_execution_queue()
            assert mock_fetch.call_count == 2  # orcarouter + openrouter

        # Второй вызов — использует кэш
        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch) as mock_fetch:
            queue2 = resolver.get_execution_queue()
            assert mock_fetch.call_count == 0  # Не вызывался

        assert queue1 == queue2

    def test_empty_priority_keywords_uses_fallback(self, monkeypatch, tmp_path):
        """Если PRIORITY_KEYWORDS пуст, используется fallback на KNOWN_FREE_MODELS."""
        env_vars = {
            "PRIORITY_KEYWORDS": "",  # Пусто
            "ORCAROUTER_API_KEY": "sk-orca-test-key-1234567890",
            "OPENROUTER_API_KEY": "sk-or-test-key-1234567890",
            "AGNES_API_KEY": "sk-agnes-test-key-1234567890",
            "MODELS_CACHE_PATH": str(tmp_path / "cache.json"),
            "DEFAULT_PROJECT_PATH": str(tmp_path),
        }
        for k, v in env_vars.items():
            monkeypatch.setenv(k, v)

        settings = Settings()
        resolver = ModelResolver(settings)

        def fake_fetch(router):
            return []  # Ничего не найдено

        with patch.object(resolver, "_fetch_models_from_router", side_effect=fake_fetch):
            queue = resolver.get_execution_queue()

        # Должны использоваться KNOWN_FREE_MODELS
        assert len(queue) > 0