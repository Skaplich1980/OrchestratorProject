# tests/test_planner_fallback.py
"""Тесты плоской очереди retry в Planner."""
from __future__ import annotations
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from openai import RateLimitError, APITimeoutError, APIError

from agents.planner import QwenPlanner
from config import Settings
from prompt_manager import PromptManager


# ---------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------
@pytest.fixture
def env_vars(tmp_path):
    return {
        "PRIORITY_KEYWORDS": "qwen3, deepseek-v3",
        "ORCAROUTER_API_KEY": "sk-orca-test-key-1234567890",
        "OPENROUTER_API_KEY": "sk-or-test-key-1234567890",
        "AGNES_API_KEY": "sk-agnes-test-key-1234567890",
        "MODELS_CACHE_PATH": str(tmp_path / "cache.json"),
        "DEFAULT_PROJECT_PATH": str(tmp_path),
        "MAX_RETRIES": "2",
    }


@pytest.fixture
def settings(monkeypatch, env_vars, tmp_path):
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    return Settings()


@pytest.fixture
def planner(settings):
    logger = MagicMock()
    prompt_manager = MagicMock(spec=PromptManager)
    prompt_manager.build_planner_system_prompt.return_value = "system"
    prompt_manager.build_planner_user_prompt.return_value = "user"
    return QwenPlanner(settings, logger, prompt_manager)


def _make_mock_response(content: str = '[{"step": 1}]'):
    """Создаёт мок ответа от OpenAI."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


# ---------------------------------------------------------------------
# Тесты
# ---------------------------------------------------------------------
class TestPlannerFallback:
    @pytest.mark.asyncio
    async def test_first_config_succeeds(self, planner):
        """Первая конфигурация успешна — она и используется."""
        queue = [
            {
                "router": "orcarouter",
                "model_id": "qwen/qwen-3-32b:free",
                "base_url": "https://api.orcarouter.ai/v1",
                "api_key": "test-key",
                "priority_rank": 0,
            }
        ]

        with patch.object(planner._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.planner.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(return_value=_make_mock_response())
                mock_client_cls.return_value = mock_client

                result = await planner.execute("test task", {"project_path": "/tmp"})

        assert result.success is True
        assert planner.model == "qwen/qwen-3-32b:free"

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_fallback(self, planner):
        """RateLimitError на первой конфигурации → мгновенный переход ко второй."""
        queue = [
            {
                "router": "orcarouter",
                "model_id": "qwen/qwen-3-32b:free",
                "base_url": "https://api.orcarouter.ai/v1",
                "api_key": "test-key-1",
                "priority_rank": 0,
            },
            {
                "router": "openrouter",
                "model_id": "qwen/qwen-2.5-72b-instruct:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key-2",
                "priority_rank": 1,
            },
        ]

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("Rate limit exceeded", response=MagicMock(), body=None)
            return _make_mock_response()

        with patch.object(planner._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.planner.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
                mock_client_cls.return_value = mock_client

                result = await planner.execute("test task", {"project_path": "/tmp"})

        assert result.success is True
        assert planner.model == "qwen/qwen-2.5-72b-instruct:free"
        assert call_count == 2  # Первая попытка + вторая конфигурация

    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback(self, planner):
        """APITimeoutError → переход к следующей конфигурации."""
        queue = [
            {
                "router": "orcarouter",
                "model_id": "qwen/qwen-3-32b:free",
                "base_url": "https://api.orcarouter.ai/v1",
                "api_key": "test-key-1",
                "priority_rank": 0,
            },
            {
                "router": "openrouter",
                "model_id": "deepseek/deepseek-chat:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key-2",
                "priority_rank": 1,
            },
        ]

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APITimeoutError(request=MagicMock())
            return _make_mock_response()

        with patch.object(planner._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.planner.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
                mock_client_cls.return_value = mock_client

                result = await planner.execute("test task", {"project_path": "/tmp"})

        assert result.success is True
        assert planner.model == "deepseek/deepseek-chat:free"

    @pytest.mark.asyncio
    async def test_model_not_found_triggers_fallback(self, planner):
        """404 (model_not_found) → переход к следующей конфигурации."""
        queue = [
            {
                "router": "orcarouter",
                "model_id": "nonexistent/model",
                "base_url": "https://api.orcarouter.ai/v1",
                "api_key": "test-key-1",
                "priority_rank": 0,
            },
            {
                "router": "openrouter",
                "model_id": "qwen/qwen-3-32b:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key-2",
                "priority_rank": 1,
            },
        ]

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise APIError("model_not_found", request=MagicMock(), body=None)
            return _make_mock_response()

        with patch.object(planner._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.planner.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
                mock_client_cls.return_value = mock_client

                result = await planner.execute("test task", {"project_path": "/tmp"})

        assert result.success is True
        assert planner.model == "qwen/qwen-3-32b:free"

    @pytest.mark.asyncio
    async def test_all_configs_fail_returns_error(self, planner):
        """Все конфигурации провалились → AgentResult(success=False)."""
        queue = [
            {
                "router": "orcarouter",
                "model_id": "qwen/qwen-3-32b:free",
                "base_url": "https://api.orcarouter.ai/v1",
                "api_key": "test-key-1",
                "priority_rank": 0,
            }
        ]

        with patch.object(planner._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.planner.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    side_effect=RateLimitError("Rate limit", response=MagicMock(), body=None)
                )
                mock_client_cls.return_value = mock_client

                result = await planner.execute("test task", {"project_path": "/tmp"})

        assert result.success is False
        assert "Все конфигурации планирования недоступны" in result.error_message

    @pytest.mark.asyncio
    async def test_json_parse_error_triggers_fallback(self, planner):
        """Невалидный JSON от модели → переход к следующей конфигурации."""
        queue = [
            {
                "router": "orcarouter",
                "model_id": "qwen/qwen-3-32b:free",
                "base_url": "https://api.orcarouter.ai/v1",
                "api_key": "test-key-1",
                "priority_rank": 0,
            },
            {
                "router": "openrouter",
                "model_id": "deepseek/deepseek-chat:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key-2",
                "priority_rank": 1,
            },
        ]

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Возвращаем невалидный JSON
                return _make_mock_response(content="This is not JSON")
            return _make_mock_response()

        with patch.object(planner._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.planner.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
                mock_client_cls.return_value = mock_client

                result = await planner.execute("test task", {"project_path": "/tmp"})

        assert result.success is True
        assert planner.model == "deepseek/deepseek-chat:free"