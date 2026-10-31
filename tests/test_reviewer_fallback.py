"""Тесты плоской очереди retry в Reviewer."""
from __future__ import annotations
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from openai import RateLimitError, APITimeoutError

from agents.reviewer import QwenReviewer
from agents.base import AgentResult
from config import Settings


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
def reviewer(settings):
    logger = MagicMock()
    return QwenReviewer(settings, logger)


def _make_mock_response(content: str = '{"success": true, "reason": "OK"}'):
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
class TestReviewerFallback:
    @pytest.mark.asyncio
    async def test_first_config_succeeds(self, reviewer):
        """Первая конфигурация успешна."""
        queue = [
            {
                "router": "orcarouter",
                "model_id": "qwen/qwen-3-32b:free",
                "base_url": "https://api.orcarouter.ai/v1",
                "api_key": "test-key",
                "priority_rank": 0,
            }
        ]

        executor_result = AgentResult(success=True, data={"stdout": "OK", "stderr": "", "returncode": 0})

        with patch.object(reviewer._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.reviewer.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(return_value=_make_mock_response())
                mock_client_cls.return_value = mock_client

                result = await reviewer.execute("test task", {"executor_result": executor_result, "step_number": 1})

        assert result.success is True
        assert reviewer.model == "qwen/qwen-3-32b:free"

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_fallback(self, reviewer):
        """RateLimitError → переход ко второй конфигурации."""
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
                raise RateLimitError("Rate limit", response=MagicMock(), body=None)
            return _make_mock_response()

        executor_result = AgentResult(success=True, data={"stdout": "OK", "stderr": "", "returncode": 0})

        with patch.object(reviewer._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.reviewer.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
                mock_client_cls.return_value = mock_client

                result = await reviewer.execute("test task", {"executor_result": executor_result, "step_number": 1})

        assert result.success is True
        assert reviewer.model == "deepseek/deepseek-chat:free"

    @pytest.mark.asyncio
    async def test_all_configs_fail_returns_error(self, reviewer):
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

        executor_result = AgentResult(success=True, data={"stdout": "OK", "stderr": "", "returncode": 0})

        with patch.object(reviewer._resolver, "get_execution_queue", return_value=queue):
            with patch("agents.reviewer.AsyncOpenAI") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create = AsyncMock(
                    side_effect=APITimeoutError(request=MagicMock())
                )
                mock_client_cls.return_value = mock_client

                result = await reviewer.execute("test task", {"executor_result": executor_result, "step_number": 1})

        assert result.success is False
        assert "Все конфигурации ревью недоступны" in result.error_message