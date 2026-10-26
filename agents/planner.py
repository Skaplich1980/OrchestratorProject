# agents/planner.py
"""Планировщик через OpenRouter (OpenAI-совместимый API).

Особенности:
  - Динамический поиск бесплатных моделей через get_free_models().
  - Fallback на резервные модели из settings.planner_models.
  - Retry с exponential backoff на 429/5xx/таймаут.
  - Глубокое логирование промтов и сырых ответов.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from openai import (
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)

from agents.base import AgentResult, BaseAgent
from config import Settings
from openrouter_utils import get_free_models

# ВАЖНО: Фигурные скобки для JSON удвоены ({{ и }}), чтобы .format() не пытался их распарсить.
_PLANNER_SYSTEM_PROMPT = """\
You are a precise project planning agent for Python codebases.
Given the goal and project context, produce a strictly valid JSON array of steps.
Each step is an object:
{{
  "step": <int>,
  "description": "<short imperative sentence>",
  "target_files": ["<relative/path>", ...]
}}
Rules:
- Return ONLY a JSON array. No markdown, no code fences, no commentary.
- Each step is atomic and independently verifiable.
- target_files are relative paths from the project root.
- Up to {max_steps} steps.
"""


class QwenPlanner(BaseAgent):
    """Агент-планировщик. Обёртка над OpenRouter (Qwen-модели)."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        super().__init__()
        self.name = "planner"
        self.role = "planner"
        self._settings = settings
        self._logger = logger
        self._models: list[str] = []

        # AsyncOpenAI-клиент для OpenRouter (OpenAI-совместимый API).
        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=120.0,
            max_retries=0,  # управляем retry вручную для логирования
        )

    async def _init_models(self) -> None:
        """Ленивая инициализация списка моделей: free → fallback."""
        if self._models:
            return

        self._logger.info("Поиск бесплатных моделей OpenRouter...")
        free_models = await get_free_models(
            sort_by="context_length",
            limit=5,
            min_context_length=32000,
        )

        if free_models:
            self._models = free_models
            self._logger.info("✅ Используются бесплатные модели: %s", self._models)
        else:
            self._models = self._settings.planner_models
            self._logger.warning(
                "⚠️ Бесплатные модели не найдены. Используем резервные: %s",
                self._models,
            )

    async def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        """Составляет план по цели.

        :param task: Текст цели (содержимое goal.txt).
        :param context: {'project_path': '...'} — для понимания модели, где работать.
        """
        await self._init_models()

        project_path: Optional[str] = context.get("project_path")
        max_steps: int = self._settings.max_steps

        system_prompt = _PLANNER_SYSTEM_PROMPT.format(max_steps=max_steps)
        user_prompt = (
            f"Project path: {project_path}\n"
            f"Goal:\n{task}\n"
            f"Return the plan as a JSON array (up to {max_steps} steps)."
        )

        self._logger.info(
            "Начало планирования. Цель: '%s'",
            task[:80].replace("\n", " "),
        )
        self._logger.debug(
            "--- ПРОМТ ПЛАНИРОВЩИКА ---\nSYSTEM:\n%s\nUSER:\n%s",
            system_prompt,
            user_prompt,
        )

        # --- RETRY LOOP: перебираем модели по приоритету ----------------
        last_error = "Неизвестная ошибка"

        for model in self._models:
            self.model = model  # обновляем для логов
            self._logger.info(">>> Попытка использования модели: %s", model)

            for attempt in range(1, self._settings.max_retries + 1):
                try:
                    response = await self._client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=2000,
                    )

                    raw_text = response.choices[0].message.content or ""
                    if not raw_text.strip():
                        raise ValueError("Модель вернула пустой ответ")

                    self._logger.debug(
                        "--- СЫРОЙ ОТВЕТ МОДЕЛИ (%s) ---\n%s",
                        model,
                        raw_text[:500],
                    )

                    plan = self._parse_json_array(raw_text)
                    tokens_used = self._extract_tokens(response)

                    self._logger.info(
                        "✓ Модель %s успешно сформировала план: %d шагов",
                        model,
                        len(plan),
                    )
                    return AgentResult(success=True, data=plan, tokens_used=tokens_used)

                except RateLimitError as e:
                    last_error = f"Rate limit: {e}"
                    self._logger.warning(
                        "Попытка %d/%d для %s: %s",
                        attempt, self._settings.max_retries, model, last_error,
                    )
                except (APITimeoutError, APIConnectionError) as e:
                    last_error = f"Network/timeout: {e}"
                    self._logger.warning(
                        "Попытка %d/%d для %s: %s",
                        attempt, self._settings.max_retries, model, last_error,
                    )
                except APIError as e:
                    last_error = f"API error: {e}"
                    # Если модель не найдена (404), нет смысла делать retry на этой же модели.
                    if "model_not_found" in str(e).lower() or "404" in str(e):
                        self._logger.error(
                            "Модель %s не найдена. Переход к следующей.", model,
                        )
                        break
                    self._logger.warning(
                        "Попытка %d/%d для %s: %s",
                        attempt, self._settings.max_retries, model, last_error,
                    )
                except (json.JSONDecodeError, ValueError) as e:
                    last_error = f"JSON parse: {e}"
                    self._logger.warning(
                        "Попытка %d/%d для %s: %s",
                        attempt, self._settings.max_retries, model, last_error,
                    )

                # Пауза перед следующей попыткой (exponential backoff).
                if attempt < self._settings.max_retries:
                    delay = self._settings.retry_delay * (
                        self._settings.retry_backoff ** (attempt - 1)
                    )
                    self._logger.info(
                        "Ожидание %.1f сек. перед повторной попыткой...", delay,
                    )
                    await asyncio.sleep(delay)

            self._logger.warning(
                "Модель %s исчерпала попытки. Переход к следующей в списке...", model,
            )

        # Все модели из списка провалились.
        error_msg = (
            f"Все модели планирования недоступны. Последняя ошибка: {last_error}"
        )
        self._logger.error(error_msg)
        return AgentResult(success=False, data=None, error_message=error_msg)

    @staticmethod
    def _parse_json_array(text: str) -> list[dict[str, Any]]:
        """Вырезает JSON-массив из ответа модели (защита от markdown-обёрток)."""
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError("JSON array not found in response")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, list):
            raise ValueError("Response is not a JSON array")
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"Step #{i} is not a dict")
        return parsed

    @staticmethod
    def _extract_tokens(response: Any) -> Optional[dict[str, int]]:
        """Достаёт usage из OpenAI-совместимого ответа."""
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            return {
                "input": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        except (AttributeError, TypeError, ValueError):
            return None