# agents/reviewer.py
"""Ревьюер через OpenRouter. Оценивает, успешно ли исполнитель выполнил шаг.

Использует ту же архитектуру, что и planner: динамический выбор модели,
fallback, retry, глубокое логирование.
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

_REVIEWER_SYSTEM_PROMPT = """\
You are a strict code review agent.
Given a task and the execution result (stdout/stderr/returncode), determine
whether the task was completed successfully.
Return ONLY a JSON object:
{"success": true/false, "reason": "<short explanation>"}
Rules:
- Be strict: errors, missing changes, incomplete work → success=false.
- reason must be a single sentence.
- No markdown, no commentary, no extra keys.
"""


class QwenReviewer(BaseAgent):
    """Агент-ревьюер (OpenRouter)."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        super().__init__()
        self.name = "reviewer"
        self.role = "reviewer"
        self._settings = settings
        self._logger = logger
        self._models: list[str] = []

        self._client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=90.0,
            max_retries=0,  # свой retry
        )

    async def _init_models(self) -> None:
        """Ленивая инициализация списка моделей: free → fallback."""
        if self._models:
            return

        self._logger.info("Поиск бесплатных моделей OpenRouter для ревьюера...")
        free_models = await get_free_models(
            sort_by="context_length",
            limit=3,
            min_context_length=16000,  # для ревьюера контекст может быть меньше
        )

        if free_models:
            self._models = free_models
            self._logger.info("✅ Ревьюер: бесплатные модели: %s", self._models)
        else:
            self._models = self._settings.reviewer_models
            self._logger.warning(
                "⚠️ Ревьюер: бесплатные модели не найдены. Используем резервные: %s",
                self._models,
            )

    async def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        executor_result: Optional[AgentResult] = context.get("executor_result")
        step_number: int = int(context.get("step_number", 0))

        await self._init_models()

        exec_data = executor_result.data if executor_result else None
        exec_error = executor_result.error_message if executor_result else "no executor result"

        user_prompt = (
            f"Task:\n{task}\n"
            f"Execution result:\n{self._format_executor_output(exec_data, exec_error)}\n"
            f'Return JSON: {{"success": true/false, "reason": "..."}}'
        )

        self._logger.info("Шаг #%d: запрос к ревьюеру", step_number)
        self._logger.debug("--- ПРОМТ РЕВЬЮЕРА ---\n%s", user_prompt)

        last_error = "Неизвестная ошибка"

        for model in self._models:
            self.model = model
            self._logger.info(">>> Ревью шага #%d. Попытка модели: %s", step_number, model)

            for attempt in range(1, self._settings.max_retries + 1):
                try:
                    response = await self._client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": _REVIEWER_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=500,
                    )

                    raw_text = response.choices[0].message.content or ""
                    if not raw_text.strip():
                        raise ValueError("Пустой ответ ревьюера")

                    self._logger.debug(
                        "--- СЫРОЙ ОТВЕТ РЕВЬЮЕРА (%s) ---\n%s",
                        model,
                        raw_text,
                    )

                    review = self._parse_review_json(raw_text)
                    tokens_used = self._extract_tokens(response)

                    is_success = bool(review.get("success", False))
                    reason = str(review.get("reason", ""))
                    self._logger.info(
                        "✓ Ревьюер (%s): success=%s, reason: %s",
                        model,
                        is_success,
                        reason[:120],
                    )
                    # ВАЖНО: success=True означает «ревьюер отработал».
                    # Сам вердикт лежит в data.success — его читает оркестратор.
                    return AgentResult(success=True, data=review, tokens_used=tokens_used)

                except RateLimitError as e:
                    last_error = f"Rate limit: {e}"
                    self._logger.warning(
                        "Попытка %d/%d для %s: %s",
                        attempt, self._settings.max_retries, model, last_error,
                    )
                except (APITimeoutError, APIConnectionError) as e:
                    last_error = f"Network: {e}"
                    self._logger.warning(
                        "Попытка %d/%d для %s: %s",
                        attempt, self._settings.max_retries, model, last_error,
                    )
                except APIError as e:
                    last_error = f"API error: {e}"
                    if "model_not_found" in str(e).lower() or "404" in str(e):
                        self._logger.error("Модель %s не найдена. Переход к следующей.", model)
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

                if attempt < self._settings.max_retries:
                    delay = self._settings.retry_delay * (
                        self._settings.retry_backoff ** (attempt - 1)
                    )
                    self._logger.info("Пауза %.1f сек.", delay)
                    await asyncio.sleep(delay)

            self._logger.warning(
                "Модель %s исчерпала попытки. Переход к следующей...", model,
            )

        error_msg = (
            f"Все модели ревью недоступны. Последняя ошибка: {last_error}"
        )
        self._logger.error(error_msg)
        # При сбое ревьюера считаем шаг НЕподтверждённым — безопаснее.
        return AgentResult(
            success=False,
            data={"success": False, "reason": error_msg},
            error_message=error_msg,
        )

    @staticmethod
    def _format_executor_output(
        data: Optional[dict[str, Any]],
        error_message: Optional[str],
    ) -> str:
        """Форматирует данные от исполнителя в компактный текст для промта."""
        if data is None and error_message:
            return f"ERROR: {error_message}"
        if data is None:
            return "(no execution data)"
        parts: list[str] = []
        stdout = str(data.get("stdout", "")).strip()[:3000]
        stderr = str(data.get("stderr", "")).strip()[:1500]
        if stdout:
            parts.append(f"stdout:\n{stdout}")
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        parts.append(f"returncode: {data.get('returncode')}")
        return "\n".join(parts)

    @staticmethod
    def _parse_review_json(text: str) -> dict[str, Any]:
        """Извлекает JSON-объект из ответа модели."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("JSON object not found")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict) or "success" not in parsed:
            raise ValueError('Response must be dict with "success" key')
        return parsed

    @staticmethod
    def _extract_tokens(response: Any) -> Optional[dict[str, int]]:
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