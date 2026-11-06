# agents/planner.py
"""Планировщик через мульти-роутерный API.
Особенности:
- Использование ModelResolver для получения приоритетной очереди моделей.
- Плоский цикл retry: при ошибке мгновенное переключение на следующий роутер/модель.
- Глубокое логирование промтов и сырых ответов.
- Промты собираются через PromptManager (без хардкода в .py).
"""
from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError

from agents.base import AgentResult, BaseAgent
from config import Settings
from model_resolver import ModelResolver

if TYPE_CHECKING:
    from prompt_manager import PromptManager


class QwenPlanner(BaseAgent):
    """Агент-планировщик."""

    def __init__(
            self,
            settings: Settings,
            logger: logging.Logger,
            prompt_manager: "PromptManager",
    ) -> None:
        super().__init__()
        self.name = "planner"
        self.role = "planner"
        self.model = "dynamic"
        self._settings = settings
        self._logger = logger
        self._prompt_manager = prompt_manager
        self._resolver = ModelResolver(settings)

    async def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        """Составляет план по цели."""
        project_path: Optional[str] = context.get("project_path")
        max_steps: int = self._settings.max_steps

        system_prompt = self._prompt_manager.build_planner_system_prompt(max_steps=max_steps)
        user_prompt = self._prompt_manager.build_planner_user_prompt(
            project_path=project_path,
            task=task,
            max_steps=max_steps,
        )

        self._logger.info("Начало планирования. Цель: '%s'", task[:80].replace("\n", " "))
        self._logger.info("--- ФИНАЛЬНЫЙ SYSTEM PROMPT ПЛАНИРОВЩИКА ---\n%s", system_prompt)
        self._logger.debug("--- USER PROMPT ПЛАНИРОВЩИКА ---\n%s", user_prompt)

        execution_queue = self._resolver.get_execution_queue()
        if not execution_queue:
            error_msg = "Не удалось сформировать очередь моделей. Проверьте .env и доступ к роутерам."
            self._logger.error(error_msg)
            return AgentResult(success=False, data=None, error_message=error_msg)

        last_error = "Неизвестная ошибка"

        for config in execution_queue:
            self.model = config["model_id"]
            self._logger.info(
                ">>> Попытка использования: %s (роутер: %s)",
                self.model, config["router"]
            )

            default_headers = {}
            if config["router"] == "openrouter":
                default_headers = {
                    "HTTP-Referer": "https://github.com/Skaplich1980/OrchestratorProject",
                    "X-Title": "Python Multi-Agent Orchestrator",
                }

            client = AsyncOpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
                timeout=120.0,
                max_retries=0,  # Управляем retry вручную для мгновенного переключения
            )

            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=10000,
                )
                raw_text = response.choices[0].message.content or ""
                if not raw_text.strip():
                    raise ValueError("Модель вернула пустой ответ")

                self._logger.debug("--- СЫРОЙ ОТВЕТ МОДЕЛИ (%s) ---\n%s", self.model, raw_text[:2500])

                plan = self._parse_json_array(raw_text)
                tokens_used = self._extract_tokens(response)

                self._logger.info("✓ Модель %s успешно сформировала план: %d шагов", self.model, len(plan))
                return AgentResult(success=True, data=plan, tokens_used=tokens_used)

            except RateLimitError as e:
                last_error = f"Rate limit: {e}"
                self._logger.warning("Для %s: %s. Переход к следующей конфигурации.", self.model, last_error)
            except (APITimeoutError, APIConnectionError) as e:
                last_error = f"Network/timeout: {e}"
                self._logger.warning("Для %s: %s. Переход к следующей конфигурации.", self.model, last_error)
            except APIError as e:
                last_error = f"API error: {e}"
                if "model_not_found" in str(e).lower() or "404" in str(e):
                    self._logger.error("Модель %s не найдена. Переход к следующей.", self.model)
                else:
                    self._logger.warning("Для %s: %s. Переход к следующей конфигурации.", self.model, last_error)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = f"JSON parse: {e}"
                self._logger.warning("Для %s: %s. Переход к следующей конфигурации.", self.model, last_error)

            # Конфигурация исчерпана — переход к следующей в очереди
            self._logger.warning("Конфигурация %s исчерпана. Переход к следующей в очереди...", self.model)

        error_msg = f"Все конфигурации планирования недоступны. Последняя ошибка: {last_error}"
        self._logger.error(error_msg)
        return AgentResult(success=False, data=None, error_message=error_msg)

    @staticmethod
    def _parse_json_array(text: str) -> list[dict[str, Any]]:
        """Вырезает JSON-массив из ответа модели (защита от markdown-обёрток)."""
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            raise ValueError("JSON array not found in response")
        parsed = json.loads(text[start: end + 1])
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