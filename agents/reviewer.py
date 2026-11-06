# agents/reviewer.py
"""Ревьюер через мульти-роутерный API. Оценивает, успешно ли исполнитель выполнил шаг.
Использует ту же архитектуру и очередь моделей, что и planner, для максимальной согласованности.
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


# Fallback на случай, если prompt_manager не передан (старые вызовы, тесты).
_FALLBACK_REVIEWER_SYSTEM_PROMPT = """\
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


def smart_truncate_output(stdout: str, stderr: str, max_length: int = 4000) -> str:
    """Умно усекает вывод: сохраняет начало и конец stderr (где Traceback и финальная ошибка)."""
    if not stdout and not stderr:
        return ""

    if stderr:
        if len(stderr) >= max_length:
            # Берём первые 500 символов (Traceback) + последние 1000 символов (финальная ошибка)
            head = stderr[:500]
            tail = stderr[-1000:]
            skipped_lines = stderr.count("\n") - head.count("\n") - tail.count("\n")
            return f"{head}\n...[пропущено {skipped_lines} строк]...\n{tail}"

        available_for_stdout = max_length - len(stderr) - 15
        if available_for_stdout > 0 and stdout:
            truncated = stdout[-available_for_stdout:]
            if len(stdout) > available_for_stdout:
                return f"...{truncated}\n---ERR---\n{stderr}"
            return f"{stdout}\n---ERR---\n{stderr}"
        return stderr

    if len(stdout) > max_length:
        return "..." + stdout[-(max_length - 3):]
    return stdout


class QwenReviewer(BaseAgent):
    """Агент-ревьюер."""

    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        prompt_manager: Optional["PromptManager"] = None,
    ) -> None:
        super().__init__()
        self.name = "reviewer"
        self.role = "reviewer"
        self.model = "dynamic"
        self._settings = settings
        self._logger = logger
        self._prompt_manager = prompt_manager
        self._resolver = ModelResolver(settings)

    async def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        executor_result: Optional[AgentResult] = context.get("executor_result")
        step_number: int = int(context.get("step_number", 0))

        exec_data = executor_result.data if executor_result else None
        exec_error = executor_result.error_message if executor_result else "no executor result"

        user_prompt = (
            f"Task:\n{task}\n"
            f"Execution result:\n{self._format_executor_output(exec_data, exec_error)}\n"
            f'Return JSON: {{"success": true/false, "reason": "..."}}'
        )

        self._logger.info("Шаг #%d: запрос к ревьюеру", step_number)
        self._logger.debug("--- ПРОМТ РЕВЬЮЕРА ---\n%s", user_prompt)

        # Системный промт из PromptManager (единая точка), иначе — fallback
        if self._prompt_manager is not None:
            system_prompt = self._prompt_manager.build_reviewer_system_prompt()
        else:
            system_prompt = _FALLBACK_REVIEWER_SYSTEM_PROMPT

        execution_queue = self._resolver.get_execution_queue()
        if not execution_queue:
            error_msg = "Не удалось сформировать очередь моделей для ревьюера."
            self._logger.error(error_msg)
            return AgentResult(success=False, data={"success": False, "reason": error_msg}, error_message=error_msg)

        last_error = "Неизвестная ошибка"

        for config in execution_queue:
            self.model = config["model_id"]
            self._logger.info(
                ">>> Ревью шага #%d. Попытка использования: %s (роутер: %s)",
                step_number, self.model, config["router"]
            )

            # Заголовки, обязательные для OpenRouter (иначе 403 на free-моделях)
            default_headers: dict[str, str] = {}
            if config["router"] == "openrouter":
                default_headers = {
                    "HTTP-Referer": "https://github.com/your-orchestrator",
                    "X-Title": "Python Multi-Agent Orchestrator",
                }

            client = AsyncOpenAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
                timeout=90.0,
                max_retries=0,
            )

            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
                raw_text = response.choices[0].message.content or ""
                if not raw_text.strip():
                    raise ValueError("Пустой ответ ревьюера")

                self._logger.debug("--- СЫРОЙ ОТВЕТ РЕВЬЮЕРА (%s) ---\n%s", self.model, raw_text)

                review = self._parse_review_json(raw_text)
                tokens_used = self._extract_tokens(response)
                is_success = bool(review.get("success", False))
                reason = str(review.get("reason", ""))

                self._logger.info(
                    "✓ Ревьюер (%s): success=%s, reason: %s",
                    self.model, is_success, reason[:120],
                )
                return AgentResult(success=True, data=review, tokens_used=tokens_used)

            except RateLimitError as e:
                last_error = f"Rate limit: {e}"
                self._logger.warning("Для %s: %s. Переход к следующей конфигурации.", self.model, last_error)
            except (APITimeoutError, APIConnectionError) as e:
                last_error = f"Network: {e}"
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

            self._logger.warning("Конфигурация %s исчерпана. Переход к следующей в очереди...", self.model)

        error_msg = f"Все конфигурации ревью недоступны. Последняя ошибка: {last_error}"
        self._logger.error(error_msg)
        return AgentResult(success=False, data={"success": False, "reason": error_msg}, error_message=error_msg)

    def _format_executor_output(self, data: Optional[dict[str, Any]], error_message: Optional[str]) -> str:
        """Форматирует данные от исполнителя, используя умное усечение."""
        if data is None and error_message:
            return f"ERROR: {error_message}"
        if data is None:
            return "(no execution data)"

        stdout = str(data.get("stdout", "")).strip()
        stderr = str(data.get("stderr", "")).strip()
        returncode = data.get("returncode")

        truncated_output = smart_truncate_output(stdout, stderr, max_length=self._settings.reviewer_max_output_length)
        parts = [truncated_output] if truncated_output else []
        parts.append(f"returncode: {returncode}")
        return "\n".join(parts)

    @staticmethod
    def _parse_review_json(text: str) -> dict[str, Any]:
        """Извлекает JSON-объект из ответа модели."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("JSON object not found")
        parsed = json.loads(text[start: end + 1])
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