# agents/base.py
"""Контракт агентов оркестратора."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentResult:
    """Единый результат работы любого агента.

    :param success: Агент отработал корректно (не сеть/парсинг).
    :param data: Полезная нагрузка (план/stdout/вердикт).
    :param error_message: Описание сбоя, если success=False.
    :param tokens_used: {'input': N, 'output': M} — для биллинга.
    """
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    tokens_used: Optional[dict[str, int]] = field(default=None)


class BaseAgent(ABC):
    """Базовый класс: все агенты реализуют async execute(task, context)."""

    name: str = "base"
    role: str = "base"
    model: str = "unknown"

    @abstractmethod
    async def execute(self, task: str, context: dict[str, Any]) -> AgentResult:
        raise NotImplementedError