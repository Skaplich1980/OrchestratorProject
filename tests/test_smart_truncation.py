# tests/test_smart_truncation.py
import pytest

# Мы импортируем функцию, которую создадим на ЭТАПЕ 2.
# Пока тест будет падать с ImportError, что нормально для TDD.
try:
    from agents.reviewer import smart_truncate_output
except ImportError:
    smart_truncate_output = None


def test_smart_truncation_keeps_stderr_and_truncates_stdout_tail():
    """
    Проверяет, что функция:
    1. Всегда включает весь stderr, если он не пустой.
    2. Берёт хвост stdout, чтобы общая длина не превышала лимит.
    """
    if smart_truncate_output is None:
        pytest.skip("Функция smart_truncate_output ещё не реализована")

    long_stdout = "A" * 4000
    important_stderr = "CRITICAL ERROR: File not found at line 42"

    result = smart_truncate_output(
        stdout=long_stdout,
        stderr=important_stderr,
        max_length=3000
    )

    # stderr должен быть целиком
    assert important_stderr in result
    # stdout должен быть обрезан, но содержать конец строки "AAA"
    assert result.endswith("AAA") or "AAA" in result.split("CRITICAL ERROR")[0]
    # Общая длина должна быть около лимита (с учётом переносов строк)
    assert len(result) <= 3200


def test_smart_truncation_handles_empty_stderr():
    """Если stderr пустой, функция просто обрезает stdout с конца."""
    if smart_truncate_output is None:
        pytest.skip("Функция smart_truncate_output ещё не реализована")

    long_stdout = "B" * 5000

    result = smart_truncate_output(
        stdout=long_stdout,
        stderr="",
        max_length=3000
    )

    assert "CRITICAL" not in result
    assert len(result) <= 3000
    assert result.endswith("BBB")