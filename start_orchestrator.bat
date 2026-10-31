@echo off
echo ============================================================
echo   Запуск Оркестратора
call .venv\Scripts\activate
python orchestrator.py
echo ============================================================
echo   Завершено
pause