# =========================================================
# LLM РОУТЕРЫ (Только бесплатные модели)
# =========================================================
# Ключевые слова для умного поиска моделей (по приоритету)
PRIORITY_KEYWORDS=qwen3, deepseek-v3, qwen-2.5-72b, deepseek-r1

# OrcaRouter (Приоритет №1)
ORCAROUTER_API_KEY=sk-orca-ВАШ_КЛЮЧ
ORCAROUTER_BASE_URL=https://api.orcarouter.ai/v1

# OpenRouter (Приоритет №2, Fallback)
OPENROUTER_API_KEY=sk-or-v1-ВАШ_КЛЮЧ
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# =========================================================
# AGNES CLI
# =========================================================
AGNES_API_KEY=sk-ВАШ_КЛЮЧ_AGNES
AGNES_CLI_PATH=agnes  # Или полный путь, например: C:\Program Files\Agnes\agnes.exe
AGNES_MODEL=agnes/agnes-2.5-flash
AGNES_TIMEOUT=300

# =========================================================
# ПРОЕКТ И ЛИМИТЫ
# =========================================================
DEFAULT_PROJECT_PATH=D:\Project\garden_helper
LOG_LEVEL=INFO          # Используйте DEBUG для просмотра полных промтов
MAX_STEPS=10
MAX_RETRIES=2
RETRY_DELAY=5.0
RETRY_BACKOFF=2.0
DELAY_BETWEEN_CALLS=5.0
ENABLE_BACKUPS=true
BACKUP_DIR=./backups
MAX_REPLAN_ATTEMPTS=2