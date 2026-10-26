# backup_manager.py
from __future__ import annotations
import logging
import shutil
from pathlib import Path


def create_backup(
        project_path: Path,
        target_files: list[str],
        backup_base_dir: Path,
        run_id: str,
        logger: logging.Logger
) -> Path | None:
    """
    Создаёт резервную копию целевых файлов перед их изменением.
    Возвращает путь к папке бэкапа или None, если бэкап не требуется.
    """
    if not target_files:
        logger.debug("Целевые файлы не указаны, бэкап пропускается.")
        return None

    backup_dir = backup_base_dir / f"run_{run_id}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up_count = 0

    for rel_path_str in target_files:
        rel_path = Path(rel_path_str)
        # Защита от выхода за пределы проекта (path traversal)
        try:
            full_path = (project_path / rel_path).resolve()
            full_path.relative_to(project_path.resolve())
        except ValueError:
            logger.warning("Попытка доступа к файлу вне проекта: %s", rel_path_str)
            continue

        if full_path.exists() and full_path.is_file():
            dest_path = backup_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(full_path, dest_path)
                backed_up_count += 1
                logger.debug("Бэкап создан: %s -> %s", full_path, dest_path)
            except Exception as e:
                logger.error("Ошибка копирования файла %s для бэкапа: %s", full_path, e)
        else:
            logger.debug("Файл не существует, бэкап не создан: %s", full_path)

    logger.info("Создан бэкап для %d файлов в папке: %s", backed_up_count, backup_dir)
    return backup_dir if backed_up_count > 0 else None