"""Project logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: str | Path, level: str = "INFO") -> logging.Logger:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fintechathon2026")
    logger.setLevel(level.upper())
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(path / "project.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

