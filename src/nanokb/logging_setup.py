"""stdlib logging 配置（方案 §3.6）。

格式：``%(asctime)s | %(levelname)s | %(stage)s | %(file)s | %(message)s``，
其中 ``stage``/``file`` 为业务上下文字段，调用方通过 ``extra={"stage": ..., "file": ...}``
注入；未注入时显示 ``-``。
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "nanokb"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(stage)s | %(file)s | %(message)s"


class _ContextFilter(logging.Filter):
    """为 LogRecord 注入默认的 stage / file 上下文字段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "stage"):
            record.stage = "-"
        if not hasattr(record, "file"):
            record.file = "-"
        return True


def get_logger() -> logging.Logger:
    """获取 nanokb 命名空间的 logger。"""
    return logging.getLogger(LOGGER_NAME)


def setup_logging(
    out_dir: Path | None = None,
    *,
    verbose: bool = False,
) -> logging.Logger:
    """配置 logging：控制台 + 可选 ``<out_dir>/build.log`` 追加写。

    重复调用安全（会清空既有 handler 避免重复输出）。
    """
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(LOG_FORMAT)
    context_filter = _ContextFilter()

    logger = get_logger()
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    logger.addHandler(console_handler)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out_dir / "build.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)

    return logger


__all__ = ["LOG_FORMAT", "LOGGER_NAME", "get_logger", "setup_logging"]
