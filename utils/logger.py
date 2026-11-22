"""Project-wide logger. Call get_logger(__name__) in every module."""
import logging
import sys
import io
import os
from configs.config import OUT_LOGS

os.makedirs(OUT_LOGS, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console — force UTF-8 so Unicode arrows/symbols work on Windows cp1252 terminals
    utf8_stream = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    ) if hasattr(sys.stdout, "buffer") else sys.stdout
    ch = logging.StreamHandler(utf8_stream)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File — always UTF-8
    fh = logging.FileHandler(os.path.join(OUT_LOGS, "project.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
