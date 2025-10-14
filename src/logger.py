import logging
from pathlib import Path

def setup_logger(name, level = logging.INFO, file_out : Path | None = None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch.setFormatter(formatter)

    logger.addHandler(ch) 

    if file_out:
        file_handler = logging.FileHandler(file_out, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
