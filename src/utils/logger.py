import logging
from pathlib import Path


def get_logger(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(path))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(handler)
    return logger
