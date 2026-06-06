import logging
import sys


def setup_console_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("lab_utils")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)

    return logger


def log_config(config: dict, logger: logging.Logger | None = None) -> None:
    if logger is None:
        logger = logging.getLogger("lab_utils")
    logger.info("Experiment config:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
