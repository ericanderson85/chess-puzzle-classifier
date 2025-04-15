import logging
from logging import Logger


def get_logger(name: str, path: str) -> Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(name)
    return logger
