import logging
import sys

from loguru import logger

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """Route standard-library logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(exception=record.exc_info).patch(
            lambda r: r.update(
                name=record.name,
                function=record.funcName,
                line=record.lineno,
                file=record.pathname,
                module=record.module,
            )
        ).log(level, record.getMessage())


def setup_logging() -> None:
    """Make loguru the single sink for both application and uvicorn logs."""
    logger.remove()
    logger.add(sys.stderr, format=_LOG_FORMAT, level="INFO")

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).propagate = True
