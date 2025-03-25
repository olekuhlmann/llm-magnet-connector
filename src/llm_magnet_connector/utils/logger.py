import logging
import sys
import os
import copy
from datetime import datetime


def _sanitize_anthropic_debug(record):
    """
    Sanitizes the debug messages from the Anthropic client by removing image data from the logs.
    A deep copy of the record is made to avoid modifying the original record.
    """
    try:
        if not record.args or not record.msg:
            return record

        if record.msg == "Request options: %s":
            # create a deep copy of the record to avoid modifying the original record
            record_copy = record.__dict__.copy()
            record_copy["args"] = copy.deepcopy(record.args)

            messages = record_copy["args"]["json_data"]["messages"]
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    for content in msg["content"]:
                        if content.get("type") == "image" and "source" in content:
                            content["source"][
                                "data"
                            ] = "[OMITTED: image data removed for logging]"

            return logging.makeLogRecord(record_copy)
        else:
            return record
    except Exception:
        return record


class SanitizingFormatter(logging.Formatter):
    """
    Custom logging formatter that sanitizes Anthropic DEBUG messages by removing image data.
    """

    def format(self, record):
        if record.name == "anthropic._base_client" and record.levelno == logging.DEBUG:
            record_sanitized = _sanitize_anthropic_debug(record)
            return super().format(record_sanitized)
        return super().format(record)


def create_logger(log_dir="logs", console_level=logging.INFO):
    """
    Creates a logger that logs messages to a timestamped file and optionally to the console.
    In Anthropic DEBUG messages, all image blocks are replaced by placeholders.

    Args:
        log_dir (str): Directory to store the log files. Default is 'logs'.
        console_level (int): Logging level for the console. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{timestamp}.log")

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = SanitizingFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler (logs everything)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Levels of logging:
if __name__ == "__main__":
    logger = create_logger()
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
