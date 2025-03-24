import logging
import sys
import os
from datetime import datetime

def create_logger(log_dir='logs', console_level=logging.INFO):
    """
    Creates a logger that logs messages to a timestamped file and optionally to the console.

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
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = os.path.join(log_dir, f'{timestamp}.log')

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
if __name__ == '__main__':
    logger = create_logger()
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.critical('Critical message')
