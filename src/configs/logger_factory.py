from enum import Flag
import io
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from configs import settings

LOG_BACKUP_COUNT = 10
#LOG_FILE_LOCATION = "logs/app.log"
LOG_FORMATTER= "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(filename)s:%(lineno)d - %(funcName)s() :: %(message)s"
LOG_LEVEL = settings.LOG_LEVEL
LOG_MAX_BYTES = 100000

os.makedirs("logs", exist_ok=True)

def get_logger_instance(name:str, log_level:str = settings.LOG_LEVEL, add_console_logs:bool = False)-> logging.Logger:
    # common logger for our application
    logger = logging.getLogger(name)
    # Level is set from configuration
    logger.setLevel(log_level)

    # create a format for the logs
    logFormatter = logging.Formatter(LOG_FORMATTER)

    if add_console_logs:
        # Create a stream handler to print logs to the console
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        console_handler = logging.StreamHandler(utf8_stdout) 
        console_handler.setFormatter(logFormatter)
        console_handler.setLevel(log_level)

    # Create a file handler to write logs to a file
    file_handler = RotatingFileHandler(f"logs/{name}.log", maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logFormatter)

    # Add the handlers to the logger
    if add_console_logs:
        logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger