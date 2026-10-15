from enum import Flag
from configs.settings import Settings

settings = Settings()

from configs.logger_factory import get_logger_instance
# various loggers for the application. Can change the access levels here, currently defaulted to settings.LOG_LEVEL
app_logger = get_logger_instance(name="app", log_level=settings.LOG_LEVEL, add_console_logs=True)

class LogErrLevel(Flag):
    STACK=False
    EXC=True