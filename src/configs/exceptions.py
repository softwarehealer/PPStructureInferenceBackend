from http import HTTPStatus
from fastapi import HTTPException
from starlette import status


class InternalException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


class ProcessingException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Post processing failed",
        )


class StreamTimeoutException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Stream timed out",
        )

class ValidationException(Exception):
    """
    A custom exception for exception in data validations.
    """
    def __init__(self, message="A custom data validation error occurred."):
        self.message = message
        super().__init__(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=self.message)

class StaleVersionException(Exception):
    """
    A custom exception for stale version which is old rvn number.
    This should translate to 400 bad data error for fastapi.
    """
    def __init__(self, message="A custom data validation error occurred."):
        self.message = message
        super().__init__(self.message)


class BadRequestException(HTTPException):
    def __init__(self, message="Bad request by user. Please check."):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )

class SecurityException(HTTPException):
    def __init__(self, message="Security exception occured. Access voilation."):
        super().__init__(
            status_code=HTTPStatus.FORBIDDEN,
            detail=message
        )

class ServiceException(HTTPException):
    def __init__(self, message="Internal server caused exception during service request"):
        super().__init__(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=message
        )


class AgentException(Exception):
    """
    A custom exception occured during Agent processing.
    """
    def __init__(self, message="A processing error in the Agent."):
        self.message = message
        super().__init__(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=self.message)