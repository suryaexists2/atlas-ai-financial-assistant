"""Application exception hierarchy.

All domain/application errors derive from `AppError` so the API layer can map
them to responses uniformly, and service code never raises raw `Exception`.
"""

from typing import Any


class AppError(Exception):
    """Base class for all expected application errors."""

    status_code = 500
    code = "app_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class DomainValidationError(AppError):
    status_code = 422
    code = "validation_error"


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthorized"


class ExternalServiceError(AppError):
    status_code = 502
    code = "external_service_error"


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"
