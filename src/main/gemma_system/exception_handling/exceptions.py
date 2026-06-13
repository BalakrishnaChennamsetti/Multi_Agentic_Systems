class AppError(Exception):
    """Base exception for the application."""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class VectorDBError(AppError):
    """Vector DB related errors."""


class DocumentLoadError(AppError):
    """Document loading errors."""


class LLMError(AppError):
    """LLM related errors."""
