"""
Custom exceptions for the application.
Provides specific error types for different components.
"""

from typing import Optional


class RAGException(Exception):
    """Base exception for RAG pipeline errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DocumentNotFoundError(RAGException):
    """Raised when a document is not found in the vector database."""
    pass


class DocumentProcessingError(RAGException):
    """Raised when document processing fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class LLMError(RAGException):
    """Raised when LLM request fails."""
    pass


class VectorDBError(RAGException):
    """Raised when vector database operations fail."""
    pass


class InvalidFileTypeError(RAGException):
    """Raised when an unsupported file type is uploaded."""
    pass


class FileSizeError(RAGException):
    """Raised when file size exceeds maximum allowed size."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid or missing."""
    pass


class RateLimitError(RAGException):
    """Raised when rate limit is exceeded."""
    pass

