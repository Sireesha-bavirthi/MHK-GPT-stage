"""
Core configuration management using Pydantic Settings.
Loads configuration from environment variables with validation.
"""

from functools import lru_cache
from typing import List, Optional
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # =============================================================================
    # API Configuration
    # =============================================================================
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_ENV: str = Field(default="development", description="Environment: development, production")
    API_RELOAD: bool = Field(default=True, description="Auto-reload on code changes")
    
    # =============================================================================
    # Company Settings
    # =============================================================================
    COMPANY_NAME: str = Field(default="MHK Tech Inc", description="Company name for chatbot context")
    
    # =============================================================================
    # LLM Configuration
    # =============================================================================

    

    # Company settings
    COMPANY_NAME: str = Field(default="MHKTech", description="Company name for prompts")
    MAX_CONVERSATION_HISTORY: int = Field(default=10, description="Max conversation messages to keep")
    LLM_PROVIDER: str = "openai"
    # OpenAI settings
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key from .env file")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="OpenAI chat model")
    OPENAI_TEMPERATURE: float = Field(default=0.7, description="OpenAI temperature")
    OPENAI_MAX_TOKENS: int = Field(default=1024, description="Max tokens for OpenAI response")
    
    # =============================================================================
    # Embeddings Configuration
    # =============================================================================
    EMBEDDING_PROVIDER: str = Field(default="sentence-transformers", description="Embedding provider")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    
    # =============================================================================
    # Qdrant Vector Database
    # =============================================================================
    # Cloud configuration (takes precedence if provided)
    QDRANT_URL: Optional[str] = Field(default=None, description="Qdrant Cloud URL (e.g., https://xxx.cloud.qdrant.io:6333)")
    QDRANT_API_KEY: Optional[str] = Field(default=None, description="Qdrant Cloud API key")
    
    # Local configuration (fallback for Docker/local deployment)
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant host (for local deployment)")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant port (for local deployment)")
    
    # Common configuration
    QDRANT_COLLECTION_NAME: str = Field(default="company_docs", description="Qdrant collection name")
    QDRANT_VECTOR_SIZE: int = Field(default=1536, description="Vector dimension size (1536 for text-embedding-3-small)")
    
    # =============================================================================
    # Document Processing
    # =============================================================================
    CHUNK_SIZE: int = Field(default=1000, description="Text chunk size in characters")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between chunks")
    CHUNKING_SEPARATORS: List[str] = Field(
        default=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        description="Separators for recursive chunking"
    )

    # Semantic Chunking Configuration
    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(default=0.75, description="Threshold for semantic similarity")
    SEMANTIC_MIN_CHUNK_SIZE: int = Field(default=100, description="Minimum chunk size for semantic chunking")
    SEMANTIC_MAX_CHUNK_SIZE: int = Field(default=2000, description="Maximum chunk size for semantic chunking")

    SUPPORTED_FILE_TYPES: str = Field(default="pdf,docx,md,txt", description="Comma-separated supported file types")
    MAX_FILE_SIZE_MB: int = Field(default=50, description="Maximum file size in MB")

    # Document paths (relative to project root)
    RAW_DOCUMENTS_PATH: str = Field(default="data/documents/raw", description="Path to raw documents")
    PROCESSED_DOCUMENTS_PATH: str = Field(default="data/documents/processed", description="Path to processed documents")

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        # Go up from backend/app/core to project root
        return Path(__file__).parent.parent.parent.parent

    @property
    def raw_documents_path_absolute(self) -> str:
        """Get absolute path to raw documents."""
        return str(self.project_root / self.RAW_DOCUMENTS_PATH)

    @property
    def processed_documents_path_absolute(self) -> str:
        """Get absolute path to processed documents."""
        return str(self.project_root / self.PROCESSED_DOCUMENTS_PATH)
    
   # =============================================================================
    # Retrieval Configuration
    # =============================================================================
    RETRIEVAL_STRATEGY: str = Field(default="hybrid", description="Retrieval strategy: basic, mmr, or hybrid")
    RETRIEVAL_TOP_K: int = Field(default=10, description="Number of documents to retrieve")  # Increased to ensure founding chunk is included
    RETRIEVAL_FETCH_K: int = Field(default=20, description="Number of candidates for MMR")
    RETRIEVAL_LAMBDA_MULT: float = Field(default=0.7, description="MMR lambda (relevance vs diversity)")
    
    # Hybrid Search Weights
    HYBRID_KEYWORD_WEIGHT: float = Field(default=0.3, description="Weight for keyword search in hybrid mode")
    HYBRID_SEMANTIC_WEIGHT: float = Field(default=0.7, description="Weight for semantic search in hybrid mode")
    # =============================================================================
    # Memory Configuration
    # =============================================================================
    CONVERSATION_MEMORY_TYPE: str = Field(default="buffer_window", description="Memory type")
    CONVERSATION_MEMORY_K: int = Field(default=10, description="Number of messages to remember")
    
    # =============================================================================
    # Security
    # =============================================================================
    SECRET_KEY: str = Field(default="change-me-in-production", description="Secret key for security")
    CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8000", description="CORS origins")
    ALLOWED_HOSTS: str = Field(default="*", description="Allowed hosts")
    
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_CHAT: str = Field(default="20/minute", description="Chat endpoint rate limit")
    RATE_LIMIT_UPLOAD: str = Field(default="5/minute", description="Upload endpoint rate limit")

    # =============================================================================
    # Agent Configuration
    # =============================================================================
    INTENT_CONFIDENCE_THRESHOLD: float = Field(default=0.85, description="Min confidence to skip clarification")
    MAX_CLARIFICATION_TURNS: int = Field(default=2, description="Max clarification loops before fallback")

    # Resilience
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, description="Failures before circuit opens")
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=30, description="Seconds before HALF_OPEN probe")
    MAX_RETRIES: int = Field(default=3, description="Max retry attempts on transient errors")

    # =============================================================================
    # Redis
    # =============================================================================
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")

    # =============================================================================
    # Cache
    # =============================================================================
    JOB_CACHE_TTL_SECONDS: int = Field(default=3600, description="JobDiva results cache TTL")
    EMBEDDING_CACHE_SIZE: int = Field(default=100, description="Embedding LRU cache size")

    # =============================================================================
    # JobDiva
    # =============================================================================
    JOBDIVA_CLIENT_ID: str = Field(default="2156", description="JobDiva client ID")
    JOBDIVA_USERNAME: str = Field(default="", description="JobDiva API username")
    JOBDIVA_PASSWORD: str = Field(default="", description="JobDiva API password")
    JOBDIVA_BASE_URL: str = Field(default="https://api.jobdiva.com/apiv2", description="JobDiva API base URL")

    # =============================================================================
    # Email (SendGrid primary, SMTP fallback)
    # =============================================================================
    SENDGRID_API_KEY: str = Field(default="", description="SendGrid API key (primary email provider)")
    SENDGRID_FROM_EMAIL: str = Field(default="backuptata2005@gmail.com", description="SendGrid from email")

    SMTP_HOST: str = Field(default="smtp.gmail.com", description="SMTP host for fallback email")
    SMTP_PORT: int = Field(default=587, description="SMTP port")
    SMTP_USERNAME: str = Field(default="backuptata2005@gmail.com", description="SMTP username")
    SMTP_PASSWORD: str = Field(default="", description="SMTP password / Gmail App Password")
    SMTP_FROM_EMAIL: str = Field(default="backuptata2005@gmail.com", description="SMTP from email")

    # =============================================================================
    # Google Calendar
    # =============================================================================
    GOOGLE_CALENDAR_ID: str = Field(default="", description="Google Calendar ID for scheduling")
    GOOGLE_SERVICE_ACCOUNT_JSON: str = Field(default="", description="Path to Google service account JSON")

    # =============================================================================
    # Calendly Links (one per meeting duration)
    # =============================================================================
    CALENDLY_15MIN_URL: str = Field(
        default="https://calendly.com/n200029-rguktn/sample-meet",
        description="Calendly URL for quick calls (15 min)"
    )
    CALENDLY_30MIN_URL: str = Field(
        default="https://calendly.com/n200029-rguktn/sample-meet",
        description="Calendly URL for normal meetings (30 min)"
    )
    CALENDLY_60MIN_URL: str = Field(
        default="https://calendly.com/n200029-rguktn/sample-meet",
        description="Calendly URL for long discussions (60 min)"
    )

    # =============================================================================
    # Email Validation
    # =============================================================================
    ABSTRACT_EMAIL_API_KEY: str = Field(
        default="",
        description="Abstract API key for email deliverability check (emailvalidation.abstractapi.com)"
    )
    
    # =============================================================================
    # Logging
    # =============================================================================
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    LOG_FILE: str = Field(default="data/logs/backend/app.log", description="Log file path")
    LOG_ROTATION: str = Field(default="10 MB", description="Log rotation size")
    LOG_RETENTION: str = Field(default="30 days", description="Log retention period")
    
    # =============================================================================
    # Performance
    # =============================================================================
    ENABLE_METRICS: bool = Field(default=False, description="Enable Prometheus metrics")
    PROMETHEUS_PORT: int = Field(default=9090, description="Prometheus port")
    ENABLE_EMBEDDING_CACHE: bool = Field(default=True, description="Enable embedding cache")
    CACHE_MAX_SIZE: int = Field(default=1000, description="Max cache entries")
    
    # =============================================================================
    # Validators
    # =============================================================================
    
    @validator("API_ENV")
    def validate_env(cls, v):
        """Validate environment value."""
        if v not in ["development", "production", "staging"]:
            raise ValueError("API_ENV must be development, production, or staging")
        return v
    
    
    # @validator("LLM_PROVIDER")
    # def validate_llm_provider(cls, v):
    #     """Validate LLM provider."""
    #     if v not in ["ollama", "openai"]:
    #         raise ValueError("LLM_PROVIDER must be ollama or openai")
    #     return v
    
    
    @validator("EMBEDDING_PROVIDER")
    def validate_embedding_provider(cls, v):
        """Validate embedding provider."""
        if v not in ["sentence-transformers", "openai"]:
            raise ValueError("EMBEDDING_PROVIDER must be sentence-transformers or openai")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level."""
        if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return v
    
    # =============================================================================
    # Helper Properties
    # =============================================================================
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.API_ENV == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.API_ENV == "production"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as list."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def supported_file_types_list(self) -> List[str]:
        """Get supported file types as list."""
        return [ft.strip() for ft in self.SUPPORTED_FILE_TYPES.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL (cloud URL if provided, otherwise local)."""
        if self.QDRANT_URL:
            return self.QDRANT_URL
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
    
    class Config:
        """Pydantic config."""
        env_file = str(Path(__file__).parent.parent.parent / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Global settings instance
settings = get_settings()