from typing import Optional, Any, Dict
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    environment: str
    components: Dict[str, str]

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Any] = None
