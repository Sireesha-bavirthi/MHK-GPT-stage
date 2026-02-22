from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentResponse(BaseModel):
    """Response model for document operations."""
    filename: str
    file_type: str
    size: int
    upload_date: Optional[datetime] = None
    status: str = "processed"
    chunk_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    filename: str
    message: str
    chunks_processed: int
    duration: float
