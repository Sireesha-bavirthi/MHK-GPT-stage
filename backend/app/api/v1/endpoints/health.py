from fastapi import APIRouter
from app.core.config import settings
from app.schemas.common import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Get system health status.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.API_ENV,
        "components": {
            "db": "connected",  # simplified
            "llm": settings.OPENAI_MODEL
        }
    }
