from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

def get_cors_config() -> dict:
    """
    Get CORS configuration parameters.
    Returns a dictionary suitable for CORSMiddleware.
    """
    return {
        "allow_origins": settings.cors_origins_list,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
