"""
FastAPI main application entry point — MHK-GPT v2 Agentic System.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
from pathlib import Path

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.security import get_cors_config
from app.api.v1.router import router as v1_router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter (shared across all endpoints via app.state)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ---------------------------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MHK-GPT v2 — Agentic AI Assistant",
    description=(
        "AI-powered agentic chatbot with RAG, Meeting Scheduling, and Job Search. "
        "Built on LangGraph with multi-intent support."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach rate limiter to app state (required by slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
cors_config = get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

# Include API routers
app.include_router(v1_router, prefix="/api/v1")

# ---------------------------------------------------------------------------
# Serve frontend over HTTP — eliminates file:// CORS null-origin issue
# Visit: http://localhost:8000/app
# ---------------------------------------------------------------------------
_frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root → frontend."""
    return RedirectResponse(url="/app")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 60)
    logger.info("  MHK-GPT v2 — Agentic AI System starting up")
    logger.info("=" * 60)
    logger.info(f"  Environment  : {settings.API_ENV}")
    logger.info(f"  LLM model    : {settings.OPENAI_MODEL}")
    logger.info(f"  Embedding    : {settings.OPENAI_EMBEDDING_MODEL}")
    logger.info(f"  Vector DB    : Qdrant @ {settings.qdrant_url}")
    logger.info(f"  Redis        : {settings.REDIS_URL}")
    logger.info(f"  Rate limit   : {settings.RATE_LIMIT_CHAT} (chat)")
    logger.info("=" * 60)

    # Pre-warm LangGraph compiled agent (fail fast if broken at startup)
    try:
        from app.agent.graph import get_agent_graph
        get_agent_graph()
        logger.info("  LangGraph agent compiled ✓")
    except Exception as e:
        logger.error(f"  LangGraph agent failed to compile: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("=== Shutting down MHK-GPT v2 ===")


# The original root endpoint is replaced by the redirect above.
# @app.get("/")
# async def root():
#     """Root endpoint."""
#     return {
#         "name": "MHK-GPT v2 — Agentic AI Assistant",
#         "version": "2.0.0",
#         "docs": "/docs",
#         "health": "/api/v1/health",
#         "agent_chat": "/api/v1/agent/chat",
#         "analytics": "/api/v1/analytics/cost",
#     }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )



