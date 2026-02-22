from fastapi import APIRouter
from app.api.v1.endpoints import chat, documents, health, agent, analytics

router = APIRouter()

# Legacy RAG endpoint (kept for backward compatibility)
router.include_router(chat.router, tags=["Chat (Legacy RAG)"])

# New agentic endpoints
router.include_router(agent.router, prefix="/agent", tags=["Agent"])
router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])

# Document management & health
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(health.router, tags=["Health"])
