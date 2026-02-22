import time
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.schemas.chat import ChatRequest, ChatResponse
from app.api.dependencies import get_rag_pipeline
from rag_pipeline import RAGPipeline

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Process a chat message using the RAG pipeline.
    """
    try:
        # Pass request.history as conversation_history if needed, 
        # but RAGPipeline uses internal memory by default.
        # If conversation_id is handled by RAGPipeline's memory (which is simple buffer for now),
        # we might need to handle per-user memory later.
        # For now, we rely on the pipeline's internal state which is shared in this singleton scope, 
        # CAUTION: This means memory is shared across requests in this simple implementation.
        # Real impl should handle session/user IDs.
        
        # Fixing potential issue: RAGPipeline memory is stateful.
        # Ideally, we should use a session-based memory or pass history.
        
        # Convert Pydantic models to dicts for the pipeline
        history_dicts = None
        if request.history:
            history_dicts = [msg.model_dump() for msg in request.history]

        result = pipeline.query(
            query=request.query,
            conversation_history=history_dicts, # Use provided history if available
            update_memory=True
        )
        # Convert RetrievalResult objects to dicts for the response
        sources = [
            {
                "file_name": r.metadata.get("file_name", "unknown"),
                "page_content": r.text,
                "metadata": r.metadata
            }
            for r in result.retrieval_results
        ]
        
        return ChatResponse(
            response=result.response,
            sources=sources,
            total_duration=result.total_duration
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
