"""
Agent chat endpoint — the new entry point for the MHK-GPT v2 agentic system.
POST /api/v1/agent/chat
"""
import time
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.agent.state import AgentState, ConversationStage, LLMCostLog
from app.agent.graph import get_agent_graph
from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def _build_initial_state(request: ChatRequest, session_id: str) -> AgentState:
    """Build a fresh AgentState from the incoming request."""
    history = []
    for msg in (request.history or []):
        history.append({"role": msg.role, "content": msg.content})

    return AgentState(
        messages=history,
        session_id=session_id,
        current_query=request.query,
        stage=ConversationStage.INTENT_DETECTION.value,
        detected_intents=[],
        active_intent_index=0,
        tool_results=[],
        # Restore meeting_session_id from frontend — critical for multi-turn meeting flows
        meeting_session_id=request.meeting_session_id or None,
        cost_log=LLMCostLog(session_id=session_id).to_dict(),
        awaiting_clarification=False,
        clarification_turns=0,
        error_message=None,
        final_response="",
    )


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def agent_chat(request: Request, body: ChatRequest):
    """
    Main agentic chat endpoint.

    - Detects intent (or multiple intents) from the user query
    - Routes to appropriate tools: RAG pipeline, JobDiva, or Meeting Scheduler
    - Aggregates multi-tool results into a single coherent response
    - Tracks LLM token usage and cost per turn
    - Rate limited to 20 req/min per IP

    NOTE: slowapi requires the Starlette Request param to be literally named 'request'.
    The ChatRequest body is received as 'body'.
    """
    start_time = time.time()

    # Session management: use provided conversation_id or generate one
    session_id = body.conversation_id or str(uuid.uuid4())

    logger.info(
        f"[AgentChat] session={session_id} "
        f"query={body.query[:80]!r} "
        f"history_len={len(body.history or [])}"
    )

    # Build initial state
    initial_state = _build_initial_state(body, session_id)

    # Run the LangGraph agent (async — required for async tool nodes)
    try:
        graph = get_agent_graph()
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"[AgentChat] Graph execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Extract response and sources
    final_response = final_state.get("final_response", "")
    if not final_response:
        final_response = "I'm sorry, I wasn't able to generate a response. Please try again."

    # Extract sources from QA tool results (if any)
    sources = []
    for tool_result in final_state.get("tool_results", []):
        if tool_result.get("type") == "general_qa":
            raw_sources = tool_result.get("sources", [])
            for src in raw_sources[:5]:
                if hasattr(src, "metadata"):
                    sources.append({
                        "file_name": src.metadata.get("source", ""),
                        "page_content": src.page_content[:300] if hasattr(src, "page_content") else "",
                        "metadata": src.metadata,
                    })
                elif isinstance(src, dict):
                    sources.append(src)

    duration = time.time() - start_time
    cost_log = final_state.get("cost_log", {})
    total_cost = cost_log.get("total_cost_usd", 0.0)
    total_tokens = cost_log.get("total_tokens", 0)

    logger.info(
        f"[AgentChat] session={session_id} "
        f"duration={duration:.2f}s "
        f"tokens={total_tokens} "
        f"cost=${total_cost:.6f} "
        f"intents={[r.get('type') for r in final_state.get('tool_results', [])]}"
    )

    return ChatResponse(
        response=final_response,
        sources=sources,
        total_duration=round(duration, 3),
        # Return meeting_session_id so frontend echoes it back next turn
        meeting_session_id=final_state.get("meeting_session_id") or None,
    )
