import json
import logging
from typing import List, Dict, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

_IN_MEMORY_SESSIONS = {}

def _redis():
    try:
        import redis
        client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        if client.ping():
            return client
    except Exception as e:
        logger.warning(f"Redis unavailable for memory: {e}")
    return None

def load_memory(session_id: str) -> Tuple[List[Dict[str, str]], str]:
    """Load conversation messages and summary from Redis or in-memory fallback."""
    client = _redis()
    if not client:
        data = _IN_MEMORY_SESSIONS.get(f"memory:{session_id}")
    else:
        raw = client.get(f"memory:{session_id}")
        data = json.loads(raw) if raw else None

    if data:
        return data.get("messages", []), data.get("summary", "")
    return [], ""

def save_memory(session_id: str, messages: List[Dict[str, str]], summary: str):
    """Save conversation messages and summary to Redis or in-memory fallback."""
    data = {"messages": messages, "summary": summary}
    client = _redis()
    if not client:
        _IN_MEMORY_SESSIONS[f"memory:{session_id}"] = data
    else:
        # Keep memory for 24 hours
        client.setex(f"memory:{session_id}", 86400, json.dumps(data))

def clear_memory(session_id: str):
    """Clear the memory for a session."""
    client = _redis()
    if not client:
        _IN_MEMORY_SESSIONS.pop(f"memory:{session_id}", None)
    else:
        client.delete(f"memory:{session_id}")