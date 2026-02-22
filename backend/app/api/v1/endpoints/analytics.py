"""
Analytics endpoints — cost tracking and usage statistics.
GET /api/v1/analytics/cost
"""
import logging
from typing import Optional
from fastapi import APIRouter, Query

from app.services.cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/cost")
async def get_cost_analytics(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format, defaults to today"),
):
    """
    Return LLM cost analytics.
    - **Aggregate totals**: total cost USD, total tokens, per-intent call counts
    - **Daily log**: all LLM calls for the specified date (file-based)
    - **Recent calls**: last 50 calls from Redis
    """
    tracker = get_cost_tracker()
    summary = tracker.get_summary()
    daily = tracker.get_daily_costs(date=date)

    return {
        "summary": summary,
        "daily_log": {
            "date": date or "today",
            "entries": daily,
            "count": len(daily),
        },
    }


@router.get("/health")
async def agent_health():
    """
    Check connectivity of all external services used by the agent.
    """
    import httpx
    from app.core.config import settings

    health = {}

    # Redis
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        r.ping()
        health["redis"] = "ok"
    except Exception as e:
        health["redis"] = f"error: {e}"

    # JobDiva (auth check)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = (
                f"{settings.JOBDIVA_BASE_URL}/authenticate"
                f"?clientid={settings.JOBDIVA_CLIENT_ID}"
                f"&username={settings.JOBDIVA_USERNAME}"
                f"&password={settings.JOBDIVA_PASSWORD}"
            )
            resp = await client.get(url, headers={"accept": "application/json"})
            health["jobdiva"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        health["jobdiva"] = f"error: {e}"

    # OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        client.models.list()
        health["openai"] = "ok"
    except Exception as e:
        health["openai"] = f"error: {e}"

    overall = "healthy" if all(v == "ok" for v in health.values()) else "degraded"
    return {"status": overall, "services": health}
