"""
LLM Cost Tracker service.
Logs token usage and USD cost per LLM call to Redis + daily log files.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.agent.state import LLMUsage, LLMCostLog
from app.core.config import settings

logger = logging.getLogger(__name__)


class CostTrackerService:
    """
    Tracks LLM usage cost across all agent invocations.
    - Writes structured JSON to daily log files
    - Accumulates totals in Redis sorted set (last 10k entries)
    """

    def __init__(self):
        self._redis_client = None
        self._log_base = settings.project_root / "data" / "logs" / "costs"
        self._log_base.mkdir(parents=True, exist_ok=True)

    @property
    def redis(self):
        if self._redis_client is None:
            try:
                import redis
                self._redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                self._redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis not available for cost tracking: {e}")
                self._redis_client = None
        return self._redis_client

    # -------------------------------------------------------------------------
    def log_usage(self, usage: LLMUsage, session_id: str = "") -> None:
        """
        Log a single LLM call's usage.
        Writes to daily JSON log file + Redis.
        """
        usage.compute_cost()

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            **usage.to_dict(),
        }

        # ----- Write to daily log file -----
        try:
            today = datetime.now()
            log_dir = self._log_base / today.strftime("%Y") / today.strftime("%m")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"costs_{today.strftime('%Y-%m-%d')}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write cost log to file: {e}")

        # ----- Accumulate in Redis -----
        try:
            if self.redis:
                score = datetime.now(timezone.utc).timestamp()
                self.redis.zadd("mhk:cost_log", {json.dumps(record): score})
                # Keep only last 10,000 entries
                self.redis.zremrangebyrank("mhk:cost_log", 0, -10001)

                # Increment aggregate counters
                pipe = self.redis.pipeline()
                pipe.incrbyfloat("mhk:cost_total_usd", usage.cost_usd)
                pipe.incrby("mhk:cost_total_tokens", usage.total_tokens)
                pipe.incr(f"mhk:cost_intent:{usage.intent}")
                pipe.execute()
        except Exception as e:
            logger.warning(f"Failed to save cost to Redis: {e}")

        logger.info(
            f"[COST] model={usage.model} intent={usage.intent} stage={usage.stage} "
            f"tokens={usage.total_tokens} (in={usage.prompt_tokens}, out={usage.completion_tokens}) "
            f"cost=${usage.cost_usd:.6f}"
        )

    def log_turn_cost(self, cost_log: LLMCostLog) -> None:
        """Log the aggregated cost for a full agent turn."""
        logger.info(
            f"[COST SUMMARY] session={cost_log.session_id} "
            f"total_tokens={cost_log.total_tokens} "
            f"total_cost=${cost_log.total_cost_usd:.6f} "
            f"calls={len(cost_log.calls)}"
        )

    # -------------------------------------------------------------------------
    def get_summary(self) -> dict:
        """Return aggregate cost summary from Redis."""
        if not self.redis:
            return {"error": "Redis unavailable", "total_cost_usd": 0}

        try:
            total_usd = float(self.redis.get("mhk:cost_total_usd") or 0)
            total_tokens = int(self.redis.get("mhk:cost_total_tokens") or 0)

            # Per-intent counters
            intent_breakdown = {}
            for intent in ["meeting_scheduler", "job_search", "general_qa"]:
                count = self.redis.get(f"mhk:cost_intent:{intent}")
                intent_breakdown[intent] = int(count or 0)

            # Recent entries (last 50)
            recent_raw = self.redis.zrevrange("mhk:cost_log", 0, 49, withscores=False)
            recent = [json.loads(r) for r in recent_raw]

            return {
                "total_cost_usd": round(total_usd, 6),
                "total_tokens": total_tokens,
                "intent_breakdown": intent_breakdown,
                "recent_calls": recent,
            }
        except Exception as e:
            logger.error(f"Failed to get cost summary: {e}")
            return {"error": str(e)}

    def get_daily_costs(self, date: Optional[str] = None) -> list:
        """Read daily cost log file and return all entries."""
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            year, month, _ = date.split("-")
            log_file = self._log_base / year / month / f"costs_{date}.jsonl"
            if not log_file.exists():
                return []
            entries = []
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries
        except Exception as e:
            logger.error(f"Failed to read daily cost log: {e}")
            return []


# Singleton
_cost_tracker: Optional[CostTrackerService] = None


def get_cost_tracker() -> CostTrackerService:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTrackerService()
    return _cost_tracker
