"""
JobDiva API integration tool.
Handles authentication (with Redis token cache), job search using the live
/api/bi/OpenJobsList endpoint, circuit breaker, and LLM formatting.

No hardcoded/sample data — only real jobs from the JobDiva API.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Live endpoint — returns all currently open jobs as a row-array response
_OPEN_JOBS_URL = "https://api.jobdiva.com/api/bi/OpenJobsList"

# Auth endpoint (base URL from settings, token cached in Redis)
# e.g. https://api.jobdiva.com/apiv2/authenticate
_AUTH_URL_TEMPLATE = (
    "{base}/authenticate"
    "?clientid={clientid}"
    "&username={username}"
    "&password={password}"
)

_CAREERS_URL = "https://www.mhktechinc.com/careers"

# Column order returned by /api/bi/OpenJobsList (row 0 is the header)
_COLUMNS = [
    "JOBID", "JOBDIVANO", "OPTIONALREFERENCENO", "DIVISIONID", "DIVISIONNAME",
    "PRIMARYRECRUITERID", "PRIMARYSALESID", "COMPANYID", "COMPANYNAME",
    "PRIMARYOWNERID", "CONTACTID", "CONTACTNAME", "ISSUEDATE", "DATEUPDATED",
    "STARTDATE", "ENDDATE", "POSITIONTYPE", "JOBSTATUS", "TITLE",
    "OPENINGS", "FILLS", "MAXALLOWEDSUBMITTALS", "CITY", "STATE", "ZIPCODE",
    "PRIORITY", "BILLRATEMIN", "BILLRATEMAX", "BILLFREQUENCY",
    "PAYRATEMIN", "PAYRATEMAX", "PAYFREQUENCY", "CURRENCY",
    "ONSITE_FLEXIBILITY", "REMOTE_PERCENTAGE",
]


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitBreaker:
    """
    Simple in-process circuit breaker.
    States: CLOSED (normal) → OPEN (too many failures) → HALF_OPEN (probing)
    """
    def __init__(self, failure_threshold: int, recovery_timeout: int):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: Optional[datetime] = None
        self._state = "CLOSED"

    @property
    def is_open(self) -> bool:
        if self._state == "OPEN":
            if (datetime.now() - self._opened_at).seconds >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                logger.info("[CircuitBreaker] JobDiva → HALF_OPEN (probing)")
                return False
            return True
        return False

    def on_success(self):
        self._failures = 0
        self._state = "CLOSED"

    def on_failure(self):
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "OPEN"
            self._opened_at = datetime.now()
            logger.warning(
                f"[CircuitBreaker] JobDiva → OPEN after {self._failures} failures. "
                f"Recovery in {self.recovery_timeout}s"
            )


_circuit_breaker = CircuitBreaker(
    failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
)


# =============================================================================
# Redis helper
# =============================================================================

def _get_redis():
    try:
        import redis as redis_lib
        client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


# =============================================================================
# Authentication
# =============================================================================

async def get_token() -> Optional[str]:
    """
    Authenticate with JobDiva API.
    Token is cached in Redis with 30-min TTL.
    """
    redis = _get_redis()
    cache_key = "jobdiva:token"

    if redis:
        cached = redis.get(cache_key)
        if cached:
            logger.debug("JobDiva token retrieved from Redis cache")
            return cached

    url = _AUTH_URL_TEMPLATE.format(
        base=settings.JOBDIVA_BASE_URL,
        clientid=settings.JOBDIVA_CLIENT_ID,
        username=settings.JOBDIVA_USERNAME,
        password=settings.JOBDIVA_PASSWORD,
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers={"accept": "application/json;charset=UTF-8"})
            resp.raise_for_status()

        raw = resp.text.strip()
        token = None

        if raw.startswith("ey"):
            token = raw
        else:
            try:
                data = resp.json()
                token = data.get("token") or data.get("SESSIONID") or data.get("sessionid")
            except Exception:
                token = raw if raw else None

        if not token:
            logger.error(f"JobDiva auth response missing token. Raw: {raw[:100]!r}")
            return None

        if redis:
            redis.setex(cache_key, 1800, token)
            logger.debug("JobDiva token cached in Redis for 30 minutes")

        _circuit_breaker.on_success()
        logger.info("JobDiva authentication successful")
        return token

    except Exception as e:
        _circuit_breaker.on_failure()
        logger.error(f"JobDiva authentication failed: {e}")
        return None


# =============================================================================
# Row-array → dict conversion
# =============================================================================

def _rows_to_dicts(data: dict) -> List[dict]:
    """
    /api/bi/OpenJobsList returns:
      {"message": "...", "data": [["COL1","COL2",...], [val1,val2,...], ...]}

    Row 0 is the header. Remaining rows are jobs.
    Returns a list of dicts keyed by column name.
    Falls back to _COLUMNS if the header row is absent/different.
    """
    rows = data.get("data", [])
    if not rows:
        return []

    # Use the actual header returned by the API if available
    first = rows[0]
    if isinstance(first, list) and all(isinstance(c, str) and c.isupper() for c in first[:3]):
        headers = first
        data_rows = rows[1:]
    else:
        headers = _COLUMNS
        data_rows = rows

    jobs = []
    for row in data_rows:
        if not isinstance(row, list):
            continue
        job = {headers[i]: row[i] if i < len(row) else None for i in range(len(headers))}
        jobs.append(job)
    return jobs


# =============================================================================
# Fetch jobs from API
# =============================================================================

@retry(
    stop=stop_after_attempt(settings.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _fetch_open_jobs(token: str) -> List[dict]:
    """Fetch all currently open jobs from /api/bi/OpenJobsList."""
    headers = {
        "accept": "application/json;charset=UTF-8",
        "Authorization": f"Bearer {token}",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(_OPEN_JOBS_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    return _rows_to_dicts(data)


# =============================================================================
# Main search function
# =============================================================================

async def search_jobs(
    role: Optional[str] = None,
    location: Optional[str] = None,
    status: Optional[str] = None,
    query_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search for open job positions from JobDiva live API.
    Fetches all jobs, caches them in Redis under a single key, and applies client-side filtering.
    """
    if _circuit_breaker.is_open:
        return {
            "error": "Job search is temporarily unavailable. Please try again in a moment.",
            "jobs": [],
            "cached": False,
        }

    redis = _get_redis()
    cache_key = "jobdiva:all_jobs"
    raw_jobs = None
    is_cached = False

    # 1) Try to load the FULL job list from Redis
    if redis:
        try:
            cached_raw = redis.get(cache_key)
            if cached_raw:
                raw_jobs = json.loads(cached_raw)
                is_cached = True
                logger.info("JobDiva full job list loaded from Redis cache")
        except Exception as e:
            logger.error(f"Failed to read from Redis cache: {e}")

    # 2) If not in Redis, fetch from JobDiva API and cache
    if raw_jobs is None:
        token = await get_token()
        if not token:
            return {
                "error": "Could not authenticate with JobDiva. Please try again later.",
                "jobs": [],
                "cached": False,
            }

        try:
            raw_jobs = await _fetch_open_jobs(token)
            _circuit_breaker.on_success()
            logger.info(f"JobDiva returned {len(raw_jobs)} open jobs from live API")
            
            # Cache the full list
            if redis:
                try:
                    redis.setex(cache_key, settings.JOB_CACHE_TTL_SECONDS, json.dumps(raw_jobs))
                    logger.info(f"JobDiva full job list cached for {settings.JOB_CACHE_TTL_SECONDS}s")
                except Exception as e:
                    logger.error(f"Failed to write to Redis cache: {e}")
        except Exception as e:
            _circuit_breaker.on_failure()
            logger.error(f"JobDiva fetch failed: {e}")
            return {
                "error": "Unable to retrieve job listings right now. Please check back shortly.",
                "jobs": [],
                "cached": False,
            }

    # 3) Client-side filtering
    filtered = raw_jobs
    if role and query_type != "job_detail":
        role_lower = role.lower()
        filtered = [
            j for j in filtered
            if role_lower in (str(j.get("TITLE") or "")).lower()
        ]
    if location:
        loc_lower = location.lower()
        filtered = [
            j for j in filtered
            if (
                loc_lower in (str(j.get("CITY") or "")).lower()
                or loc_lower in (str(j.get("STATE") or "")).lower()
                or loc_lower in (str(j.get("ZIPCODE") or "")).lower()
            )
        ]
    if status:
        status_lower = status.lower()
        filtered = [
            j for j in filtered
            if status_lower in (str(j.get("JOBSTATUS") or "")).lower()
        ]

    return {
        "jobs": filtered,
        "total": len(filtered),
        "raw_total": len(raw_jobs),
        "filters": {"role": role, "location": location, "status": status},
        "cached": is_cached,
    }


# =============================================================================
# Helpers
# =============================================================================

def _get_apply_url(job: dict) -> str:
    """
    All MHK Tech jobs are internal — not published on the public JobDiva portal.
    Always direct candidates to MHK Tech's own careers page.
    """
    return _CAREERS_URL


def _fmt_salary(job: dict) -> str:
    """Format pay/bill rate from a live JobDiva job record."""
    pay_min = job.get("PAYRATEMIN")
    pay_max = job.get("PAYRATEMAX")
    pay_freq = (job.get("PAYFREQUENCY") or "Hourly").capitalize()

    bill_min = job.get("BILLRATEMIN")
    bill_max = job.get("BILLRATEMAX")

    # Prefer pay rate; fall back to bill rate
    rate_min = _nonzero(pay_min) or _nonzero(bill_min)
    rate_max = _nonzero(pay_max) or _nonzero(bill_max)
    freq = pay_freq if _nonzero(pay_min) else (job.get("BILLFREQUENCY") or "Hourly").capitalize()

    if rate_min and rate_max:
        return f"${float(rate_min):,.0f}–${float(rate_max):,.0f}/{freq.lower()}"
    if rate_min:
        return f"${float(rate_min):,.0f}+/{freq.lower()}"
    return "Not disclosed"


def _nonzero(val) -> Optional[float]:
    """Return float value if nonzero/non-null, else None."""
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


# =============================================================================
# LLM Formatter
# =============================================================================

def format_jobs_for_llm(search_result: Dict[str, Any]) -> str:
    """
    Format job search results as rich structured text for the LLM.
    """
    if search_result.get("error"):
        return (
            f" {search_result['error']}\n"
            f" You can browse open positions at: {_CAREERS_URL}"
        )

    jobs = search_result.get("jobs", [])
    if not jobs:
        filters = search_result.get("filters", {})
        role_str = f" for '{filters.get('role')}'" if filters.get("role") else ""
        loc_str = f" in {filters.get('location')}" if filters.get("location") else ""
        return (
            f"No open positions found{role_str}{loc_str} at MHK Tech at this time.\n"
            f"You can check current openings at: {_CAREERS_URL}"
        )

    lines = [f"Found {len(jobs)} open position(s) at MHK Tech:\n"]
    for i, job in enumerate(jobs[:20], 1):
        title        = job.get("TITLE") or "Untitled"
        city         = job.get("CITY") or ""
        state        = job.get("STATE") or ""
        job_status   = job.get("JOBSTATUS") or "Open"
        job_ref      = job.get("JOBDIVANO") or job.get("JOBID") or str(i)
        pos_type     = job.get("POSITIONTYPE") or ""
        openings     = job.get("OPENINGS") or "1"
        remote_pct   = _nonzero(job.get("REMOTE_PERCENTAGE"))

        # Location
        location_str = ", ".join(filter(None, [city, state])) or "Not specified"
        if remote_pct and remote_pct >= 100:
            location_str = "Remote"
        elif remote_pct and remote_pct > 0:
            location_str = f"{location_str} ({int(remote_pct)}% Remote)"

        salary_str = _fmt_salary(job)
        apply_url  = _get_apply_url(job)

        block = [
            f"{i}. **{title}** (Ref: {job_ref})",
            f"    Location : {location_str} | Type: {pos_type or 'N/A'} | Status: {job_status}",
            f"    Pay Rate  : {salary_str}",
            f"    Openings  : {openings}",
            f"    Apply     : {apply_url}",
        ]
        lines.append("\n".join(block))

    if search_result.get("cached"):
        lines.append("\n_(Results cached — refreshed periodically)_")

    return "\n\n".join(lines)