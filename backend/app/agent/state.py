"""
Agent state definitions for MHK-GPT v2 agentic system.
Defines ConversationStage enum and AgentState for LangGraph.
"""
from enum import Enum
from typing import TypedDict, List, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# Conversation Stage Enum
# =============================================================================

class ConversationStage(Enum):
    IDLE = "idle"
    INTENT_DETECTION = "intent_detection"
    CLARIFICATION = "clarification"
    TOOL_SELECTION = "tool_selection"

    # Meeting Scheduler states
    SCHEDULER_EMAIL_REQUESTED = "scheduler_email_requested"
    SCHEDULER_OTP_SENT = "scheduler_otp_sent"
    SCHEDULER_OTP_VERIFIED = "scheduler_otp_verified"
    SCHEDULER_CALENDLY_DISPLAYED = "scheduler_calendly_displayed"
    SCHEDULER_SLOT_SELECTED = "scheduler_slot_selected"
    SCHEDULER_EMAIL_SENT = "scheduler_email_sent"
    SCHEDULER_COMPLETED = "scheduler_completed"
    SCHEDULER_FAILED = "scheduler_failed"

    # Job Search states
    JOB_CACHE_CHECK = "job_cache_check"
    JOB_API_FETCH = "job_api_fetch"
    JOB_ANSWERING = "job_answering"
    JOB_COMPLETED = "job_completed"

    # General Q&A states
    QA_RETRIEVING = "qa_retrieving"
    QA_GENERATING = "qa_generating"
    QA_COMPLETED = "qa_completed"

    # Error states
    ERROR_RECOVERABLE = "error_recoverable"
    ERROR_FATAL = "error_fatal"


# =============================================================================
# Intent Types
# =============================================================================

class IntentType(str, Enum):
    MEETING_SCHEDULER = "meeting_scheduler"
    JOB_SEARCH = "job_search"
    GENERAL_QA = "general_qa"


class SubIntentType(str, Enum):
    # Meeting sub-intents → maps to Calendly duration
    QUICK_CALL = "quick_call"           # 15 min
    NORMAL_MEET = "normal_meet"         # 30 min
    LONG_DISCUSSION = "long_discussion" # 60 min
    # Job sub-intents
    LIST_ALL_JOBS = "list_all_jobs"
    SEARCH_BY_ROLE = "search_by_role"
    SEARCH_BY_LOCATION = "search_by_location"
    # QA (no sub-intents needed)
    GENERAL = "general"


# =============================================================================
# LLM Cost Tracking
# =============================================================================

# Pricing per million tokens (USD) — update as OpenAI changes pricing
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


@dataclass
class LLMUsage:
    """Token usage and cost for a single LLM call."""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    stage: str = ""
    intent: str = ""

    def compute_cost(self) -> None:
        """Compute USD cost from token counts and known pricing."""
        pricing = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        self.cost_usd = (
            (self.prompt_tokens / 1_000_000) * pricing["input"]
            + (self.completion_tokens / 1_000_000) * pricing["output"]
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "stage": self.stage,
            "intent": self.intent,
        }


@dataclass
class LLMCostLog:
    """Aggregated cost log for a single agent turn."""
    session_id: str = ""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: List[LLMUsage] = field(default_factory=list)

    def add(self, usage: LLMUsage) -> None:
        """Add a single LLM call's usage."""
        self.calls.append(usage)
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost_usd += usage.cost_usd

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "calls": [c.to_dict() for c in self.calls],
        }


# =============================================================================
# Detected Intent
# =============================================================================

@dataclass
class IntentResult:
    """A single detected intent from the user message."""
    intent: IntentType
    sub_intent: SubIntentType
    confidence: float
    extracted_params: dict = field(default_factory=dict)  # role/location/etc.

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "sub_intent": self.sub_intent.value,
            "confidence": self.confidence,
            "extracted_params": self.extracted_params,
        }


# =============================================================================
# LangGraph Agent State
# =============================================================================

class AgentState(TypedDict):
    """
    Full state for the LangGraph agent.
    Passed between all nodes in the graph.
    """
    # Core conversation
    messages: List[dict]              # OpenAI-format conversation history
    session_id: str                   # Unique per-user session
    current_query: str                # The current user message

    # Stage tracking
    stage: str                        # ConversationStage.value

    # Multi-intent support
    detected_intents: List[dict]      # List of IntentResult.to_dict()
    active_intent_index: int          # Which intent is being processed
    tool_results: List[dict]          # One result dict per intent

    # Meeting scheduler sub-state key (actual session in Redis)
    meeting_session_id: Optional[str]

    # Cost tracking for this turn
    cost_log: dict                    # LLMCostLog.to_dict()

    # Control flow
    awaiting_clarification: bool
    clarification_turns: int          # Max 2 turns before giving up
    error_message: Optional[str]      # Set on ERROR states

    # Final aggregated response
    final_response: str
