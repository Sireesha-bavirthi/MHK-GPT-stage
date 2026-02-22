"""
LangGraph Agent Nodes.
All compute happens here — each node receives AgentState, does work, and returns a partial state update.
"""

import json
import logging
import asyncio
from typing import List, Optional
from datetime import datetime, timezone

from openai import OpenAI
from app.core.config import settings
from app.agent.state import (
    AgentState, ConversationStage, IntentType, SubIntentType, IntentResult,
    LLMUsage, LLMCostLog, PRICING,
)
from app.services.cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared OpenAI client
# ---------------------------------------------------------------------------

def _openai_client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def _chat_completion(messages: list, model: str = None, **kwargs) -> tuple:
    """
    Call OpenAI chat completion.
    Returns (content, usage_dict).
    """
    model = model or settings.OPENAI_MODEL
    client = _openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=kwargs.get("temperature", settings.OPENAI_TEMPERATURE),
        max_tokens=kwargs.get("max_tokens", settings.OPENAI_MAX_TOKENS),
    )
    content = resp.choices[0].message.content or ""
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "total_tokens": resp.usage.total_tokens,
    }
    return content, usage


def _track_cost(usage: dict, model: str, stage: str, intent: str, session_id: str) -> LLMUsage:
    """Build LLMUsage, compute cost, log it."""
    llm_usage = LLMUsage(
        model=model,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        stage=stage,
        intent=intent,
    )
    llm_usage.compute_cost()
    get_cost_tracker().log_usage(llm_usage, session_id=session_id)
    return llm_usage


# =============================================================================
# Node 1: Intent Detection
# =============================================================================

INTENT_SYSTEM_PROMPT = """You are an intent classifier for MHK Tech's AI assistant.
Your name is MHK Nova, and you are a helpful assistant for MHK Tech.
Use "we" and "our" when referring to MHK — you're part of the team
You will receive the CONVERSATION HISTORY followed by the LATEST USER MESSAGE.
Your job: classify only the LATEST user message into one or more intents.

══ OUTPUT FORMAT ══
Respond ONLY with a JSON array. No markdown, no explanation. Example:
[
  {"intent": "job_search", "sub_intent": "skills_query", "confidence": 0.95,
   "extracted_params": {"role": "Data Engineer", "location": null, "status": null, "query_type": "skills_query"}},
  {"intent": "general_qa", "sub_intent": "general", "confidence": 0.97, "extracted_params": {}}
]

══ INTENTS ══
intent values: meeting_scheduler | job_search | general_qa

sub_intent values:
  meeting_scheduler → quick_call (≤15min) | normal_meet (30–45min, default) | long_discussion (≥1hr)
  job_search        → list_all_jobs | search_by_role | search_by_location | skills_query | salary_query | experience_query | remote_query | job_detail
  general_qa        → general

extracted_params (job_search only):
  { "role": "<title or null>", "location": "<city/state or null>", "status": "<status or null>", "query_type": "<one of the sub_intents above>" }

══ HOW TO USE CONVERSATION HISTORY ══
The history gives you CONTEXT to correctly classify short or ambiguous messages.

1. FOLLOW-UP DETECTION
   If the latest message is short and the previous topic was job_search, treat it as job_search:
   - "what about data science?", "any remote ones?", "yes", "tell me more", "what skills?",
     "how much does it pay?", "is it remote?" → KEEP as job_search, carry over the role/location.
   If the previous topic was meeting_scheduler, treat the reply as meeting_scheduler:
   - "2", "30 minutes", an email address, a 6-digit code → meeting_scheduler 0.99.
   (Note: If the scheduling is effectively finished and the user just says "ok", "thanks" or "no thanks", classify as general_qa).

2. CONTEXT CARRY-OVER
   If the user mentioned a role/location earlier and now asks a follow-up without repeating it,
   extract those params from conversation history:
   - Earlier: "show me Data Engineer jobs in Houston"
   - Latest:  "what skills do I need?" → job_search, skills_query, role="Data Engineer", location="Houston"

3. TOPIC SWITCHING & MIDDLE QUESTIONS
   If the user clearly changes topic or asks an out-of-flow question, classify the new topic:
   - Earlier job conversation, but latest: "I want to schedule a meeting" → meeting_scheduler.
   - Earlier meeting flow, but latest: "actually, show me open jobs" → job_search.
   - Earlier meeting flow, but latest: "What is AI?" or "Who is the CEO?" → general_qa.

4. COMPANY INFO, RAG RETRIEVAL & BOT IDENTITY / INFORMATIVE QUESTIONS / FACTS
   If the user asks about the company (e.g. MHK Tech, headquarters, locations, projects, clients, CEO, founding date, services), asks about your identity (e.g. "who are you", "what is MHK Nova", "what can you do"), or asks any informational question that does not fit job search or meeting scheduling:
   - Classify as general_qa, sub_intent: general with confidence 0.95 or higher.

5. GREETINGS, CHITCHAT, CONFIRMATIONS & CLOSINGS
   If the user says things like "hi", "hello", "hey", "good morning", "how are you", "bye", "goodbye", "thank you", "thanks", "see you later", "ok", "okay", "sure", "sounds good", "perfect", "done", "no thanks",
   classify as general_qa, sub_intent: general with confidence 1.0.

6. AMBIGUOUS / VERY SHORT MESSAGES (without clear history context)
   If you truly cannot tell — use general_qa with confidence ≤ 0.60.
   Do NOT invent a team name or ask clarifying questions; just classify.

══ MEETING SCHEDULER RULES ══
Always classify as meeting_scheduler with confidence ≥ 0.92 when the user says ANY of:
  "schedule a meeting", "book a meeting", "set up a meeting", "arrange a meeting",
  "schedule a call", "book a call", "set up a call",
  "talk to someone", "speak to hr", "connect with hr", "connect with your team",
  "meet with", "i want to meet", "can we meet", "schedule time", "book time",
  "i’d like to schedule", "can you schedule", "help me schedule", "schedule with",
  or any paraphrase meaning “I want to have a meeting/call with someone at MHK Tech”.
  Sub-intent: normal_meet (default), quick_call (quick/brief/short), long_discussion (long/in-depth/detailed).
  Do NOT ask which team — the scheduling tool handles that.

══ JOB SEARCH SUB-INTENT GUIDE ══
  list_all_jobs       → "what jobs are open?", "show all positions"
  search_by_role      → "are there Python developer roles?", "show me Data Engineer jobs"
  search_by_location  → "jobs in Houston", "what’s available in Texas?"
  skills_query        → "what skills do I need?", "what qualifications are required?"
  salary_query        → "what’s the salary?", "how much does it pay?", "compensation?"
  experience_query    → "how many years of experience?", "is this entry-level?"
  remote_query        → "is it remote?", "can I work from home?"
  job_detail          → "tell me more about the Senior Data Engineer role", "can you explain Data Engineer role", "what is data engineering role in your company", "explain [Role Name]", "describe the [Role Name] job"
"""


def intent_detection_node(state: AgentState) -> dict:
    """Detect one or more intents from the current user query.

    Short-circuits to meeting_scheduler if there is an active meeting session
    in a non-terminal state (email/OTP/duration/slot steps) — avoids wasting
    tokens classifying short follow-up replies like '2', an OTP, or an email.
    """
    query = state["current_query"]
    session_id = state["session_id"]
    model = settings.OPENAI_MODEL
    cost_log = LLMCostLog(session_id=session_id)

    # ------------------------------------------------------------------
    # Fast-path: active meeting session → check heuristic to decide if we route
    # to meeting_scheduler or fall through to LLM for middle questions.
    # ------------------------------------------------------------------
    meeting_session_id = state.get("meeting_session_id")
    if meeting_session_id:
        from app.tools.meeting_scheduler import load_session
        import re
        sess = load_session(meeting_session_id)
        if sess and sess.state not in ("initial", "completed"):
            query_lower = query.lower().strip()
            cancel_keywords = ["cancel", "stop", "exit", "quit", "nevermind"]
            is_cancel = any(kw in query_lower for kw in cancel_keywords)
            
            should_fast_path = is_cancel
            if not should_fast_path:
                if sess.state == "awaiting_email" and "@" in query and "." in query:
                    should_fast_path = True
                elif sess.state == "awaiting_otp":
                    if re.search(r"\b\d{6}\b", query):
                        should_fast_path = True
                    elif "@" in query and "." in query:
                        should_fast_path = True # changing email address mid-OTP
                    elif any(word in query_lower for word in ["wrong", "incorrect", "change", "not my", "other", "different"]):
                        should_fast_path = True # wants to change email but hasn't provided it yet
                elif sess.state in ("awaiting_duration", "awaiting_slot") and len(query) <= 5:
                    should_fast_path = True
                elif any(kw in query_lower for kw in ["yes", "ok", "sure", "no"]):
                    should_fast_path = True
            
            if should_fast_path:
                intents = [IntentResult(
                    intent=IntentType.MEETING_SCHEDULER,
                    sub_intent=SubIntentType(sess.sub_intent) if sess.sub_intent else SubIntentType.NORMAL_MEET,
                    confidence=1.0,
                )]
                logger.info(
                    f"[IntentNode] session={session_id} fast-path → meeting_scheduler "
                    f"(meeting_state={sess.state})"
                )
                return {
                    "stage": ConversationStage.TOOL_SELECTION.value,
                    "detected_intents": [i.to_dict() for i in intents],
                    "awaiting_clarification": False,
                    "cost_log": cost_log.to_dict(),
                }

    # ------------------------------------------------------------------
    # Fast-path 2: obvious meeting/scheduling keywords → skip LLM entirely
    # ------------------------------------------------------------------
    _MEETING_KEYWORDS = (
        "schedule a meeting", "book a meeting", "set up a meeting", "arrange a meeting",
        "schedule a call", "book a call", "set up a call",
        "talk to someone", "speak to hr", "connect with hr", "connect with your team",
        "meet with", "i want to meet", "can we meet", "schedule time", "book time",
        "i'd like to schedule", "can you schedule", "help me schedule",
        "schedule with", "set up with", "arrange with",
    )
    query_lower = query.lower().strip()
    if any(kw in query_lower for kw in _MEETING_KEYWORDS):
        intents = [IntentResult(
            intent=IntentType.MEETING_SCHEDULER,
            sub_intent=SubIntentType.NORMAL_MEET,
            confidence=1.0,
        )]
        logger.info(f"[IntentNode] session={session_id} keyword fast-path → meeting_scheduler")
        return {
            "stage": ConversationStage.TOOL_SELECTION.value,
            "detected_intents": [i.to_dict() for i in intents],
            "awaiting_clarification": False,
            "cost_log": cost_log.to_dict(),
        }

    # ------------------------------------------------------------------
    # Build LLM messages — include last N conversation turns for context
    # ------------------------------------------------------------------
    conversation_history = state.get("messages", [])
    # Include up to last 10 messages (5 full turns) for richer context
    recent_history = conversation_history[-10:] if conversation_history else []

    llm_messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
    # Add conversation history as individual messages
    for msg in recent_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            llm_messages.append({"role": role, "content": content})
    # Finally the current query
    llm_messages.append({"role": "user", "content": query})

    try:
        content, usage = _chat_completion(llm_messages, model=model, temperature=0.1, max_tokens=512)
        llm_usage = _track_cost(usage, model, stage="intent_detection", intent="unknown", session_id=session_id)
        cost_log.add(llm_usage)

        intents_raw = json.loads(content)
        intents = []
        for item in intents_raw:
            try:
                intents.append(IntentResult(
                    intent=IntentType(item["intent"]),
                    sub_intent=SubIntentType(item["sub_intent"]),
                    confidence=float(item["confidence"]),
                    extracted_params=item.get("extracted_params", {}),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed intent item {item}: {e}")

        if not intents:
            raise ValueError("No valid intents parsed")

    except Exception as e:
        logger.error(f"Intent detection failed: {e}")
        intents = [IntentResult(
            intent=IntentType.GENERAL_QA,
            sub_intent=SubIntentType.GENERAL,
            confidence=0.5,
        )]

    threshold = settings.INTENT_CONFIDENCE_THRESHOLD
    def get_threshold(intent_type: IntentType) -> float:
        if intent_type == IntentType.JOB_SEARCH:
            return 0.65
        elif intent_type == IntentType.GENERAL_QA:
            return 0.30  # Low threshold so we always pass general questions to the LLM
        return threshold

    all_above = all(i.confidence >= get_threshold(i.intent) for i in intents)

    logger.info(
        f"[IntentNode] session={session_id} "
        f"intents={[(i.intent.value, round(i.confidence,2)) for i in intents]} "
        f"history_turns={len(recent_history)//2}"
    )

    return {
        "stage": ConversationStage.CLARIFICATION.value if not all_above else ConversationStage.TOOL_SELECTION.value,
        "detected_intents": [i.to_dict() for i in intents],
        "awaiting_clarification": not all_above,
        "cost_log": cost_log.to_dict(),
    }


# =============================================================================
# Node 2: Clarification
# =============================================================================

# Static rephrase message — no LLM call, no verbose clarifying questions
_REPHRASE_MESSAGE = "I'm not sure I understood that. Could you rephrase or tell me what you'd like help with? (e.g. \"show me open jobs\", \"schedule a meeting\", or ask me anything about MHK Tech)"


def clarification_node(state: AgentState) -> dict:
    """Ask user to rephrase. After max turns, fall back to general_qa."""
    session_id = state["session_id"]
    turns = state.get("clarification_turns", 0)

    # If already asked once, fall back to general_qa rather than looping
    if turns >= settings.MAX_CLARIFICATION_TURNS:
        logger.info(f"[ClarificationNode] Max turns reached, falling back to general_qa")
        fallback = IntentResult(
            intent=IntentType.GENERAL_QA, sub_intent=SubIntentType.GENERAL, confidence=1.0
        )
        return {
            "stage": ConversationStage.TOOL_SELECTION.value,
            "detected_intents": [fallback.to_dict()],
            "awaiting_clarification": False,
            "clarification_turns": 0,
        }

    logger.info(f"[ClarificationNode] session={session_id} asking user to rephrase (turn {turns+1})")
    return {
        "stage": ConversationStage.CLARIFICATION.value,
        "final_response": _REPHRASE_MESSAGE,
        "awaiting_clarification": True,
        "clarification_turns": turns + 1,
        "cost_log": LLMCostLog(session_id=session_id).to_dict(),
    }


# =============================================================================
# Node 3: Tool Execution (handles all intents in parallel for multi-intent)
# =============================================================================

async def _run_general_qa(query: str, session_id: str, messages: list, cost_log: LLMCostLog) -> dict:
    """Run the existing RAG pipeline for general Q&A."""
    # RAGPipeline lives at backend/rag_pipeline.py (root-level module, not inside app/)
    from rag_pipeline import RAGPipeline

    try:
        pipeline = RAGPipeline()
        result = pipeline.query(query=query, conversation_history=messages, update_memory=False)

        # Track cost from RAG pipeline token usage
        # RAGPipelineResult stores prompt/completion counts in result.metadata
        meta = result.metadata or {}
        prompt_tok = meta.get("prompt_tokens", 0)
        completion_tok = meta.get("completion_tokens", 0)
        if prompt_tok or completion_tok:
            usage = LLMUsage(
                model=settings.OPENAI_MODEL,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                total_tokens=prompt_tok + completion_tok,
                stage=ConversationStage.QA_GENERATING.value,
                intent=IntentType.GENERAL_QA.value,
            )
            usage.compute_cost()
            cost_log.add(usage)
            get_cost_tracker().log_usage(usage, session_id=session_id)

        # RAGPipelineResult uses .response, not .final_answer
        answer = result.response if result.success else (
            result.response or "I'm sorry, I couldn't find an answer. Please try rephrasing your question."
        )
        sources = [r.metadata.get("source", "") for r in (result.retrieval_results or []) if r.metadata]

        return {
            "type": "general_qa",
            "response": answer,
            "sources": sources,
        }
    except Exception as e:
        logger.error(f"General QA failed: {e}")
        return {
            "type": "general_qa",
            "response": "I encountered an error answering your question. Please try again.",
            "sources": [],
        }


async def _run_job_search(extracted_params: dict, query: str, session_id: str, cost_log: LLMCostLog) -> dict:
    """Run JobDiva job search and use LLM to answer naturally."""
    from app.tools.jobdiva import search_jobs, format_jobs_for_llm

    role = extracted_params.get("role")
    location = extracted_params.get("location")
    status = extracted_params.get("status")
    query_type = extracted_params.get("query_type")

    search_result = await search_jobs(role=role, location=location, status=status, query_type=query_type)
    jobs_text = format_jobs_for_llm(search_result)

    # Use LLM to compose a natural answer from job data
    model = settings.OPENAI_MODEL
    system_msg = """You are MHK Nova, MHK Tech's friendly AI assistant answering job-related questions.
You have structured job data (title, location, salary, skills, experience, apply link) provided below.
Use "we" and "our" when referring to MHK — you're part of the team
Guidelines — follow ALL of these:

1. APPLY LINKS — Always include the apply link () for any job you mention. Never omit it.
2. SKILLS / QUALIFICATIONS — When the user asks "what skills do I need?" or "am I qualified?" or
   "what qualifications does X require?" → list the exact skills from the job data clearly.
3. SALARY / COMPENSATION — When asked about pay, salary, rate, or compensation:
   - If salary is available → state the range and type (e.g. "$90,000–$120,000/year").
   - If not disclosed → say "The salary for this role is not publicly listed. Please reach out to hr@mhktechinc.com for compensation details."
4. EXPERIENCE — When asked about experience level, years needed, or seniority → state the requirement from the data.
5. REMOTE / LOCATION — When asked if a role is remote, hybrid, or on-site → answer based on the REMOTE field and location.
6. LISTING ALL JOBS — When the user asks "what jobs are available?" or similar → list all jobs concisely:
   one per line with: title, location, salary range, and apply link.
7. SPECIFIC JOB DETAIL — When the user asks about ONE specific role → give full detail:
   skills, experience, salary, remote, description, and apply link.
8. NO MATCH — If no jobs match the query → warmly say so and direct them to https://www.mhktechinc.com/careers
   and hr@mhktechinc.com for openings not yet listed.
9. TONE — Be warm, professional, and encouraging. End with an invitation to apply or ask more questions.

Format apply links like:
 **Apply here:** <URL>"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Job data:\n{jobs_text}\n\nUser question: {query}"},
    ]
    try:
        content, usage = _chat_completion(messages, model=model, temperature=0.4, max_tokens=900)
        llm_usage = _track_cost(usage, model, stage=ConversationStage.JOB_ANSWERING.value, intent=IntentType.JOB_SEARCH.value, session_id=session_id)
        cost_log.add(llm_usage)
    except Exception as e:
        logger.error(f"Job answer generation failed: {e}")
        content = jobs_text  # fallback: return raw formatted data

    return {"type": "job_search", "response": content, "raw_jobs": search_result.get("jobs", [])}


async def _run_meeting_scheduler(
    sub_intent: str, session_id: str, query: str, meeting_session_id: Optional[str]
) -> dict:
    """Run meeting scheduler state machine."""
    from app.tools.meeting_scheduler import handle_message

    scheduler_session_id = meeting_session_id or session_id
    response, updated_session = await handle_message(
        session_id=scheduler_session_id,
        user_message=query,
        sub_intent=sub_intent,
    )
    return {
        "type": "meeting_scheduler",
        "response": response,
        "meeting_session_id": scheduler_session_id,
        "meeting_state": updated_session.state,
    }


async def tool_selection_node(state: AgentState) -> dict:
    """
    Execute all detected intents concurrently (multi-intent support).
    Returns aggregated tool_results.
    LangGraph will properly await this when graph.ainvoke() is used.
    """
    intents = state.get("detected_intents", [])
    query = state["current_query"]
    session_id = state["session_id"]
    messages = state.get("messages", [])
    meeting_session_id = state.get("meeting_session_id")
    cost_log = LLMCostLog(session_id=session_id)

    tasks = []
    intent_types = []
    for intent_dict in intents:
        intent = intent_dict.get("intent")
        sub_intent = intent_dict.get("sub_intent", "general")
        params = intent_dict.get("extracted_params", {})

        if intent == IntentType.MEETING_SCHEDULER.value:
            tasks.append(_run_meeting_scheduler(sub_intent, session_id, query, meeting_session_id))
        elif intent == IntentType.JOB_SEARCH.value:
            tasks.append(_run_job_search(params, query, session_id, cost_log))
        else:
            tasks.append(_run_general_qa(query, session_id, messages, cost_log))
        intent_types.append(intent)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    tool_results = []
    new_meeting_session_id = meeting_session_id
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Tool {intent_types[i]} raised exception: {res}")
            tool_results.append({"type": intent_types[i], "response": "An error occurred. Please try again.", "error": str(res)})
        else:
            tool_results.append(res)
            if res.get("type") == "meeting_scheduler":
                # Clear session ID from state if completed so we don't carry ghosts
                if res.get("meeting_state") in ("completed", "initial"):
                    new_meeting_session_id = None
                else:
                    new_meeting_session_id = res.get("meeting_session_id", new_meeting_session_id)

    logger.info(f"[ToolNode] session={session_id} ran {len(tool_results)} tool(s)")

    return {
        "stage": ConversationStage.QA_COMPLETED.value,
        "tool_results": tool_results,
        "meeting_session_id": new_meeting_session_id,
        "cost_log": cost_log.to_dict(),
    }


# =============================================================================
# Node 4: Response Aggregation
# =============================================================================

def response_node(state: AgentState) -> dict:
    """
    Aggregate results from all tools into a single, coherent response.
    If only 1 tool ran, just pass through its response.
    If multiple tools ran, use LLM to weave them together gracefully.
    """
    tool_results = state.get("tool_results", [])
    session_id = state["session_id"]
    cost_log = LLMCostLog(session_id=session_id)

    if not tool_results:
        return {
            "final_response": "I'm sorry, I wasn't able to process your request. Please try again.",
            "stage": ConversationStage.ERROR_RECOVERABLE.value,
        }

    if len(tool_results) == 1:
        combined = tool_results[0].get("response", "")
    else:
        # Multi-intent: weave responses
        model = settings.OPENAI_MODEL
        responses_text = "\n\n---\n\n".join(
            f"[{r['type'].upper()}]\n{r.get('response','')}" for r in tool_results
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are MHK Tech's AI assistant. The user asked about multiple topics. "
                    "Combine the following separate answers into one smooth, coherent response. "
                    "Preserve all important details. Use natural transitions between topics."
                ),
            },
            {"role": "user", "content": responses_text},
        ]
        try:
            combined, usage = _chat_completion(messages, model=model, temperature=0.4, max_tokens=800)
            llm_usage = _track_cost(usage, model, stage="response_aggregation", intent="multi", session_id=session_id)
            cost_log.add(llm_usage)
        except Exception as e:
            logger.error(f"Response aggregation failed: {e}")
            combined = "\n\n".join(r.get("response", "") for r in tool_results)

    # Append middle-question reminder if applicable
    meeting_session_id = state.get("meeting_session_id")
    if meeting_session_id and not any(r.get("type") == "meeting_scheduler" for r in tool_results):
        from app.tools.meeting_scheduler import load_session
        sess = load_session(meeting_session_id)
        if sess and sess.state not in ("initial", "completed"):
            reminder = ""
            if sess.state == "awaiting_email":
                reminder = "provide your business email"
            elif sess.state == "awaiting_otp":
                reminder = "enter the 6-digit verification code"
            elif sess.state == "awaiting_duration":
                reminder = "choose a meeting duration (1, 2, or 3)"
            elif sess.state == "awaiting_slot":
                reminder = "choose a time slot (1, 2, or 3)"
            
            if reminder:
                combined += f"\n\n_*(Note: We are still in the middle of scheduling your meeting. Please {reminder} when you're ready.)*_"

    logger.info(f"[ResponseNode] session={session_id} aggregated {len(tool_results)} responses")
    return {
        "final_response": combined,
        "stage": ConversationStage.IDLE.value,
        "cost_log": cost_log.to_dict(),
    }


# =============================================================================
# Node 5: Memory Update
# =============================================================================

def memory_update_node(state: AgentState) -> dict:
    """Append this turn's messages to the conversation history."""
    messages = list(state.get("messages", []))
    query = state.get("current_query", "")
    response = state.get("final_response", "")

    if query:
        messages.append({"role": "user", "content": query})
    if response:
        messages.append({"role": "assistant", "content": response})

    # Keep last N turns
    max_history = settings.MAX_CONVERSATION_HISTORY * 2  # each turn = 2 messages
    if len(messages) > max_history:
        messages = messages[-max_history:]

    # Log total turn cost
    cost_log_dict = state.get("cost_log", {})
    if cost_log_dict:
        total = cost_log_dict.get("total_cost_usd", 0)
        total_tokens = cost_log_dict.get("total_tokens", 0)
        logger.info(
            f"[MemoryNode] session={state['session_id']} "
            f"turn_tokens={total_tokens} turn_cost=${total:.6f}"
        )
        get_cost_tracker().log_turn_cost(
            type("CL", (), {
                "session_id": state["session_id"],
                "total_tokens": total_tokens,
                "total_cost_usd": total,
                "calls": [],
            })()
        )

    return {"messages": messages}