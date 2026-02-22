"""
LangGraph StateGraph assembly for MHK-GPT v2 agentic system.
Wires all nodes together with conditional routing for multi-intent support.
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from app.agent.state import AgentState, ConversationStage
from app.agent.nodes import (
    intent_detection_node,
    clarification_node,
    tool_selection_node,
    response_node,
    memory_update_node,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> Literal["clarify", "tools"]:
    """After intent detection: go to clarification or tools."""
    if state.get("awaiting_clarification"):
        turns = state.get("clarification_turns", 0)
        if turns < settings.MAX_CLARIFICATION_TURNS:
            return "clarify"
    return "tools"


def route_after_clarification(state: AgentState) -> Literal["intent", "end"]:
    """
    After clarification response is generated:
    - If still clarifying → return response to user (END for this turn)
    - Graph resumes on user's next message
    """
    return "end"


def route_after_tools(state: AgentState) -> Literal["response", "end"]:
    """After tool execution, always go to response aggregation."""
    return "response"


# ---------------------------------------------------------------------------
# Build compiled graph
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.
    Returns a compiled graph ready for .invoke().
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("intent_detection", intent_detection_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("tool_selection", tool_selection_node)
    graph.add_node("response", response_node)
    graph.add_node("memory_update", memory_update_node)

    # Entry point
    graph.set_entry_point("intent_detection")

    # Conditional edge: after intent detection
    graph.add_conditional_edges(
        "intent_detection",
        route_after_intent,
        {
            "clarify": "clarification",
            "tools": "tool_selection",
        },
    )

    # After clarification → end this turn (response already set in state)
    graph.add_edge("clarification", "memory_update")

    # Tools → response → memory → END
    graph.add_edge("tool_selection", "response")
    graph.add_edge("response", "memory_update")
    graph.add_edge("memory_update", END)

    compiled = graph.compile()
    logger.info("LangGraph agent compiled successfully")
    return compiled


# Singleton compiled graph
_agent_graph = None


def get_agent_graph():
    """Get or build the singleton compiled agent graph."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph
