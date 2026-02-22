"""
Unit tests for agent intent detection node.
Tests that specific queries map to correct intents with sufficient confidence.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.agent.state import IntentType, SubIntentType, ConversationStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_openai_response(intents: list) -> MagicMock:
    """Build a mock OpenAI response with given intents JSON."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = json.dumps(intents)
    mock_resp.usage.prompt_tokens = 100
    mock_resp.usage.completion_tokens = 50
    mock_resp.usage.total_tokens = 150
    return mock_resp


def run_intent_detection(query: str, mock_intents: list) -> dict:
    """Run intent_detection_node with a mocked OpenAI call."""
    from app.agent.nodes import intent_detection_node
    state = {
        "current_query": query,
        "session_id": "test-session",
        "messages": [],
        "stage": ConversationStage.INTENT_DETECTION.value,
        "detected_intents": [],
        "active_intent_index": 0,
        "tool_results": [],
        "meeting_session_id": None,
        "cost_log": {},
        "awaiting_clarification": False,
        "clarification_turns": 0,
        "error_message": None,
        "final_response": "",
    }
    with patch("app.agent.nodes._chat_completion") as mock_chat:
        mock_chat.return_value = (json.dumps(mock_intents), {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        with patch("app.agent.nodes.get_cost_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()
            return intent_detection_node(state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIntentDetection:

    def test_meeting_scheduler_quick_call(self):
        """'Can we have a quick call?' → meeting_scheduler / quick_call"""
        result = run_intent_detection(
            "Can we have a quick call this week?",
            [{"intent": "meeting_scheduler", "sub_intent": "quick_call", "confidence": 0.95, "extracted_params": {}}]
        )
        assert result["awaiting_clarification"] is False
        intents = result["detected_intents"]
        assert len(intents) == 1
        assert intents[0]["intent"] == IntentType.MEETING_SCHEDULER.value
        assert intents[0]["sub_intent"] == SubIntentType.QUICK_CALL.value
        assert intents[0]["confidence"] >= 0.85

    def test_job_search_by_location(self):
        """'Data Engineer jobs in Texas' → job_search / search_by_location"""
        result = run_intent_detection(
            "Are there any Data Engineer openings in Texas?",
            [{"intent": "job_search", "sub_intent": "search_by_location", "confidence": 0.93,
              "extracted_params": {"role": "Data Engineer", "location": "Texas"}}]
        )
        assert result["awaiting_clarification"] is False
        intents = result["detected_intents"]
        assert intents[0]["intent"] == IntentType.JOB_SEARCH.value
        assert intents[0]["extracted_params"]["role"] == "Data Engineer"
        assert intents[0]["extracted_params"]["location"] == "Texas"

    def test_general_qa(self):
        """'When was MHK founded?' → general_qa"""
        result = run_intent_detection(
            "When was MHK Tech Inc. founded?",
            [{"intent": "general_qa", "sub_intent": "general", "confidence": 0.98, "extracted_params": {}}]
        )
        assert result["awaiting_clarification"] is False
        assert result["detected_intents"][0]["intent"] == IntentType.GENERAL_QA.value

    def test_multi_intent_meeting_and_qa(self):
        """'What services does MHK offer AND schedule a meeting' → 2 intents"""
        result = run_intent_detection(
            "What services does MHK offer? Also, I want to schedule a meeting.",
            [
                {"intent": "general_qa", "sub_intent": "general", "confidence": 0.95, "extracted_params": {}},
                {"intent": "meeting_scheduler", "sub_intent": "normal_meet", "confidence": 0.91, "extracted_params": {}},
            ]
        )
        assert result["awaiting_clarification"] is False
        assert len(result["detected_intents"]) == 2

    def test_low_confidence_triggers_clarification(self):
        """Low-confidence intent → awaiting_clarification = True"""
        result = run_intent_detection(
            "I need something...",
            [{"intent": "general_qa", "sub_intent": "general", "confidence": 0.60, "extracted_params": {}}]
        )
        assert result["awaiting_clarification"] is True

    def test_list_all_jobs(self):
        """'What openings do you have?' → job_search / list_all_jobs"""
        result = run_intent_detection(
            "What job openings are currently available at MHK Tech?",
            [{"intent": "job_search", "sub_intent": "list_all_jobs", "confidence": 0.92, "extracted_params": {}}]
        )
        assert result["detected_intents"][0]["intent"] == IntentType.JOB_SEARCH.value
        assert result["detected_intents"][0]["sub_intent"] == SubIntentType.LIST_ALL_JOBS.value

    def test_long_discussion_intent(self):
        """'I need a 1-hour deep dive' → meeting_scheduler / long_discussion"""
        result = run_intent_detection(
            "Can we set up a long deep-dive session, maybe an hour or so?",
            [{"intent": "meeting_scheduler", "sub_intent": "long_discussion", "confidence": 0.89, "extracted_params": {}}]
        )
        assert result["detected_intents"][0]["sub_intent"] == SubIntentType.LONG_DISCUSSION.value
