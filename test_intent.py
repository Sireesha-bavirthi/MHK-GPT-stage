import os
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from app.agent.nodes import intent_detection_node

state = {
    "current_query": "who are you?",
    "session_id": "test_session",
    "messages": [
        {"role": "user", "content": "what is Nova?"},
        {"role": "assistant", "content": "I'm not sure I understood that. Could you rephrase or tell me what you'd like help with? (e.g. \"show me open jobs\", \"schedule a meeting\", or ask me anything about MHK Tech)"},
        {"role": "user", "content": "what is MHK Nova?"},
        {"role": "assistant", "content": "I'm not sure I understood that. Could you rephrase or tell me what you'd like help with? (e.g. \"show me open jobs\", \"schedule a meeting\", or ask me anything about MHK Tech)"}
    ]
}

try:
    res = intent_detection_node(state)
    print("\nResult for 'who are you?' WITH history:")
    print(res)

except Exception as e:
    print(f"Error: {e}")
