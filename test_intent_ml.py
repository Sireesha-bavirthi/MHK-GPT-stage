import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from app.agent.nodes import intent_detection_node

state = {
    "current_query": "what is Machine Learning",
    "session_id": "test_session",
    "messages": []
}

res1 = intent_detection_node(state)
print("Result for 'what is Machine Learning':", res1)

state["current_query"] = "what is Machine Learning?"
res2 = intent_detection_node(state)
print("Result for 'what is Machine Learning?':", res2)
