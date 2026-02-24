import asyncio
import json
from app.agent.nodes import AgentState, intent_detection_node, tool_selection_node, response_node, memory_update_node

async def run_chat():
    state = {
        "session_id": "test_memory_router_01",
        "current_query": "I want to schedule a meeting",
        "messages": [],
        "cost_log": {}
    }


    queries = [
        "schedule a meet",
        "what is ML?",  # LLM CHAT test
        "mhktestuser1@mhktechinc.com", # Valid email
        "2 hours", # Invalid duration test
        "never mind" # Explicit cancellation test
    ]

    for q in queries:
        print(f"\n>> USER: {q}")
        state["current_query"] = q

        # Intent
        res1 = intent_detection_node(state)
        state.update(res1)
        
        # Tool
        res2 = await tool_selection_node(state)
        state.update(res2)

        # Response
        res3 = response_node(state)
        state.update(res3)
        print(f"<< BOT: {state['final_response']}")

        # Memory Update
        res4 = memory_update_node(state)
        state.update(res4)
        print(f"   [Memory Msg Count]: {len(state['messages'])}")
        if state.get("summary"):
            print(f"   [Memory Summary]: {state['summary']}")

asyncio.run(run_chat())