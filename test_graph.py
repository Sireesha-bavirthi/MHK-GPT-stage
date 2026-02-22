import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from app.agent.graph import get_agent_graph
from app.agent.state import AgentState

async def run_test():
    graph = get_agent_graph()
    
    state = {
        "current_query": "what is Nova?",
        "session_id": "test_session_graph",
        "messages": [],
        "clarification_turns": 0
    }

    print("Running graph for 'what is Nova?'")
    result = await graph.ainvoke(state)
    print("\nResult:")
    print(result.get("final_response"))
    print("\nStage:")
    print(result.get("stage"))

    # Now with history
    state2 = {
        "current_query": "what is MHK Nova?",
        "session_id": "test_session_graph",
        "messages": [
            {"role": "user", "content": "what is Nova?"},
            {"role": "assistant", "content": result.get("final_response", "")}
        ],
        "clarification_turns": 1
    }

    print("\nRunning graph for 'what is MHK Nova?' with history")
    result2 = await graph.ainvoke(state2)
    print("\nResult:")
    print(result2.get("final_response"))
    print("\nStage:")
    print(result2.get("stage"))

    state3 = {
        "current_query": "who are you?",
        "session_id": "test_session_graph",
        "messages": [
            {"role": "user", "content": "what is Nova?"},
            {"role": "assistant", "content": result.get("final_response", "")},
            {"role": "user", "content": "what is MHK Nova?"},
            {"role": "assistant", "content": result2.get("final_response", "")}
        ],
        "clarification_turns": 2
    }

    print("\nRunning graph for 'who are you?' with history")
    result3 = await graph.ainvoke(state3)
    print("\nResult:")
    print(result3.get("final_response"))
    print("\nStage:")
    print(result3.get("stage"))

if __name__ == "__main__":
    asyncio.run(run_test())
