import asyncio
from app.agent.nodes import AgentState, intent_detection_node, tool_selection_node, response_node

async def run():
    state = {
        "session_id": "test_persona",
        "current_query": "What is MHK Nova?",
        "messages": []
    }
    
    # 1. Intent Node
    res1 = intent_detection_node(state)
    state.update(res1)
    
    # 2. Tool Node
    res2 = await tool_selection_node(state)
    state.update(res2)

    # 3. Response Node
    res3 = response_node(state)
    print("FINAL RESPONSE:", res3["final_response"])

asyncio.run(run())