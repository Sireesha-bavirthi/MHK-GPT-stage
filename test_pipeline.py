import asyncio
from app.agent.nodes import _run_job_search, LLMCostLog
from app.core.config import settings

async def test():
    cost = LLMCostLog(session_id="test")
    extracted_params = {"role": "Data Engineer", "query_type": "job_detail"}
    query = "can you explain Data Engineer role"
    res = await _run_job_search(extracted_params, query, "test", cost)
    print("Result:", res)
    
if __name__ == "__main__":
    asyncio.run(test())
