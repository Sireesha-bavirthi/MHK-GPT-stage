import asyncio
from app.agent.nodes import _chat_completion, INTENT_SYSTEM_PROMPT
from app.core.config import settings
import json

async def test():
    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": "can you explain Data Engineer role"}
    ]
    content, _ = _chat_completion(messages, model=settings.OPENAI_MODEL, temperature=0.1, max_tokens=512)
    print("Raw Output:", content)
    
if __name__ == "__main__":
    asyncio.run(test())
