import sys
import os

# Ensure backend directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

import asyncio
from app.tools.meeting_scheduler import check_daily_meeting_limit, handle_message

async def main():
    print("Beginning Rate Limit Test")
    email = "test.ratelimit@mhktechinc.com"
    session_id = "test_sess_123"
    
    # In initial state, user says "schedule"
    res_msg, session = await handle_message(session_id, "schedule a meeting", sub_intent="normal_meet")
    # Now in awaiting_email state, user provides email
    res_msg, session = await handle_message(session_id, email, sub_intent="normal_meet")
    print(f"Response: {res_msg}")
    
    # Simulate finishing first meeting flow (resetting state to initial for the same email)
    # The limit check already happens when they enter the email, which increments the counter.
    
    # 2. Second attempt - should be allowed
    print("\n[Attempt 2]")
    session_id2 = "test_sess_456"
    res_msg2, session2 = await handle_message(session_id2, "schedule", sub_intent="normal_meet")
    res_msg2, session2 = await handle_message(session_id2, email, sub_intent="normal_meet")
    print(f"Response: {res_msg2}")
    
    # 3. Third attempt - should be BLOCKED by rate limit
    print("\n[Attempt 3]")
    session_id3 = "test_sess_789"
    res_msg3, session3 = await handle_message(session_id3, "schedule", sub_intent="normal_meet")
    res_msg3, session3 = await handle_message(session_id3, email, sub_intent="normal_meet")
    print(f"Response: {res_msg3}")
    
    # Verify the limit check function directly 
    print("\n[Direct Check]")
    allowed = check_daily_meeting_limit(email)
    print(f"Is 4th attempt allowed?: {allowed}")

if __name__ == "__main__":
    asyncio.run(main())