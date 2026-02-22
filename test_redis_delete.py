from app.tools.meeting_scheduler import MeetingSession, save_session, delete_session, load_session

sess = MeetingSession(session_id="test_del", state="initial")
save_session(sess)
print("Saved:", load_session("test_del"))
delete_session("test_del")
print("Deleted:", load_session("test_del"))
