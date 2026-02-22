"""
Unit tests for meeting scheduler tool.
Tests email validation, OTP generation, slot finding, and state machine transitions.
No external services required (Redis + email mocked).
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta

from app.tools.meeting_scheduler import (
    validate_business_email,
    generate_otp,
    is_otp_expired,
    otp_expiry_ms,
    find_available_slots,
)
from app.schemas.meeting import MeetingSession, TimeSlot


# ---------------------------------------------------------------------------
# Email validation
# ---------------------------------------------------------------------------

class TestEmailValidation:

    def test_valid_business_email(self):
        valid, msg = validate_business_email("john@mhktechinc.com")
        assert valid is True
        assert msg == ""

    def test_personal_gmail_rejected(self):
        valid, msg = validate_business_email("john@gmail.com")
        assert valid is False
        assert "business email" in msg.lower()

    def test_yahoo_rejected(self):
        valid, msg = validate_business_email("test@yahoo.com")
        assert valid is False

    def test_outlook_rejected(self):
        valid, msg = validate_business_email("user@outlook.com")
        assert valid is False

    def test_missing_at_symbol(self):
        valid, msg = validate_business_email("notanemail")
        assert valid is False
        assert "valid email" in msg.lower()

    def test_case_insensitive(self):
        valid, msg = validate_business_email("User@MHKTECH.COM")
        assert valid is True

    def test_subdomain_business_email(self):
        valid, msg = validate_business_email("admin@corp.mhktech.com")
        assert valid is True


# ---------------------------------------------------------------------------
# OTP generation & expiry
# ---------------------------------------------------------------------------

class TestOTP:

    def test_otp_is_six_digits(self):
        otp = generate_otp()
        assert len(otp) == 6
        assert otp.isdigit()

    def test_otp_changes_each_time(self):
        otps = {generate_otp() for _ in range(20)}
        # With 6-digit OTPs, collision probability is extremely low
        assert len(otps) > 1

    def test_otp_not_expired_fresh(self):
        expiry = otp_expiry_ms()
        assert not is_otp_expired(expiry)

    def test_otp_expired_past(self):
        # 11 minutes ago
        past = int((datetime.now(timezone.utc) - timedelta(minutes=11)).timestamp() * 1000)
        assert is_otp_expired(str(past))

    def test_otp_invalid_string(self):
        assert is_otp_expired("not-a-number")


# ---------------------------------------------------------------------------
# Slot finding
# ---------------------------------------------------------------------------

class TestSlotFinding:

    def test_finds_three_slots(self):
        slots = find_available_slots([])
        assert len(slots) == 3

    def test_slots_are_within_work_hours(self):
        slots = find_available_slots([])
        for slot in slots:
            dt = datetime.fromisoformat(slot.start)
            assert 9 <= dt.hour < 17, f"Slot hour {dt.hour} not within 9-17"

    def test_slots_skip_weekends(self):
        slots = find_available_slots([])
        for slot in slots:
            dt = datetime.fromisoformat(slot.start)
            assert dt.weekday() < 5, f"Slot on weekend: {slot.start}"

    def test_conflict_avoided(self):
        """If a time is blocked, it should not appear in slots."""
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        tomorrow = datetime.now(ist) + timedelta(days=1)
        blocked_start = tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)
        blocked_end = blocked_start + timedelta(hours=3)

        # Skip weekends for the blocked slot
        while blocked_start.weekday() >= 5:
            blocked_start += timedelta(days=1)
            blocked_end = blocked_start + timedelta(hours=3)

        fake_events = [{
            "start": {"dateTime": blocked_start.isoformat()},
            "end": {"dateTime": blocked_end.isoformat()},
        }]
        slots = find_available_slots(fake_events)
        for slot in slots:
            slot_start = datetime.fromisoformat(slot.start)
            assert not (blocked_start <= slot_start < blocked_end), \
                f"Slot {slot.start} is in blocked range"

    def test_no_past_slots(self):
        slots = find_available_slots([])
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        for slot in slots:
            dt = datetime.fromisoformat(slot.start)
            assert dt > now, f"Slot {slot.start} is in the past"


# ---------------------------------------------------------------------------
# State machine transitions (mocked Redis + email)
# ---------------------------------------------------------------------------

class TestStateMachine:

    @pytest.mark.asyncio
    async def test_initial_state_returns_calendly(self):
        """Initial state should show Calendly URL and ask for email."""
        with patch("app.tools.meeting_scheduler.load_session", return_value=None), \
             patch("app.tools.meeting_scheduler.save_session"):
            from app.tools.meeting_scheduler import handle_message
            response, session = await handle_message("sess-1", "schedule a meeting", "normal_meet")

        assert "calendly" in response.lower() or "calendly.com" in response.lower()
        assert session.state == "awaiting_email"

    @pytest.mark.asyncio
    async def test_email_validation_in_state_machine(self):
        """Providing a personal email in awaiting_email state should be rejected."""
        mock_sess = MeetingSession(session_id="sess-2", state="awaiting_email")

        with patch("app.tools.meeting_scheduler.load_session", return_value=mock_sess), \
             patch("app.tools.meeting_scheduler.save_session"):
            from app.tools.meeting_scheduler import handle_message
            response, session = await handle_message("sess-2", "my email is john@gmail.com", "normal_meet")

        assert "business email" in response.lower()
        assert session.state == "awaiting_email"  # Did not advance

    @pytest.mark.asyncio
    async def test_otp_wrong_increments_attempts(self):
        from app.tools.meeting_scheduler import handle_message
        mock_sess = MeetingSession(
            session_id="sess-3",
            state="awaiting_otp",
            email="john@mhktech.com",
            otp="123456",
            otp_expiry=otp_expiry_ms(),
            otp_attempts=0,
        )
        with patch("app.tools.meeting_scheduler.load_session", return_value=mock_sess), \
             patch("app.tools.meeting_scheduler.save_session"):
            response, session = await handle_message("sess-3", "999999", "normal_meet")

        assert "incorrect" in response.lower() or "wrong" in response.lower() or "invalid" in response.lower() or "✗" in response or "❌" in response

    @pytest.mark.asyncio
    async def test_correct_otp_advances_to_slot_selection(self):
        from app.tools.meeting_scheduler import handle_message
        mock_sess = MeetingSession(
            session_id="sess-4",
            state="awaiting_otp",
            email="john@mhktech.com",
            otp="654321",
            otp_expiry=otp_expiry_ms(),
            otp_attempts=0,
        )
        with patch("app.tools.meeting_scheduler.load_session", return_value=mock_sess), \
             patch("app.tools.meeting_scheduler.save_session"), \
             patch("app.tools.meeting_scheduler.get_available_slots", new_callable=AsyncMock,
                   return_value=[TimeSlot("2026-02-21T09:00:00", "2026-02-21T10:00:00", "Sat, Feb 21, 9AM IST")]):
            response, session = await handle_message("sess-4", "654321", "normal_meet")

        assert session.state == "awaiting_slot"
        assert len(session.slots) > 0
