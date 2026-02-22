"""
Meeting Scheduler schemas and session data models.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class TimeSlot:
    """A bookable time slot."""
    start: str          # ISO 8601
    end: str            # ISO 8601
    label: str          # Human-readable, e.g. "Fri, Feb 21, 2026, 10:00 AM"


@dataclass
class MeetingSession:
    """
    Mirrors the n8n Google Sheets session state.
    Stored in Redis keyed by session_id, TTL 15 minutes.
    """
    session_id: str
    state: str = "initial"          # Maps to ConversationStage scheduler states
    email: str = ""
    otp: str = ""
    otp_expiry: str = ""            # Unix ms timestamp as string
    otp_attempts: int = 0
    created_at: str = ""            # Unix ms timestamp as string
    slots: List[TimeSlot] = field(default_factory=list)
    selected_slot: Optional[TimeSlot] = None
    sub_intent: str = "normal_meet" # quick_call | normal_meet | long_discussion
    calendly_url: str = ""
    calendar_event_link: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "state": self.state,
            "email": self.email,
            "otp": self.otp,
            "otp_expiry": self.otp_expiry,
            "otp_attempts": self.otp_attempts,
            "created_at": self.created_at,
            "slots": [{"start": s.start, "end": s.end, "label": s.label} for s in self.slots],
            "selected_slot": {
                "start": self.selected_slot.start,
                "end": self.selected_slot.end,
                "label": self.selected_slot.label
            } if self.selected_slot else None,
            "sub_intent": self.sub_intent,
            "calendly_url": self.calendly_url,
            "calendar_event_link": self.calendar_event_link,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MeetingSession":
        session = cls(session_id=data["session_id"])
        session.state = data.get("state", "initial")
        session.email = data.get("email", "")
        session.otp = data.get("otp", "")
        session.otp_expiry = data.get("otp_expiry", "")
        session.otp_attempts = int(data.get("otp_attempts", 0))
        session.created_at = data.get("created_at", "")
        session.sub_intent = data.get("sub_intent", "normal_meet")
        session.calendly_url = data.get("calendly_url", "")
        session.calendar_event_link = data.get("calendar_event_link", "")
        slots_data = data.get("slots", [])
        session.slots = [
            TimeSlot(start=s["start"], end=s["end"], label=s["label"])
            for s in slots_data if isinstance(s, dict)
        ]
        sel = data.get("selected_slot")
        if sel:
            session.selected_slot = TimeSlot(start=sel["start"], end=sel["end"], label=sel["label"])
        return session
