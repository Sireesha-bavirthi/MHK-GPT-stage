"""
Meeting Scheduler Tool.

State machine:
  initial → awaiting_email → awaiting_otp → awaiting_duration → awaiting_slot → completed

Flow:
  1. Ask for business email
  2. Domain block-list check → Abstract API deliverability check
  3. Send OTP to email; verify (max 3 attempts)
  4. AFTER verification: offer 3 duration options (15 min / 30 min / 1 hour)
  5. Show available calendar slots for chosen duration
  6. Confirm slot → create Google Calendar event → show join link
     (Calendly link only as last-resort fallback if Google Calendar unavailable)

Email delivery: SMTP Gmail (primary) → SendGrid (fallback)
Calendar: Google Calendar API via service account
Session: Redis, 15-min TTL
"""

import json
import logging
import random
import string
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple

import httpx

from app.core.config import settings
from app.schemas.meeting import MeetingSession, TimeSlot

logger = logging.getLogger(__name__)

# Personal email providers — always block these
_PERSONAL_PROVIDERS = frozenset({
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com",
    "yahoo.co.in", "rediffmail.com", "live.com", "msn.com", "googlemail.com",
    "aol.com", "protonmail.com", "me.com", "mac.com", "ymail.com",
    "zoho.com", "tutanota.com", "mail.com", "inbox.com", "yandex.com"
})
# Temporary disposable email domains
_DISPOSABLE_DOMAINS = frozenset({
    "mailinator.com","guerrillamail.com","sharklasers.com","10minutemail.com",
    "temp-mail.org","yopmail.com","maildrop.cc","getnada.com","throwawaymail.com",
    "dispostable.com"
})
SESSION_TTL = 15 * 60    # 15 minutes
OTP_EXPIRY  = 10 * 60    # 10 minutes

# Duration map: user choice → (sub_intent, label)
DURATION_OPTIONS = {
    "1": ("quick_call",      "15 minutes"),
    "2": ("normal_meet",     "30 minutes"),
    "3": ("long_discussion", "1 hour"),
}

DURATION_MINUTES = {
    "quick_call":      15,
    "normal_meet":     30,
    "long_discussion": 60,
}

CALENDLY_URLS = {
    "quick_call":      settings.CALENDLY_15MIN_URL,
    "normal_meet":     settings.CALENDLY_30MIN_URL,
    "long_discussion": settings.CALENDLY_60MIN_URL,
}

# =============================================================================
# Redis helpers
# =============================================================================

def _redis():
    try:
        import redis as r
        client = r.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable in meeting scheduler: {e}")
        return None


def load_session(session_id: str) -> Optional[MeetingSession]:
    client = _redis()
    if not client:
        return None
    raw = client.get(f"meeting:{session_id}")
    if not raw:
        return None
    try:
        return MeetingSession.from_dict(json.loads(raw))
    except Exception as e:
        logger.error(f"Failed to deserialise meeting session: {e}")
        return None


def save_session(session: MeetingSession) -> None:
    client = _redis()
    if not client:
        return
    try:
        client.setex(
            f"meeting:{session.session_id}",
            SESSION_TTL,
            json.dumps(session.to_dict()),
        )
    except Exception as e:
        logger.error(f"Failed to save meeting session: {e}")


def delete_session(session_id: str) -> None:
    client = _redis()
    if client:
        client.delete(f"meeting:{session_id}")


# =============================================================================
# Email sending (SMTP primary, SendGrid fallback)
# =============================================================================

async def _send_via_smtp(to_email: str, subject: str, html_body: str) -> bool:
    try:
        import aiosmtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.SMTP_FROM_EMAIL
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))

        await aiosmtplib.send(
            msg,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USERNAME,
            password=settings.SMTP_PASSWORD,
            use_tls=False,
            start_tls=True,
        )
        logger.info(f"OTP email sent via SMTP to {to_email}")
        return True
    except Exception as e:
        logger.error(f"SMTP send failed: {e}")
        return False


async def _send_via_sendgrid(to_email: str, subject: str, html_body: str) -> bool:
    if not settings.SENDGRID_API_KEY:
        return False
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": settings.SENDGRID_FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/html", "value": html_body}],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                json=payload,
                headers={"Authorization": f"Bearer {settings.SENDGRID_API_KEY}"},
            )
            resp.raise_for_status()
        logger.info(f"Email sent via SendGrid to {to_email}")
        return True
    except Exception as e:
        logger.error(f"SendGrid send failed: {e}")
        return False


async def _send_email(to_email: str, subject: str, html_body: str) -> bool:
    """Try SMTP first (App Password set), then SendGrid."""
    if settings.SMTP_PASSWORD:
        if await _send_via_smtp(to_email, subject, html_body):
            return True
    return await _send_via_sendgrid(to_email, subject, html_body)


async def send_otp_email(to_email: str, otp: str) -> bool:
    html_body = f"""
<div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:24px;border:1px solid #e5e7eb;border-radius:8px;">
  <h2 style="color:#1e293b;margin-bottom:8px;"> Meeting Verification Code</h2>
  <p style="color:#475569;">Use the code below to verify your email and proceed with scheduling your meeting with <strong>MHK Tech</strong>:</p>
  <div style="background:#f1f5f9;padding:28px;margin:20px 0;text-align:center;border-radius:6px;">
    <span style="color:#6b46c1;font-size:42px;font-weight:700;letter-spacing:12px;">{otp}</span>
  </div>
  <p style="color:#475569;">This code expires in <strong>10 minutes</strong>. Do not share it with anyone.</p>
  <p style="color:#94a3b8;font-size:12px;">If you didn't request this, you can safely ignore this email.</p>
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0;">
  <p style="color:#94a3b8;font-size:11px;">MHK Tech Inc. | hr@mhktechinc.com</p>
</div>
"""
    return await _send_email(to_email, "Your MHK Tech Meeting Verification Code", html_body)


async def send_confirmation_email(to_email: str, slot_label: str, meet_link: str) -> bool:
    html_body = f"""
<div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:24px;border:1px solid #e5e7eb;border-radius:8px;">
  <h2 style="color:#1e293b;"> Meeting Confirmed!</h2>
  <p style="color:#475569;">Your meeting with <strong>MHK Tech</strong> has been scheduled:</p>
  <div style="background:#f0fdf4;padding:16px;border-radius:6px;border-left:4px solid #22c55e;margin:16px 0;">
    <strong style="color:#15803d;"> {slot_label}</strong>
  </div>
  <p style="color:#475569;"><strong>Join link:</strong><br>
    <a href="{meet_link}" style="color:#6b46c1;">{meet_link}</a>
  </p>
  <p style="color:#94a3b8;font-size:12px;">A calendar invite has been sent to {to_email}.</p>
  <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0;">
  <p style="color:#94a3b8;font-size:11px;">MHK Tech Inc. | hr@mhktechinc.com</p>
</div>
"""
    return await _send_email(to_email, "Meeting Confirmed — MHK Tech", html_body)


# =============================================================================
# Rate Limiting (Daily Meeting Limits)
# =============================================================================

def check_daily_meeting_limit(email: str, max_meetings: int = 2) -> bool:
    """
    Check if the user has reached the daily limit for scheduling meetings.
    Uses Redis to store the count. TTL is set to 24 hours.
    Returns True if allowed, False if limit reached.
    """
    client = _redis()
    if not client:
         # If Redis is down, fail open to avoid totally breaking the system
        return True
        
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    key = f"meeting_limit:{email.lower()}:{today}"
    
    try:
        current_count = client.get(key)
        if current_count and int(current_count) >= max_meetings:
            logger.info(f"Rate limit hit for meeting scheduling: {email} ({current_count}/{max_meetings})")
            return False
            
        # Increment token and set TTL to 24 hours if it's a new key
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 24 * 60 * 60) # 24 hours
        pipe.execute()
        
        return True
    except Exception as e:
        logger.error(f"Failed to check daily limit in Redis: {e}")
        return True # Fail open

# =============================================================================
# Email Validation
# =============================================================================

def validate_business_email(email: str) -> Tuple[bool, str]:
    """Step 1: Basic format + domain block-list."""
    import re
    email = email.strip().lower()
    if not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email):
        return False, "That doesn't look like a valid email address. Please try again."
    domain = email.split("@")[1]
    if domain in _PERSONAL_PROVIDERS:
        return False, (
            f"Please use your **business email** — personal addresses like **@{domain}** are not accepted."
        )
    if domain in _DISPOSABLE_DOMAINS:
        return False, "Temporary or disposable email addresses are not allowed."
    return True, ""


async def validate_email_with_abstract(email: str) -> Tuple[bool, str]:
    """Step 2: Abstract Email Reputation API deliverability check. Fails open on API error."""
    api_key = settings.ABSTRACT_EMAIL_API_KEY
    if not api_key:
        return True, ""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "https://emailreputation.abstractapi.com/v1/",
                params={"api_key": api_key, "email": email},
            )
            resp.raise_for_status()
            data = resp.json()

        deliverability = data.get("email_deliverability", {})
        domain_info = data.get("email_domain", {})

        status = deliverability.get("status", "").lower()
        is_smtp_valid = deliverability.get("is_smtp_valid", True)
        is_free_provider = domain_info.get("is_free_email_provider", False)

        logger.info(
            f"Abstract email reputation for {email}: "
            f"status={status}, smtp_valid={is_smtp_valid}, free_provider={is_free_provider}"
        )

        # Block free/personal email providers caught by the API (catches typos like gamil.com too)
        if is_free_provider:
            domain = email.split("@")[1]
            return False, (
                f"Please use your **business email** — free email providers like **@{domain}** "
                f"are not accepted."
            )

        # Block undeliverable addresses
        if status == "undeliverable" or not is_smtp_valid:
            return False, (
                f"The email **{email}** appears to be invalid or undeliverable. "
                f"Please double-check and try again."
            )

        return True, ""

    except Exception as e:
        logger.warning(f"Abstract API check failed ({e}) — allowing through")
        return True, ""


# =============================================================================
# OTP Helpers
# =============================================================================

def generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


def otp_expiry_ms() -> str:
    return str(int((datetime.now(timezone.utc) + timedelta(seconds=OTP_EXPIRY)).timestamp() * 1000))


def is_otp_expired(expiry_ms_str: str) -> bool:
    try:
        expiry = int(expiry_ms_str)
        return int(datetime.now(timezone.utc).timestamp() * 1000) > expiry
    except Exception:
        return True


# =============================================================================
# Google Calendar Integration
# =============================================================================

def _build_calendar_service():
    """
    Build Google Calendar service from service account.
    Supports:
      - GOOGLE_SERVICE_ACCOUNT_JSON as a file path (e.g. google_service_account.json)
      - GOOGLE_SERVICE_ACCOUNT_JSON as an inline JSON string (starts with '{')
    """
    raw = settings.GOOGLE_SERVICE_ACCOUNT_JSON
    if not raw or not raw.strip():
        return None
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        raw = raw.strip()
        if raw.startswith("{"):
            sa_info = json.loads(raw)
            creds = service_account.Credentials.from_service_account_info(
                sa_info, scopes=["https://www.googleapis.com/auth/calendar"]
            )
        else:
            # Resolve relative paths relative to the backend/ directory
            if not os.path.isabs(raw):
                raw = os.path.join(os.path.dirname(__file__), "..", "..", raw)
            creds = service_account.Credentials.from_service_account_file(
                raw, scopes=["https://www.googleapis.com/auth/calendar"]
            )

        # Delegate to the calendar owner so events appear on their calendar
        calendar_id = settings.GOOGLE_CALENDAR_ID
        if calendar_id and "@" in calendar_id:
            creds = creds.with_subject(calendar_id)

        return build("calendar", "v3", credentials=creds)
    except Exception as e:
        logger.warning(f"Google Calendar service build failed: {e}")
        return None


def find_available_slots(existing_events: list, duration_minutes: int = 30) -> List[TimeSlot]:
    """Find next 3 available weekday slots (9am–5pm IST)."""
    from zoneinfo import ZoneInfo

    ist = ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    slots: List[TimeSlot] = []
    WORK_START, WORK_END = 9, 17
    slot_delta = timedelta(minutes=duration_minutes)

    for day_offset in range(10):
        if len(slots) >= 3:
            break
        check_day = now + timedelta(days=day_offset)
        if check_day.weekday() >= 5:
            continue

        start_hour = WORK_START if day_offset > 0 else max(now.hour + 1, WORK_START)
        if start_hour >= WORK_END:
            continue

        for hour in range(start_hour, WORK_END):
            for minute in [0, 30]:
                if len(slots) >= 3:
                    break
                slot_start = check_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
                slot_end = slot_start + slot_delta

                if slot_start <= now or slot_end.hour > WORK_END:
                    continue

                conflict = any(
                    slot_start < datetime.fromisoformat(e.get("end", {}).get("dateTime") or e.get("end", {}).get("date", str(slot_end)))
                    and slot_end > datetime.fromisoformat(e.get("start", {}).get("dateTime") or e.get("start", {}).get("date", str(slot_start)))
                    for e in existing_events
                )
                if not conflict:
                    slots.append(TimeSlot(
                        start=slot_start.isoformat(),
                        end=slot_end.isoformat(),
                        label=slot_start.strftime("%a, %b %-d %Y, %-I:%M %p IST"),
                    ))
    return slots


async def get_available_slots(sub_intent: str = "normal_meet") -> List[TimeSlot]:
    duration_mins = DURATION_MINUTES.get(sub_intent, 30)
    service = _build_calendar_service()
    existing_events = []
    if service:
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        try:
            result = service.events().list(
                calendarId=settings.GOOGLE_CALENDAR_ID,
                timeMin=now.isoformat(),
                timeMax=(now + timedelta(days=7)).isoformat(),
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            existing_events = result.get("items", [])
        except Exception as e:
            logger.warning(f"Could not fetch existing events: {e}")
    return find_available_slots(existing_events, duration_minutes=duration_mins)


async def create_calendar_event(
    slot: TimeSlot, attendee_email: str, sub_intent: str = "normal_meet"
) -> Optional[str]:
    """Create Google Calendar event; returns meet/event link or None."""
    service = _build_calendar_service()
    if not service:
        return None

    duration_label = {
        "quick_call":      "Quick Call (15 min)",
        "normal_meet":     "Meeting (30 min)",
        "long_discussion": "Discussion (1 Hour)",
    }.get(sub_intent, "Meeting")

    event_body = {
        "summary": f"MHK Tech — {duration_label}",
        "description": "Meeting scheduled via MHK Nova AI assistant.",
        "start": {"dateTime": slot.start, "timeZone": "Asia/Kolkata"},
        "end":   {"dateTime": slot.end,   "timeZone": "Asia/Kolkata"},
        "attendees": [
            {"email": attendee_email},
            {"email": settings.GOOGLE_CALENDAR_ID, "organizer": True},
        ],
        "conferenceData": {
            "createRequest": {"requestId": f"mhk-{attendee_email}-{slot.start}"}
        },
        "sendUpdates": "all",
    }
    try:
        created = service.events().insert(
            calendarId=settings.GOOGLE_CALENDAR_ID,
            body=event_body,
            conferenceDataVersion=1,
            sendUpdates="all",
        ).execute()
        link = created.get("hangoutLink") or created.get("htmlLink")
        logger.info(f"Google Calendar event created: {link}")
        return link
    except Exception as e:
        logger.error(f"Failed to create Calendar event: {e}")
        return None


# =============================================================================
# Duration selection message (shown AFTER email verification)
# =============================================================================

DURATION_PROMPT = (
    " **Email verified!**\n\n"
    "Please choose your preferred meeting duration:\n\n"
    "  **15 minutes** — Quick call\n"
    "  **30 minutes** — Standard meeting _(recommended)_\n"
    "  **1 hour** — In-depth discussion\n\n"
    "Reply with **1**, **2**, or **3**."
)


# =============================================================================
# Main State-Machine Handler
# =============================================================================

async def handle_message(
    session_id: str,
    user_message: str,
    sub_intent: str = "normal_meet",
) -> Tuple[str, MeetingSession]:
    """
    State machine:
      initial → awaiting_email → awaiting_otp → awaiting_duration → awaiting_slot → completed

    The Calendly link is NEVER shown upfront. It only appears as a fallback
    when Google Calendar event creation fails and no other option is available.
    """
    session = load_session(session_id)
    if not session:
        session = MeetingSession(
            session_id=session_id,
            state="initial",
            created_at=str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            sub_intent=sub_intent,
            calendly_url=CALENDLY_URLS.get(sub_intent, settings.CALENDLY_30MIN_URL),
        )

    state = session.state

    # -------------------------------------------------------------------------
    # CANCELLATION
    # -------------------------------------------------------------------------
    user_msg_lower = user_message.strip().lower()
    cancel_keywords = ["cancel", "stop", "exit", "quit", "nevermind"]
    if state not in ("initial", "completed") and any(kw in user_msg_lower for kw in cancel_keywords):
        delete_session(session.session_id)
        session.state = "completed"
        return "No problem! I've cancelled the meeting scheduling. Let me know if there's anything else I can help you with! ", session

    # -------------------------------------------------------------------------
    # INITIAL — ask for email first (no Calendly link here)
    # -------------------------------------------------------------------------
    if state == "initial":
        session.state = "awaiting_email"
        save_session(session)
        return (
            "I'd love to help you schedule a meeting with our team! \n\n"
            "To get started, please provide your **business email address**.\n"
            "_(Personal addresses like Gmail, Yahoo, etc. are not accepted.)_"
        ), session

    # -------------------------------------------------------------------------
    # AWAITING EMAIL — validate then send OTP
    # -------------------------------------------------------------------------
    elif state == "awaiting_email":
        import re
        email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", user_message)
        if not email_match:
            return "Please provide a valid business email address to continue.", session

        email = email_match.group(0).lower()

        # Step 1: domain block-list
        is_valid, error_msg = validate_business_email(email)
        if not is_valid:
            return error_msg, session

        # Step 2: Abstract API deliverability check
        api_valid, api_msg = await validate_email_with_abstract(email)
        if not api_valid:
            return api_msg, session

        # Step 3: Daily Rate Limit Check
        if not check_daily_meeting_limit(email):
            delete_session(session.session_id)
            session.state = "completed"
            return (
                " You have reached the maximum limit of 2 meeting requests per day for this email address. "
                "Please try again tomorrow or contact us directly at hr@mhktechinc.com."
            ), session

        # All validation passed — send OTP
        otp = generate_otp()
        session.email = email
        session.otp = otp
        session.otp_expiry = otp_expiry_ms()
        session.otp_attempts = 0
        session.state = "awaiting_otp"
        save_session(session)

        sent = await send_otp_email(email, otp)
        if sent:
            return (
                f" A 6-digit verification code has been sent to **{email}**.\n\n"
                f"Please enter the code here to continue. _(Expires in 10 minutes)_"
            ), session
        else:
            return (
                " We had trouble sending the verification email. "
                "Please check your address and try again."
            ), session

    # -------------------------------------------------------------------------
    # AWAITING OTP — verify, max 3 attempts
    # -------------------------------------------------------------------------
    elif state == "awaiting_otp":
        import re
        
        # Check if the user is trying to provide a new email address instead of the OTP
        email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", user_message)
        if email_match:
            # Recursively call the initial step with the newly provided email
            session.state = "awaiting_email"
            save_session(session)
            return await handle_message(session_id, user_message, sub_intent)
            
        # Check if the user says "wrong email" but hasn't given the new one yet
        user_msg_lower = user_message.strip().lower()
        if any(word in user_msg_lower for word in ["wrong", "incorrect", "change", "not my", "other", "different"]) and not re.search(r"\b\d{6}\b", user_message):
            session.state = "awaiting_email"
            save_session(session)
            return "No problem! Please provide your correct **business email address**.", session

        otp_match = re.search(r"\b(\d{6})\b", user_message)
        if not otp_match:
            return "Please enter the **6-digit code** sent to your email.", session

        entered_otp = otp_match.group(1)

        if is_otp_expired(session.otp_expiry):
            session.state = "awaiting_email"
            session.otp = ""
            save_session(session)
            return (
                " Your verification code has **expired**.\n\n"
                "Please provide your business email again to receive a new code."
            ), session

        if entered_otp != session.otp:
            session.otp_attempts += 1
            remaining = max(0, 3 - session.otp_attempts)
            if session.otp_attempts >= 3:
                delete_session(session.session_id)
                return (
                    " **Too many incorrect attempts.** Your session has been locked for security.\n\n"
                    "Please say _'schedule a meeting'_ to start over."
                ), session
            save_session(session)
            return (
                f" Incorrect code. **{remaining} attempt{'s' if remaining != 1 else ''} remaining.**"
            ), session

        #  OTP correct — move to duration selection
        session.state = "awaiting_duration"
        save_session(session)
        return DURATION_PROMPT, session

    # -------------------------------------------------------------------------
    # AWAITING DURATION — 1 / 2 / 3 choice (shown after email verified)
    # -------------------------------------------------------------------------
    elif state == "awaiting_duration":
        import re
        choice_match = re.search(r"\b([123])\b", user_message)
        if not choice_match:
            return (
                "Please reply with **1**, **2**, or **3**:\n\n"
                " 15 minutes  |   30 minutes  |   1 hour"
            ), session

        choice = choice_match.group(1)
        new_sub_intent, duration_label = DURATION_OPTIONS[choice]
        session.sub_intent = new_sub_intent
        session.calendly_url = CALENDLY_URLS[new_sub_intent]
        session.state = "completed"
        delete_session(session.session_id)

        # Return the Calendly link — Calendly handles slot selection natively
        return (
            f" **{duration_label} meeting selected!**\n\n"
            f" **[Click here to pick your slot]({session.calendly_url})**\n\n"
            f"Use the link above to choose a time that works for you. "
            f"A confirmation will be sent to **{session.email}**. "
        ), session


    # -------------------------------------------------------------------------
    # AWAITING SLOT — pick a slot, create calendar event
    # -------------------------------------------------------------------------
    elif state == "awaiting_slot":
        import re
        slot_match = re.search(r"\b([1-3])\b", user_message)
        if not slot_match:
            return "Please reply with **1**, **2**, or **3** to choose a time slot.", session

        idx = int(slot_match.group(1)) - 1
        if idx >= len(session.slots):
            return "Invalid selection. Please choose **1**, **2**, or **3**.", session

        chosen = session.slots[idx]
        session.selected_slot = chosen
        session.state = "completed"

        # Create Google Calendar event (real event on the calendar)
        meet_link = await create_calendar_event(chosen, session.email, session.sub_intent)

        if not meet_link:
            # Fallback: Calendly link for the chosen duration
            meet_link = session.calendly_url

        session.calendar_event_link = meet_link
        delete_session(session.session_id)

        # Send confirmation email
        await send_confirmation_email(session.email, chosen.label, meet_link)

        return (
            f" **Meeting confirmed for {chosen.label}!**\n\n"
            f" A calendar invite has been sent to **{session.email}**.\n"
            f" Join link: {meet_link}\n\n"
            f"Looking forward to speaking with you! If you need to reschedule, "
            f"reach out at [hr@mhktechinc.com](mailto:hr@mhktechinc.com)."
        ), session

    # -------------------------------------------------------------------------
    # COMPLETED — cleanup legacy sessions
    # -------------------------------------------------------------------------
    elif state == "completed":
        delete_session(session.session_id)
        return "If you want to schedule another meeting, just ask! Otherwise, let me know how else I can help you.", session

    else:
        return "I'm not sure how to help with that. Say _'schedule a meeting'_ to begin.", session