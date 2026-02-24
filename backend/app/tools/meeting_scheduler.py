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


_IN_MEMORY_SESSIONS = {}

def load_session(session_id: str) -> Optional[MeetingSession]:
    client = _redis()
    if not client:
        # Fallback to in-memory
        raw = _IN_MEMORY_SESSIONS.get(session_id)
        if not raw:
            return None
        return MeetingSession.from_dict(raw)
        
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
        # Fallback to in-memory
        _IN_MEMORY_SESSIONS[session.session_id] = session.to_dict()
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
    if not client:
        # Fallback to in-memory
        _IN_MEMORY_SESSIONS.pop(session_id, None)
        return
        
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
  <h2 style="color:#1e293b;margin-bottom:8px;">Verification Code to proceed with the Meeting Scheduling </h2>
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
    is_educational = any(domain.endswith(ext) for ext in [".edu", ".ac.uk", ".ac.in", ".edu.in", ".ac.us"])

    if domain in _PERSONAL_PROVIDERS and not is_educational:
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
        domain = email.split("@")[1]
        
        # Explicitly allow educational domains regardless of what the API says
        is_educational = any(domain.endswith(ext) for ext in [".edu", ".ac.uk", ".ac.in", ".edu.in", ".ac.us"])
        
        if is_free_provider and not is_educational:
            return False, (
                f"Please use your **business or educational email** — free email providers like **@{domain}** "
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


async def _route_meeting_message(state: str, message: str) -> dict:
    from app.services.llm.openai_client import get_openai_client
    
    system_msg = (
        "You are an intelligent routing assistant for a meeting scheduling flow.\\n"
        f"The user is currently in the state: '{state}'.\\n"
        "Analyze the user's message and determine their intent. Respond ONLY with a valid JSON object.\\n\\n"
        "***CRITICAL BUSINESS RULE***:\\n"
        "We ONLY accept business or educational emails. We absolutely DO NOT accept personal emails (e.g. gmail, yahoo, outlook, hotmail, etc). If a user says they don't have a business email, you MUST politely tell them that we cannot proceed without one. Never suggest they can use a personal email.\\n\\n"
        "***CRITICAL ROUTING RULE***:\\n"
        "If the user provides an email address (even inside a sentence), you MUST output 'PROVIDE_EMAIL'.\\n"
        "If the user provides a 6-digit code, you MUST output 'PROVIDE_OTP'.\\n"
        "DO NOT use 'CHAT' if they provided the requested information for the current step.\\n\\n"
        "- CANCEL: User wants to cancel, stop, exit, or expresses lack of interest.\\n"
        "- PROVIDE_EMAIL: User is providing an email address.\\n"
        "- PROVIDE_OTP: User is providing a numerical verification code.\\n"
        "- PROVIDE_DURATION: User is providing a meeting duration. Map to 'quick_call' (15m), 'normal_meet' (30m), or 'long_discussion' (1hr/60m).\\n"
        "- UNSUPPORTED_DURATION: User is requesting a time duration we do not support (e.g., '2 hours', '5 minutes', 'all day').\\n"
        "- CHAT: User is making a comment, asking a question, or says they cannot provide the requested info (e.g. 'I don't have a business email').\\n\\n"
        "JSON Format:\\n"
        "{\\n"
        "  \"action\": \"<ACTION>\",\\n"
        "  \"data\": \"<extracted data, e.g., 'normal_meet', 'quick_call', 'long_discussion', or the email string, or the OTP string, or null>\",\\n"
        "  \"response\": \"<If action is CHAT, provide a polite, conversational response directly answering the user, strictly enforcing any business rules if necessary, and reminding them of the current step. Otherwise null.>\"\\n"
        "}"
    )
    
    try:
        client = get_openai_client()
        response = client.chat_completion(
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": message}],
            temperature=0.0,
            max_tokens=100
        )
        response_text = response.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"Meeting router failed: {e}")
        return {"action": "UNKNOWN", "data": None}


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
      initial → awaiting_email → awaiting_otp → awaiting_duration → completed

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

    # Route message via LLM
    route = await _route_meeting_message(state, user_message)
    action = route.get("action", "")
    data = route.get("data")

    # -------------------------------------------------------------------------
    # CANCELLATION AND CHAT
    # -------------------------------------------------------------------------
    if action == "CANCEL":
        delete_session(session.session_id)
        session.state = "completed"
        return "No problem! I've cancelled the meeting scheduling. Let me know if there's anything else I can help you with! ", session

    if action == "CHAT":
        chat_resp = route.get("response")
        if not chat_resp:
            chat_resp = f"Please provide the requested information for the current step: {state.replace('_', ' ')}. Or say 'cancel' to stop."
        return chat_resp, session

    # -------------------------------------------------------------------------
    # AWAITING EMAIL — validate then send OTP
    # -------------------------------------------------------------------------
    elif state == "awaiting_email":
        email = data if action == "PROVIDE_EMAIL" and data else None
        if not email:
            import re
            email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", user_message)
            email = email_match.group(0).lower() if email_match else None

        if not email:
            return "Please provide a valid business email address to continue.", session

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
            return (
                f"You have reached the maximum limit of 2 meeting requests per day for the email address **{email}**.\\n"
                "Please try again tomorrow or provide a different business email address to continue."
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
        if action == "PROVIDE_EMAIL" and data:
            session.state = "awaiting_email"
            save_session(session)
            return await handle_message(session_id, user_message, sub_intent)

        entered_otp = str(data).strip() if action == "PROVIDE_OTP" and data else None
        if not entered_otp:
            import re
            otp_match = re.search(r"\b(\d{6})\b", user_message)
            entered_otp = otp_match.group(1) if otp_match else None

        if not entered_otp and "wrong" in user_message.lower():
             session.state = "awaiting_email"
             save_session(session)
             return "No problem! Please provide your correct **business email address**.", session

        if not entered_otp:
            return "Please enter the **6-digit code** sent to your email.", session

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
    # AWAITING DURATION — choice (shown after email verified)
    # -------------------------------------------------------------------------
    elif state == "awaiting_duration":
        if action == "UNSUPPORTED_DURATION":
            return (
                "Sorry, I can only schedule meetings for **15 minutes**, **30 minutes**, or **1 hour**.\\n"
                "Please choose one of these options by replying with **1**, **2**, or **3**."
            ), session

        # Check LLM router data
        choice = None
        if action == "PROVIDE_DURATION" and data in ("quick_call", "normal_meet", "long_discussion", "1", "2", "3"):
            mapping = {
                "1": "quick_call", "quick_call": "quick_call",
                "2": "normal_meet", "normal_meet": "normal_meet",
                "3": "long_discussion", "long_discussion": "long_discussion"
            }
            choice = mapping.get(data)
        
        if not choice:
            import re
            # Only match the exact digit or digit with whitespace to avoid grabbing '2' from '2 hours'
            choice_match = re.search(r"^\s*([123])\s*$", user_message)
            choice = choice_match.group(1) if choice_match else None
            if choice == "1": choice = "quick_call"
            elif choice == "2": choice = "normal_meet"
            elif choice == "3": choice = "long_discussion"

        if not choice:
            return (
                "Please reply with a valid duration like '15 minutes', '30 minutes', or '1 hour':\n\n"
                " 15 minutes  |   30 minutes  |   1 hour"
            ), session

        new_sub_intent = choice
        duration_label = "15 minutes" if choice == "quick_call" else "30 minutes" if choice == "normal_meet" else "1 hour"
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
    # COMPLETED — cleanup legacy sessions
    # -------------------------------------------------------------------------
    elif state == "completed":
        delete_session(session.session_id)
        return "If you want to schedule another meeting, just ask! Otherwise, let me know how else I can help you.", session

    else:
        return "I'm not sure how to help with that. Say _'schedule a meeting'_ to begin.", session