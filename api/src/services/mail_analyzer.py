"""Pre-LLM email analyzer.

Determines, using lightweight deterministic rules, whether an incoming email is a
support request worth routing through the LLM. This avoids spending a paid LLM call
on clearly non-support mail (newsletters, auto-replies, marketing, system
notifications, etc.).

The heuristic is intentionally rule-based and fast. It is not a replacement for the
LLM's categorization — it is a cheap gate that runs *before* the LLM.
"""

import re

# Strongly indicate the mail IS a support request.
_SUPPORT_PATTERNS = [
    re.compile(r"\b(help|assist(ance|ed)?)\b", re.IGNORECASE),
    re.compile(
        r"\b(issue|issues|problem|problems|trouble|broken|failure|failing)\b", re.IGNORECASE
    ),
    re.compile(r"\b(error|exception|bug|glitch|malfunction)\b", re.IGNORECASE),
    re.compile(r"\b(not working|don'?t work|do not work|cannot|can'?t|unable to)\b", re.IGNORECASE),
    re.compile(r"\b(fix|repair|resolve|request|support|urgent|escalat)\w*\b", re.IGNORECASE),
    re.compile(r"\b(question|inquiry|enquir\w*|need help|need assistance)\b", re.IGNORECASE),
    re.compile(
        r"\b(access|permission|reset|login|password|account|refund|invoice)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(how do i|how can i|can you|please (help|fix|assist|investigat))\b", re.IGNORECASE
    ),
    re.compile(r"\b(deadline|delay(ed)?|error code|exception)\b", re.IGNORECASE),
]

# Strong non-support signals, each paired with a machine-readable category.
_NON_SUPPORT_PATTERNS = [
    (
        re.compile(
            r"\b(out of office|o\.?o\.?o|automatic(ally)? reply|auto-?reply)\b", re.IGNORECASE
        ),
        "auto_reply",
    ),
    (
        re.compile(r"\b(do not reply|don'?t reply|no-?reply|automated message)\b", re.IGNORECASE),
        "auto_reply",
    ),
    (
        re.compile(
            r"\b(newsletter|unsubscribe|marketing|promotion|promo|sale|discount|limited time)\b",
            re.IGNORECASE,
        ),
        "marketing",
    ),
    (
        re.compile(
            r"\b(subscription (confirm|expired)|your order|stock alert|price drop|"
            r"weekly digest|monthly summary|billing statement)\b",
            re.IGNORECASE,
        ),
        "marketing",
    ),
]

# Sender addresses that are almost always automated, non-support senders.
_NO_REPLY_SENDERS = re.compile(
    r"(no-?reply|do-?not-?reply|noreply|donotreply|info|newsletter|billing|orders?)@",
    re.IGNORECASE,
)

# Leading "Re:" / "Fwd:" markers indicate a reply/forward thread.
_THREAD_RE = re.compile(r"^\s*(re|fw|fwd|sv)(\s*:|>|\[|\()", re.IGNORECASE)

_FORWARD_SIGNAL = re.compile(r"^--+\s*forwarded", re.MULTILINE | re.IGNORECASE)

_MIN_MEANINGFUL_BODY_LENGTH = 10


def _strip_quoted(text: str) -> str:
    """Remove quoted/forwarded lines that add noise to keyword matching.

    Lines starting with '>' (email quoting) are the most common source of false
    positives, since quoted conversation often contains support-ish words the
    sender did not write.
    """
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))


def is_support_email(subject: str, body: str, sender_email: str = "") -> dict:
    """Classify whether an email is a support request.

    Returns a dict:
        - is_support (bool): True if the mail should go through LLM routing.
        - category (str): Machine-readable signal for the verdict, one of
          "support", "auto_reply", "marketing", "no_reply", "empty", "thread".
        - reason (str): Human-readable short explanation.
    """
    subject_text = subject or ""
    body_text = _strip_quoted(body or "")

    # 1. Empty bodies are too ambiguous to classify as support.
    if not subject_text.strip() and not body_text.strip():
        return {"is_support": False, "category": "empty", "reason": "empty subject and body"}

    # 2. Automated / no-reply senders.
    if sender_email and _NO_REPLY_SENDERS.search(sender_email):
        return {
            "is_support": False,
            "category": "no_reply",
            "reason": f"automated sender: {sender_email}",
        }

    combined = f"{subject_text}\n{body_text}"

    # 3. Strong non-support signals (auto-reply / marketing).
    for pattern, category in _NON_SUPPORT_PATTERNS:
        if pattern.search(combined):
            return {
                "is_support": False,
                "category": category,
                "reason": pattern.pattern,
            }

    # 4. Explicit support indicators -> route to LLM.
    for pattern in _SUPPORT_PATTERNS:
        if pattern.search(combined):
            return {
                "is_support": True,
                "category": "support",
                "reason": pattern.pattern,
            }

    # 5. Reply/forward threads suggest an ongoing conversation worth routing.
    if _THREAD_RE.match(subject_text) or _FORWARD_SIGNAL.search(body_text):
        return {
            "is_support": True,
            "category": "thread",
            "reason": "thread marker detected",
        }

    # 6. Too short to be a meaningful request.
    if len(body_text.strip()) < _MIN_MEANINGFUL_BODY_LENGTH:
        return {
            "is_support": False,
            "category": "empty",
            "reason": "body too short to be a support request",
        }

    # 7. Default: treat human mail as support to avoid dropping real requests.
    return {
        "is_support": True,
        "category": "support",
        "reason": "no negative signal found",
    }
