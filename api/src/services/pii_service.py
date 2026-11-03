"""PII (Personally Identifiable Information) redaction service.

Sanitizes sensitive data (email addresses, credit cards, SSNs, passwords, API
tokens, phone numbers) from email subject and body before transmitting text to
remote third-party LLMs. Original text is preserved unchanged in the local
database.

Design notes (false-positive control):
- Credit cards are only redacted when the candidate number passes the Luhn
  checksum, so arbitrary 16-digit order/reference numbers are preserved.
- Phone numbers must be bounded by non-digits on both sides, so longer numeric
  sequences (credit cards, IP addresses) are not partially consumed.
- Key-value secrets capture the full value up to line end (multi-word values
  like ``password = sup3r s3cret`` are fully masked, not just the first token).
"""

import re

# Credit card regex: matches 13 to 19 digits formatted with spaces, dashes, or
# consecutive digits. Candidates are additionally validated with the Luhn
# checksum before redaction, so innocent 16-digit reference numbers stay intact.
_CREDIT_CARD_REGEX = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?|"  # Visa
    r"5[1-5][0-9]{14}|"  # MasterCard
    r"3[47][0-9]{13}|"  # American Express
    r"3(?:0[0-5]|[68][0-9])[0-9]{11}|"  # Diners Club
    r"6(?:011|5[0-9]{2})[0-9]{12}|"  # Discover
    r"(?:2131|1800|35\d{3})\d{11}|"  # JCB
    r"\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}"  # Generic 16-digit chunked
    r")\b"
)

# Social Security Numbers (US format: XXX-XX-XXXX)
_SSN_REGEX = re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")

# Email addresses (customer and staff PII). Requires an ``@`` and a dotted domain
# with a 2+ character TLD, so bare ``user@host`` tokens and prose are preserved.
_EMAIL_REGEX = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")

# Passwords, API keys, tokens, Bearer tokens, and secrets in key-value format.
# The value runs to end-of-line (but never crosses newlines) so multi-word
# secrets like ``password = sup3r s3cret`` — and multi-credential lines — are
# fully masked rather than leaking their tail. This is conservative on purpose:
# the copy that reaches a remote LLM is aggressively scrubbed while the original
# stays intact in the local database.
_SECRET_KV_REGEX = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|token|auth[_-]?token|private[_-]?key)"
    r"[ \t]*[:=][ \t]*([^\r\n]+)",
)

_BEARER_TOKEN_REGEX = re.compile(r"(?i)\bBearer\s+[a-zA-Z0-9_\-\.=]{16,}\b")

# Phone numbers (international and national formats with area codes).
# Fallback-negative-lookbehind/lookahead on digits prevent partial matches:
#  - a bare number must be exactly 10 digits (no leading/trailing digit),
#  - international numbers must carry an explicit `+` prefix,
#  - area codes may be wrapped in parentheses.
_PHONE_REGEX = re.compile(
    r"(?<!\d)(?:"
    r"(?:\+\d{1,3}[-.\s]?"  # international: explicit +country prefix
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"
    r"|"  # or
    r"(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}"  # national/local NPA-NXX-XXXX
    r")(?!\d)"
)


def _luhn_check(number: str) -> bool:
    """Return True when ``number`` (digits and optional separators) passes the Luhn checksum.

    Used to distinguish real payment-card numbers from arbitrary long numeric
    strings (order IDs, reference numbers) before redacting them.
    """
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_credit_cards(text: str) -> str:
    """Replace credit-card-shaped candidates only when they pass the Luhn checksum."""

    def _replace(match):
        candidate = re.sub(r"[-\s]", "", match.group(0))
        if _luhn_check(candidate):
            return "[REDACTED_CREDIT_CARD]"
        return match.group(0)

    return _CREDIT_CARD_REGEX.sub(_replace, text)


def redact_emails(text: str) -> str:
    """Redact only email addresses.

    Use this for context that is otherwise safe to send to a remote model (for
    example KB policy text whose ``{{contact_email}}`` placeholders resolve to
    staff addresses): the broader :func:`redact_pii` rules would mangle innocent
    policy prose such as "rotate the API key".
    """
    if not text:
        return text or ""
    return _EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)


def redact_pii(text: str) -> str:
    """Redact sensitive PII and secrets from the provided text string.

    Returns the sanitized string with sensitive entities replaced with
    standardized tokens like `[REDACTED_CREDIT_CARD]`, `[REDACTED_SSN]`,
    `[REDACTED_SECRET]`, or `[REDACTED_PHONE]`.
    """
    if not text:
        return text or ""

    sanitized = text

    # Redact Bearer tokens
    sanitized = _BEARER_TOKEN_REGEX.sub("Bearer [REDACTED_TOKEN]", sanitized)

    # Redact key-value secrets (e.g., "password: mySecretPassword123")
    sanitized = _SECRET_KV_REGEX.sub(r"\1: [REDACTED_SECRET]", sanitized)

    # Redact Credit Cards (Luhn-validated)
    sanitized = _redact_credit_cards(sanitized)

    # Redact SSNs
    sanitized = _SSN_REGEX.sub("[REDACTED_SSN]", sanitized)

    # Redact email addresses (customer/staff identity, incl. header addresses)
    sanitized = _EMAIL_REGEX.sub("[REDACTED_EMAIL]", sanitized)

    # Redact Phone Numbers
    sanitized = _PHONE_REGEX.sub("[REDACTED_PHONE]", sanitized)

    return sanitized
