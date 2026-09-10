import os
from pathlib import Path

from dotenv import load_dotenv

# Look for `.env` in the repo root (the workspace root that contains `api/`),
# then fall back to searching upward from this file.
_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
if not _ENV_PATH.exists():
    _ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

imap_config = {
    "email": os.environ.get("IMAP_EMAIL"),
    "password": os.environ.get("IMAP_PASSWORD"),
    "host": os.environ.get("IMAP_HOST", "imap.gmail.com"),
    "folder": os.environ.get("IMAP_FOLDER", "inbox"),
}

smtp_config = {
    "host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
    "port": int(os.environ.get("SMTP_PORT", "587")),
    "user": os.environ.get("SMTP_USER") or os.environ.get("IMAP_EMAIL"),
    "password": os.environ.get("SMTP_PASSWORD") or os.environ.get("IMAP_PASSWORD"),
}

# Master switch for forwarding assigned tickets to staff by email. Disabled by
# default so a development IMAP-only setup never triggers real SMTP traffic.
email_forwarding_enabled = os.environ.get("EMAIL_FORWARDING", "false").lower() in (
    "true",
    "1",
    "yes",
)


def _parse_api_keys(value: str | None) -> tuple[str, ...]:
    """Split a comma-separated API key list, dropping empty entries."""
    return tuple(k.strip() for k in (value or "").split(",") if k.strip())


# API key(s) accepted for authenticated API access. Empty means auth is
# disabled (local/dev convenience); set API_KEYS (or API_KEY) to enforce it.
api_keys = _parse_api_keys(os.environ.get("API_KEYS") or os.environ.get("API_KEY"))
