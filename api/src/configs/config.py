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
