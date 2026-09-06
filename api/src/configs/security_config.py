"""Security validation configuration for inbound email guardrails.

All values here are routing policy, not secrets, so they live in code where they
can be reviewed, unit-tested, and version-controlled. If a value needs to differ
across environments it can be overridden from an environment variable (the
prefixed helpers below make that easy), but nothing sensitive belongs here.
"""

import os


def _env_list(name: str, default: list[str]) -> list[str]:
    """Read a comma-separated environment list, falling back to `default`."""
    raw = os.environ.get(name)
    if not raw:
        return default
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Trusted domains (the organisation's own). Used to whitelist senders and to
# detect typosquat / lookalike domains that mimic these.
# ---------------------------------------------------------------------------
TRUSTED_DOMAINS: list[str] = _env_list(
    "SECURITY_TRUSTED_DOMAINS",
    ["gmail.com", "outlook.com", "hotmail.com", "google.com", "live.com", "yahoo.com"],
)


def _load_domain_list(filename: str) -> list[str]:
    """Load a list of domains from a local configuration file."""
    path = os.path.join(os.path.dirname(__file__), filename)
    try:
        with open(path, encoding="utf-8") as f:
            return [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# Known disposable / throwaway email domains. These are frequently abused for
# spam, abuse, or illegitimate support requests.
# ---------------------------------------------------------------------------
DISPOSABLE_DOMAINS: list[str] = _env_list(
    "SECURITY_DISPOSABLE_DOMAINS", _load_domain_list("disposable_email_blocklist.conf")
)

# ---------------------------------------------------------------------------
# Domains commonly associated with spam / phishing campaigns.
# ---------------------------------------------------------------------------
SPAM_DOMAINS: list[str] = _env_list(
    "SECURITY_SPAM_DOMAINS",
    [
        "sendspamhere.com",
        "mailforspam.com",
        "spamgourmet.com",
    ],
)

# ---------------------------------------------------------------------------
# Dangerous attachment file extensions. Files with these extensions can execute
# code or contain malicious content; they are blocked outright.
# ---------------------------------------------------------------------------
DANGEROUS_EXTENSIONS: set[str] = {
    ".exe",
    ".bat",
    ".cmd",
    ".com",
    ".scr",
    ".pif",
    ".ps1",
    ".ps2",
    ".psm1",
    ".vbs",
    ".vbe",
    ".js",
    ".jse",
    ".jar",
    ".msi",
    ".msp",
    ".mst",
    ".reg",
    ".lnk",
    ".app",
    ".gadget",
    ".hta",
    ".wsf",
    ".wsc",
    ".cpl",
    ".ocx",
    ".sys",
    ".dll",
}

# ---------------------------------------------------------------------------
# Archive / container extensions. Not inherently malicious but are a common
# delivery mechanism for payloads, so they are flagged (not blocked outright)
# under the balanced policy.
# ---------------------------------------------------------------------------
ARCHIVE_EXTENSIONS: set[str] = {
    ".zip",
    ".rar",
    ".7z",
    ".arj",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".iso",
}

# ---------------------------------------------------------------------------
# Macro-enabled Office documents (can contain VBA). Flagged under balanced.
# ---------------------------------------------------------------------------
MACRO_OFFICE_EXTENSIONS: set[str] = {
    ".docm",
    ".xlsm",
    ".pptm",
    ".dotm",
    ".xlam",
    ".ppam",
    ".xlsb",
}

# Maximum allowed attachment size and total number of attachments before the
# mail is flagged.
MAX_ATTACHMENT_SIZE_BYTES: int = _env_int("SECURITY_MAX_ATTACHMENT_BYTES", 25 * 1024 * 1024)
MAX_ATTACHMENT_COUNT: int = _env_int("SECURITY_MAX_ATTACHMENT_COUNT", 10)

# ---------------------------------------------------------------------------
# Suspicion score (0-100) thresholds. Score is the weighted sum of finding
# severities. Below PASS_THRESHOLD the mail passes cleanly. A single medium
# finding (weight 30) must remain a *flag* rather than a quarantine, so the
# threshold sits between one and two medium findings.
# ---------------------------------------------------------------------------
SUSPICION_THRESHOLD: int = _env_int("SECURITY_SUSPICION_THRESHOLD", 45)

# Severity weights contributing to the score.
SEVERITY_WEIGHTS = {
    "high": 60,
    "medium": 30,
    "low": 10,
    "info": 0,
}
