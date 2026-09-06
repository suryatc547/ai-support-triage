"""Deterministic inbound email validation (guardrails).

This module is the first gate an email passes after it is fetched from the IMAP
inbox and de-duplicated — it runs *before* the analyzer and *before* any LLM
call. Security decisions here are intentionally deterministic and offline so the
system never depends on a paid model or external service to stay safe.

It emits structured :class:`Finding` objects, aggregates them into a suspicion
score, and returns a verdict:

- ``pass``       : the mail is trusted and should be processed normally.
- ``flag``       : the mail has minor suspicion but is safe to process; callers
                   should surface the findings for awareness.
- ``quarantine`` : the mail failed validation; callers should NOT auto-assign it
                   and should hold it for security review.

Policy (see ``security_config`` for the exact blocklists):
- Hard-block: disposable/throwaway domains, known spam domains, and dangerous
  executable/script attachment extensions.
- Flag: lookalike (typosquat) domains, macro-enabled or archive attachments,
  and unusual attachment sizes/counts.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from email.message import EmailMessage
from email.utils import parseaddr

from ..configs import security_config as cfg

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single validation issue detected on an email."""

    type: str
    severity: str  # one of: high | medium | low | info
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Aggregated outcome of validating a single email."""

    verdict: str  # pass | flag | quarantine
    score: int  # 0-100 cumulative suspicion
    findings: list[Finding] = field(default_factory=list)
    is_trusted: bool = False
    quarantined_reason: str | None = None  # populated when verdict == quarantine

    @property
    def blocked(self) -> bool:
        return self.verdict == "quarantine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sender(sender_header: str) -> tuple[str, str]:
    """Return (display_name, email_address) from a raw From/Reply-To header."""
    if not sender_header:
        return "", ""
    name, addr = parseaddr(sender_header)
    return name or "", (addr or "").strip()


def _domain_of(email_addr: str) -> str:
    """Return the lower-cased domain of an email address (or '' if malformed)."""
    if "@" not in email_addr:
        return ""
    return email_addr.rsplit("@", 1)[1].strip().lower().rstrip(".")


def _is_trusted_domain(domain: str) -> bool:
    """True if the domain is the organisation's own (or a subdomain of it)."""
    for trusted in cfg.TRUSTED_DOMAINS:
        safe = trusted.strip().lower().lstrip(".")
        if domain == safe or domain.endswith("." + safe):
            return True
    return False


def _is_disposable(domain: str) -> bool:
    """True if the domain is (or is a subdomain of) a disposable-email domain."""
    return any(domain == d or domain.endswith("." + d) for d in cfg.DISPOSABLE_DOMAINS)


def _is_spam(domain: str) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in cfg.SPAM_DOMAINS)


def _lookalike(domain: str) -> str | None:
    """Return the trusted domain a suspicious domain most resembles, if any.

    Detects typosquatting: a domain that is nearly identical to a trusted domain
    (confusable-character substitution and/or a small edit distance).
    """
    if not domain:
        return None

    check = domain
    # Normalise confusable characters for homograph / substitution detection.
    confusable = {
        "0": "o",
        "1": "l",
        "3": "e",
        "5": "s",
        "8": "b",
        "@": "a",
        "$": "s",
    }
    folded = "".join(confusable.get(ch, ch) for ch in check)

    for trusted in cfg.TRUSTED_DOMAINS:
        safe = trusted.strip().lower().lstrip(".")
        if not safe:
            continue
        folded_trusted = "".join(confusable.get(ch, ch) for ch in safe)

        if domain == safe:
            continue  # exact trusted domain
        if domain.endswith("." + safe):
            continue  # genuine subdomain of a trusted domain

        # Very short edit distance -> likely typo (e.g. test.co vs test.com).
        if SequenceMatcher(None, check, safe).ratio() >= 0.9:
            return safe
        # Confusable-folded equality (e.g. g00gle vs google).
        if folded == folded_trusted:
            return safe

    return None


def _attachment_ext(filename: str) -> str:
    """Return the lower-cased final file extension (without the dot), or ''."""
    if not filename:
        return ""
    base = filename.strip()
    if "." not in base:
        return ""
    # Strip trailing spaces/quotes that some filenames carry.
    base = base.rstrip()
    ext = base.rsplit(".", 1)[1].lower()
    # Guard against the extension being part of a hidden/space name.
    if not ext.isalnum():
        return ""
    return "." + ext


def _is_double_extension(filename: str) -> bool:
    """Detect payload-hiding double extensions like ``invoice.pdf.exe``."""
    base = filename.strip().rstrip(".")
    dots = base.split(".")
    if len(dots) < 3:
        return False
    return len(dots) >= 3 and bool(dots[-1]) and bool(dots[-2])


# ---------------------------------------------------------------------------
# Per-domain checks
# ---------------------------------------------------------------------------


def _check_sender(sender_header: str) -> list[Finding]:
    findings: list[Finding] = []
    _name, addr = _parse_sender(sender_header)
    domain = _domain_of(addr)

    if not addr or "@" not in addr:
        findings.append(
            Finding(
                "malformed_sender",
                "medium",
                f"Malformed or missing sender address: {sender_header!r}",
            )
        )
        return findings

    if _is_trusted_domain(domain):
        return findings  # trusted sender short-circuits the remaining checks

    if _is_disposable(domain):
        findings.append(
            Finding(
                "disposable_domain",
                "high",
                f"Sender uses a disposable/throwaway email domain '{domain}'.",
                {"domain": domain},
            )
        )

    if _is_spam(domain):
        findings.append(
            Finding(
                "spam_domain",
                "high",
                f"Sender domain '{domain}' is on the spam blocklist.",
                {"domain": domain},
            )
        )

    lookalike = _lookalike(domain)
    if lookalike:
        findings.append(
            Finding(
                "typosquat",
                "medium",
                f"Domain '{domain}' closely resembles trusted domain '{lookalike}'.",
                {"domain": domain, "lookalike": lookalike},
            )
        )

    return findings


def _check_attachments(message: EmailMessage) -> list[Finding]:
    findings: list[Finding] = []

    if not message.is_multipart():
        return findings

    attachments = []
    for part in message.walk():
        filename = part.get_filename()
        if not filename:
            continue
        attachments.append((filename, part.get_content_type(), part.get_payload(decode=True)))

    count = len(attachments)
    if count > cfg.MAX_ATTACHMENT_COUNT:
        findings.append(
            Finding(
                "too_many_attachments",
                "medium",
                f"Email has {count} attachments (limit {cfg.MAX_ATTACHMENT_COUNT}).",
                {"count": count},
            )
        )

    for filename, _content_type, payload in attachments:
        ext = _attachment_ext(filename)
        size = len(payload) if payload is not None else None

        if ext in cfg.DANGEROUS_EXTENSIONS:
            findings.append(
                Finding(
                    "dangerous_attachment",
                    "high",
                    f"Attachment '{filename}' has a dangerous executable/script extension.",
                    {"filename": filename, "extension": ext},
                )
            )
            # Dangerous content is a block regardless of other formality.
            continue

        if _is_double_extension(filename):
            findings.append(
                Finding(
                    "double_extension",
                    "medium",
                    f"Attachment '{filename}' appears to hide a second extension.",
                    {"filename": filename},
                )
            )

        if ext in cfg.ARCHIVE_EXTENSIONS or ext in cfg.MACRO_OFFICE_EXTENSIONS:
            findings.append(
                Finding(
                    "risky_attachment",
                    "low" if ext in cfg.ARCHIVE_EXTENSIONS else "medium",
                    f"Attachment '{filename}' is a {ext[1:].upper()} file "
                    f"used to deliver payloads.",
                    {"filename": filename, "extension": ext},
                )
            )

        if size is not None and size > cfg.MAX_ATTACHMENT_SIZE_BYTES:
            limit_mb = cfg.MAX_ATTACHMENT_SIZE_BYTES // (1024 * 1024)
            findings.append(
                Finding(
                    "oversized_attachment",
                    "low",
                    f"Attachment '{filename}' exceeds the {limit_mb}MB limit.",
                    {"filename": filename, "size": size},
                )
            )

    return findings


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _score(findings: list[Finding]) -> int:
    return sum(cfg.SEVERITY_WEIGHTS.get(f.severity, 0) for f in findings)


def _verdict(findings: list[Finding], score: int, email_addr: str) -> tuple[str, str | None]:
    """Decide the verdict and human reason from the collected findings."""
    if any(f.severity == "high" for f in findings):
        reasons = [f.message for f in findings if f.severity == "high"]
        return "quarantine", "; ".join(reasons)
    if score >= cfg.SUSPICION_THRESHOLD:
        return "quarantine", "Cumulative suspicion score exceeded threshold"
    if score > 0:
        return "flag", None
    return "pass", None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_email(message: EmailMessage) -> ValidationResult:
    """Validate a parsed email message and return a guardrail verdict.

    :param message: An ``email.message.EmailMessage`` as produced by the IMAP
        ingestion pipeline.
    :return: A :class:`ValidationResult` with the verdict, suspicion score, and
        the list of findings.
    """
    findings: list[Finding] = []
    is_trusted = False

    sender_header = str(message.get("From", ""))
    _name, sender_addr = _parse_sender(sender_header)
    sender_domain = _domain_of(sender_addr)

    if sender_domain and _is_trusted_domain(sender_domain):
        # Mail from the organisation's own domain is trusted and short-circuits
        # the sender checks (attachments are still screened).
        is_trusted = True
    else:
        findings.extend(_check_sender(sender_header))

    findings.extend(_check_attachments(message))

    score = _score(findings)
    verdict, quarantined_reason = _verdict(findings, score, sender_addr)

    return ValidationResult(
        verdict=verdict,
        score=score,
        findings=findings,
        is_trusted=is_trusted,
        quarantined_reason=quarantined_reason,
    )
