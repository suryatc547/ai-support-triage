import email
import imaplib
import json
import logging
from email.policy import default

from ..configs.config import imap_config
from ..models import database, models
from .mail_analyzer import is_support_email
from .rag import process_ticket_via_rag
from .security_analyzer import refine_result
from .validation_service import validate_email

logger = logging.getLogger(__name__)

_SEEN_FLAG = "(\\Seen)"


def _extract_body(msg) -> str:
    """Extract the plain-text body from an email message, preferring text/plain parts."""
    if msg.is_multipart():
        for part in msg.iter_parts():
            if part.get_content_type() == "text/plain":
                return part.get_content()
        return ""
    return msg.get_content()


def _findings_to_json(findings) -> str:
    """Encode validation findings as a compact JSON string for storage."""
    return json.dumps(
        [{"type": f.type, "severity": f.severity, "message": f.message} for f in findings]
    )


def fetch_and_process_emails():
    """Fetch unread emails and persist the resulting tickets.

    Each message is first run through a lightweight pre-LLM analyzer:
      - Support mail is routed through the LLM for categorization/assignment.
      - Non-support mail (newsletters, auto-replies, marketing, etc.) is recorded
        as a `not_support:<signal>` ticket without spending an LLM call.

    A message is only marked as read (\\Seen) once its full processing — analyzer,
    optional RAG classification, ticket persistence, and the DB commit — succeeds.
    If any step fails, the message is left unread so it is retried on the next sync,
    and the error is logged without aborting the rest of the batch.
    """
    email_addr = imap_config.get("email")
    password = imap_config.get("password")
    folder = imap_config.get("folder", "inbox")

    if not email_addr or not password:
        logger.error("IMAP configuration missing: set IMAP_EMAIL and IMAP_PASSWORD")
        return

    mail = None
    db = None
    try:
        mail = imaplib.IMAP4_SSL(imap_config["host"])
        mail.login(email_addr, password)
        mail.select(folder)

        status, data = mail.search(None, "UNSEEN")
        mail_ids = data[0].split()

        db = database.SessionLocal()

        for m_id in mail_ids:
            try:
                status, msg_data = mail.fetch(m_id, "(RFC822)")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email, policy=default)

                subject = str(msg["subject"])
                sender = str(msg["from"])
                body = _extract_body(msg)

                # Dedup on the RFC Message-ID header (globally unique, stable per
                # email). If a ticket for this mail already exists, skip processing
                # to avoid duplicates while still marking the mail as read.
                message_id = msg.get("Message-ID")
                message_id_value = message_id.strip() if message_id else None

                if (
                    message_id_value
                    and db.query(models.Ticket)
                    .filter(models.Ticket.message_id == message_id_value)
                    .first()
                ):
                    mail.store(m_id, "+FLAGS", _SEEN_FLAG)
                    logger.info("Skipped already-processed mail %s: %s", m_id, subject)
                    continue

                # --- Security guardrails (runs before the analyzer / any LLM) ---
                validation = validate_email(msg)

                # Hybrid: for a flagged-but-not-quarantined mail, optionally ask the
                # LLM whether it looks like phishing. Best-effort; never weakens a
                # hard block and can only escalate.
                if validation.verdict == "flag":
                    validation = refine_result(validation, subject, body, sender)

                if validation.verdict == "quarantine":
                    # Suspicious mail is quarantined for security review: a ticket is
                    # recorded, flagged, and explicitly NOT auto-assigned.
                    ticket = models.Ticket(
                        message_id=message_id_value,
                        subject=subject,
                        body=body,
                        sender_email=sender,
                        ticket_type="quarantined:suspicious",
                        status="quarantined",
                        assigned_to=None,
                        security_flag="quarantined",
                        suspicion_score=validation.score,
                        validation_findings=_findings_to_json(validation.findings),
                    )
                    db.add(ticket)
                    db.commit()
                    mail.store(m_id, "+FLAGS", _SEEN_FLAG)
                    logger.warning(
                        "Quarantined suspicious mail %s: %s (reason: %s)",
                        m_id,
                        subject,
                        validation.quarantined_reason,
                    )
                    continue

                # --- Pre-LLM analyzer: cheap, deterministic gate that skips the LLM
                # for clearly non-support mail (newsletters, auto-replies, etc.).
                analysis = is_support_email(subject, body, sender)

                if analysis["is_support"]:
                    # RAG classification + assignment. A raised error here means
                    # the ticket could not be processed and the mail must stay
                    # unread.
                    rag_result = process_ticket_via_rag(subject, body, db)
                    ticket = models.Ticket(
                        message_id=message_id_value,
                        subject=subject,
                        body=body,
                        sender_email=sender,
                        ticket_type=rag_result.get("category"),
                        assigned_to=rag_result.get("assignee_id"),
                        security_flag=None if validation.verdict == "pass" else "flagged",
                        suspicion_score=validation.score,
                        validation_findings=_findings_to_json(validation.findings),
                    )
                else:
                    # Not a support request — record it as such and avoid wasting a
                    # paid LLM call on it.
                    ticket = models.Ticket(
                        message_id=message_id_value,
                        subject=subject,
                        body=body,
                        sender_email=sender,
                        ticket_type=f"not_support:{analysis['category']}",
                        assigned_to=None,
                        security_flag=None if validation.verdict == "pass" else "flagged",
                        suspicion_score=validation.score,
                        validation_findings=_findings_to_json(validation.findings),
                    )

                db.add(ticket)
                db.commit()

                # Processing succeeded — mark the message as read so it is not
                # re-fetched on a subsequent sync.
                mail.store(m_id, "+FLAGS", _SEEN_FLAG)
                logger.info(
                    "Processed mail %s: %s (support=%s)",
                    m_id,
                    subject,
                    analysis["is_support"],
                )

            except Exception:
                # Do NOT mark the message as read; it will be retried next sync.
                db.rollback()
                logger.exception("Failed to process mail %s, leaving unread", m_id)

    except Exception:
        logger.exception("Error connecting to IMAP")
    finally:
        if db is not None:
            db.close()
        if mail is not None:
            try:
                mail.close()
                mail.logout()
            except Exception:
                # Best-effort cleanup: a failure here is not worth surfacing to the caller.
                pass
