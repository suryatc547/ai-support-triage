import email
import imaplib
import json
import logging
from email.policy import default

from ..configs.config import email_forwarding_enabled, imap_config
from ..logging_config import new_trace_id, trace_context
from ..models import database, models
from .audit_service import record_audit_event
from .email_service import forward_ticket_email
from .mail_analyzer import is_support_email
from .pii_service import redact_pii
from .rag import process_ticket_via_rag
from .security_analyzer import refine_result
from .validation_service import validate_email

logger = logging.getLogger(__name__)

_SEEN_FLAG = "(\\Seen)"

# Socket timeout for IMAP network operations. A slow or dead mail server must
# fail the sync with an error instead of leaving the sync session hung forever.
_IMAP_SOCKET_TIMEOUT = 45


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


def _ingest_message(
    msg, db, providers: list | None = None, security_providers: list | None = None
) -> dict:
    """Run the full per-message pipeline and persist the resulting ticket.

    The pipeline is: dedup on Message-ID -> security guardrails -> pre-LLM
    analyzer -> RAG classification/assignment -> atomic ticket + audit
    persistence -> optional email forwarding (gated). It is shared by the IMAP
    sync loop and offline end-to-end tests so both drive identical behaviour.

    ``providers`` and ``security_providers`` are optional LLM chain overrides
    (defaulting to the configured chain via :func:`select_llm`) — tests inject
    the real local model here so runs stay offline and deterministic.

    Returns a summary dict (``result`` in ``processed``/``quarantined``/
    ``skipped``, plus forwarding counters). Raises on processing failure so the
    caller can leave the message unread for a later retry.
    """
    summary = {
        "result": "processed",
        "forwarded": 0,
        "forward_failed": 0,
        "forward_skipped": 0,
    }

    subject = str(msg["subject"])
    sender = str(msg["from"])
    body = _extract_body(msg)
    # Logs are shipped to aggregators with broader access than the DB, so never
    # write the raw subject (names, account/order refs, addresses) to them.
    log_subject = redact_pii(subject)

    # Dedup on the RFC Message-ID header (globally unique, stable per email).
    # If a ticket for this mail already exists, skip processing to avoid
    # duplicates (the sync loop still marks the mail as read).
    message_id = msg.get("Message-ID")
    message_id_value = message_id.strip() if message_id else None

    if (
        message_id_value
        and db.query(models.Ticket).filter(models.Ticket.message_id == message_id_value).first()
    ):
        logger.info("Skipped already-processed mail: %s", log_subject)
        summary["result"] = "skipped"
        return summary

    # --- Security guardrails (runs before the analyzer / any LLM) ---
    validation = validate_email(msg)

    # Hybrid: for a flagged-but-not-quarantined mail, optionally ask the LLM
    # whether it looks like phishing. Best-effort; never weakens a hard block
    # and can only escalate.
    if validation.verdict == "flag":
        validation = refine_result(validation, subject, body, sender, providers=security_providers)

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
        # Same single-transaction invariant as the normal path:
        # add ticket -> flush -> audit event -> commit, so a flush failure can
        # never leave a committed ticket without its audit trail.
        db.add(ticket)
        db.flush()
        record_audit_event(
            db,
            ticket.id,
            event_type="quarantined",
            actor="system/security",
            details={
                "reason": validation.quarantined_reason,
                "suspicion_score": validation.score,
            },
        )
        db.commit()
        logger.warning(
            "Quarantined suspicious mail: %s (reason: %s)",
            log_subject,
            validation.quarantined_reason,
        )
        summary["result"] = "quarantined"
        return summary

    # --- Pre-LLM analyzer: cheap, deterministic gate that skips the LLM for
    # clearly non-support mail (newsletters, auto-replies, etc.).
    analysis = is_support_email(subject, body, sender)

    if analysis["is_support"]:
        # RAG classification + assignment. A raised error here means the ticket
        # could not be processed and the mail must stay unread.
        rag_result = process_ticket_via_rag(subject, body, db, providers=providers)
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
        # Not a support request — record it as such and avoid wasting a paid
        # LLM call on it.
        rag_result = None
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

    # Single atomic commit: add ticket + all audit events together so a crash
    # after `ticket.id` is valid never leaves an audit gap, and a flush failure
    # never leaves a committed ticket without an audit trail.
    db.add(ticket)
    db.flush()

    record_audit_event(
        db,
        ticket.id,
        event_type="created",
        actor="system/imap",
        details={
            "message_id": message_id_value,
            "sender": sender,
            "subject": subject,
        },
    )
    if analysis["is_support"] and rag_result is not None:
        confidence = rag_result.get("confidence", 0.0)
        record_audit_event(
            db,
            ticket.id,
            event_type="classified",
            actor="system/rag",
            details=rag_result,
        )
        if confidence < 0.70:
            record_audit_event(
                db,
                ticket.id,
                event_type="needs_review",
                actor="system/rag",
                details={
                    "confidence": confidence,
                    "reasoning": rag_result.get("reasoning", ""),
                    "category": rag_result.get("category", ""),
                },
            )
    db.commit()

    # Forward email to the assigned department if assigned. Gated by
    # EMAIL_FORWARDING so a dev IMAP-only setup never sends mail by default;
    # misses are counted (and logged) but never fatal.
    if ticket.assigned_to and email_forwarding_enabled:
        assignee = db.query(models.User).filter(models.User.id == ticket.assigned_to).first()
        if assignee:
            forward_ok = forward_ticket_email(ticket, assignee)
            record_audit_event(
                db,
                ticket.id,
                event_type="forwarded",
                actor="system/smtp",
                details={
                    "assignee_id": assignee.id,
                    "email": assignee.email,
                    "success": forward_ok,
                },
            )
            db.commit()
            if forward_ok:
                summary["forwarded"] = 1
            else:
                summary["forward_failed"] = 1
        else:
            summary["forward_skipped"] = 1
    elif ticket.assigned_to:
        summary["forward_skipped"] = 1

    logger.info("Processed mail: %s (support=%s)", log_subject, analysis["is_support"])
    return summary


def fetch_and_process_emails():
    """Run one IMAP sync, tagging every log line with a `sync-*` trace ID.

    Background tasks spawned by the API do not inherit the HTTP request trace
    context, so each sync run generates its own ID to keep the whole
    pipeline — fetch, validation, analyser, RAG, persistence — traceable.
    """
    run_id = new_trace_id("sync")
    logger.info("START IMAP sync run %s", run_id)
    with trace_context(run_id):
        _run_imap_sync()
    logger.info("END IMAP sync run %s", run_id)


def _run_imap_sync():
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
        mail = imaplib.IMAP4_SSL(imap_config["host"], timeout=_IMAP_SOCKET_TIMEOUT)
        mail.login(email_addr, password)
        mail.select(folder)

        status, data = mail.search(None, "UNSEEN")
        mail_ids = data[0].split()

        db = database.SessionLocal()

        counts = {"processed": 0, "forwarded": 0, "forward_failed": 0, "forward_skipped": 0}
        logger.info("Sync starting: %d unread message(s)", len(mail_ids))

        for m_id in mail_ids:
            try:
                status, msg_data = mail.fetch(m_id, "(RFC822)")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email, policy=default)

                summary = _ingest_message(msg, db)

                if summary["result"] != "skipped":
                    counts["processed"] += 1
                counts["forwarded"] += summary["forwarded"]
                counts["forward_failed"] += summary["forward_failed"]
                counts["forward_skipped"] += summary["forward_skipped"]

                # Processing succeeded — mark the message as read so it is not
                # re-fetched on a subsequent sync.
                mail.store(m_id, "+FLAGS", _SEEN_FLAG)

            except Exception:
                # Do NOT mark the message as read; it will be retried next sync.
                db.rollback()
                logger.exception("Failed to process mail %s, leaving unread", m_id)

        logger.info(
            "Sync complete: %d processed, %d forwarded, %d failed, %d skipped (forwarding=%s)",
            counts["processed"],
            counts["forwarded"],
            counts["forward_failed"],
            counts["forward_skipped"],
            email_forwarding_enabled,
        )

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
