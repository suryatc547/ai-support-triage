"""Audit service for tracking ticket lifecycle events and state transitions."""

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from ..models import models

logger = logging.getLogger(__name__)


def record_audit_event(
    db: Session,
    ticket_id: int,
    event_type: str,
    actor: str,
    details: dict[str, Any] | str | None = None,
) -> models.TicketAuditLog | None:
    """Record an immutable audit log entry for a ticket lifecycle event.

    :param db: Active SQLAlchemy database session.
    :param ticket_id: ID of the ticket.
    :param event_type: Event classification ('created', 'quarantined', 'classified',
                       'reassigned', 'forwarded', 'resolved', etc.).
    :param actor: Identifying string for the actor ('system/imap', 'system/rag',
                  'system/smtp', 'user/<id>').
    :param details: Optional metadata dict or string describing the transition.
    """
    if not ticket_id:
        return None

    try:
        details_str = json.dumps(details) if isinstance(details, dict) else details
        log_entry = models.TicketAuditLog(
            ticket_id=ticket_id,
            event_type=event_type,
            actor=actor,
            details=details_str,
        )
        db.add(log_entry)
        db.flush()
        return log_entry
    except Exception:
        # Flush/commit errors should never leave the session in a poisoned
        # state — call rollback so subsequent work in the same transaction
        # does not hit a ``PendingRollbackError``.
        db.rollback()
        logger.exception("Failed to record audit event for ticket #%s", ticket_id)
        return None
