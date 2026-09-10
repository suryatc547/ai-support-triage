"""Email forwarding service for assigned support tickets.

When a support ticket is categorized and assigned to a department / staff member,
this service forwards the customer's original email to the department's email address
using SMTP.
"""

import logging
import smtplib
import ssl
from email.message import EmailMessage

from ..configs.config import smtp_config
from ..models import models

logger = logging.getLogger(__name__)


def build_forward_message(
    ticket: models.Ticket, assignee: models.User, from_addr: str
) -> EmailMessage:
    """Construct the forwarded email with structured metadata and the original email body."""
    msg = EmailMessage()
    msg["Subject"] = f"Fwd: [Ticket #{ticket.id}] {ticket.subject or 'No Subject'}"
    msg["From"] = from_addr
    msg["To"] = assignee.email
    if ticket.sender_email:
        msg["Reply-To"] = ticket.sender_email

    header_lines = [
        f"Ticket ID   : #{ticket.id}",
        f"Department  : {assignee.department}",
        f"Assigned To : {assignee.name} <{assignee.email}>",
        f"Status      : {ticket.status}",
        f"Category    : {ticket.ticket_type or 'unclassified'}",
        f"From        : {ticket.sender_email or 'unknown'}",
        f"Subject     : {ticket.subject or 'No Subject'}",
    ]
    if ticket.security_flag:
        header_lines.append(
            f"Security    : {ticket.security_flag} (score {ticket.suspicion_score})"
        )

    separator = "-" * 50
    dept_info = (
        f"Support Ticket #{ticket.id} has been assigned to your department "
        f"({assignee.department}).\n\n"
    )
    content = (
        dept_info
        + f"{separator}\n"
        + "\n".join(header_lines)
        + f"\n{separator}\n\n"
        + (ticket.body or "")
    )
    msg.set_content(content)
    return msg


def forward_ticket_email(ticket: models.Ticket, assignee: models.User) -> bool:
    """Forward a support ticket to the assigned staff member's email address.

    Returns True if forwarded successfully, or False if skipped/failed.
    This operation is non-fatal: exceptions are logged and swallowed so ticket
    persistence and ingestion flows never abort due to transient SMTP errors.
    """
    if not assignee or not assignee.email:
        logger.warning(
            "Cannot forward ticket #%s: missing assignee email",
            getattr(ticket, "id", None),
        )
        return False

    host = smtp_config.get("host")
    port = smtp_config.get("port", 587)
    user = smtp_config.get("user")
    password = smtp_config.get("password")

    if not user or not password:
        logger.warning(
            "SMTP credentials not configured (set SMTP_USER/SMTP_PASSWORD or "
            "IMAP_EMAIL/IMAP_PASSWORD). Skipping email forward for ticket #%s to %s.",
            getattr(ticket, "id", None),
            assignee.email,
        )
        return False

    msg = build_forward_message(ticket, assignee, from_addr=user)

    try:
        if port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(user, password)
                server.send_message(msg)

        logger.info(
            "Forwarded ticket #%s to %s <%s> for %s via %s:%s",
            ticket.id,
            assignee.name,
            assignee.email,
            assignee.department,
            host,
            port,
        )
        return True
    except Exception:
        # Non-fatal by design: do not dump a traceback for every failed forward —
        # a compact warning keeps sync logs readable while the per-run summary in
        # imap_service still surfaces the aggregate failure count.
        logger.warning(
            "Failed to forward ticket #%s to %s <%s> via SMTP (%s:%s)",
            ticket.id,
            assignee.name,
            assignee.email,
            host,
            port,
        )
        return False
