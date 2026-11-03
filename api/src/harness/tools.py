"""Typed tool definitions for the Support Agentic Harness."""

import logging
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import models
from ..services.rag import _bm25_search, _load_knowledge_base_docs

logger = logging.getLogger(__name__)


def _escape_like(value: str) -> str:
    """Escape SQL LIKE wildcards so model-supplied text matches literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_kb_tool(
    query: str, department: str | None = None, users: list | None = None, k: int = 3
) -> list[dict[str, Any]]:
    """Search the support knowledge base and routing policies via BM25 retrieval."""
    try:
        docs = _load_knowledge_base_docs(users)
        if department:
            # Scope retrieval to the department's own policy docs via the
            # metadata breadcrumbs (team/department/source) the chunker stamps,
            # not by bare-substring matching against page content (which is a
            # hair-trigger — e.g. a department like "hr" matches any doc whose
            # prose contains the letters "hr"). Categories use underscores
            # (e.g. "it_support") while metadata stores spaces ("it support"),
            # so normalise before comparing.
            dept_lower = department.lower().strip().replace("_", " ")
            filtered = [
                d
                for d in docs
                if dept_lower in (d.metadata.get("department") or "").lower()
                or dept_lower in (d.metadata.get("team") or "").lower()
                or dept_lower in str(d.metadata.get("source") or "").lower()
            ]
            docs = filtered if filtered else docs

        matched_docs = _bm25_search(docs, query, k=k)
        results = []
        for d in matched_docs:
            results.append(
                {
                    "source": d.metadata.get("source", "kb"),
                    "snippet": d.page_content[:400],
                }
            )
        return results
    except Exception as exc:
        logger.warning("search_kb_tool error: %s", exc)
        return []


def lookup_customer_history_tool(
    sender_email: str, db: Session | None, limit: int = 5
) -> list[dict[str, Any]]:
    """Retrieve previous tickets submitted by this customer to identify recurring issues."""
    if db is None or not sender_email:
        return []

    try:
        tickets = (
            db.query(models.Ticket)
            .filter(models.Ticket.sender_email == sender_email)
            .order_by(models.Ticket.created_at.desc())
            .limit(limit)
            .all()
        )
        history = []
        for t in tickets:
            history.append(
                {
                    "ticket_id": t.id,
                    "subject": t.subject,
                    "category": t.ticket_type,
                    "status": t.status,
                    "security_flag": t.security_flag,
                }
            )
        return history
    except Exception as exc:
        logger.warning("lookup_customer_history_tool error: %s", exc)
        return []


def check_staff_workload_tool(
    db: Session | None, department: str | None = None
) -> list[dict[str, Any]]:
    """Inspect active open ticket workloads per staff member to assist balanced routing."""
    if db is None:
        return []

    try:
        user_query = db.query(models.User)
        if department:
            user_query = user_query.filter(
                models.User.department.ilike(f"%{_escape_like(department.strip())}%", escape="\\")
            )
        users = user_query.all()

        workloads = []
        for u in users:
            open_count = (
                db.query(func.count(models.Ticket.id))
                .filter(
                    models.Ticket.assigned_to == u.id,
                    models.Ticket.status.in_(["open", "in_progress"]),
                )
                .scalar()
                or 0
            )
            workloads.append(
                {
                    "user_id": u.id,
                    "name": u.name,
                    "department": u.department,
                    "email": u.email,
                    "open_tickets": open_count,
                }
            )
        return workloads
    except Exception as exc:
        logger.warning("check_staff_workload_tool error: %s", exc)
        return []


def draft_resolution_tool(category: str, inquiry: str, context: str) -> str:
    """Generate a standard first-contact response draft for common, resolvable inquiries."""
    category_normalized = (category or "").lower()

    if "it" in category_normalized or "vpn" in inquiry.lower():
        return (
            "Hello,\n\n"
            "Thank you for contacting IT Support. If you have VPN/connectivity issues, "
            "please verify:\n"
            "1. You have a stable primary internet connection.\n"
            "2. Your corporate MFA token is authenticated.\n"
            "3. Restart the VPN client and select the nearest gateway.\n\n"
            "An IT Support specialist has been notified and will assist if the issue persists."
        )
    elif "hr" in category_normalized or "leave" in inquiry.lower() or "benefit" in inquiry.lower():
        return (
            "Hello,\n\n"
            "Thank you for reaching out to HR. For leave requests and benefits, "
            "please visit the internal HR portal (hr.internal/portal). "
            "An HR team member has also been assigned to your ticket to provide personal guidance."
        )
    elif "facilities" in category_normalized or "badge" in inquiry.lower():
        return (
            "Hello,\n\n"
            "Thank you for contacting Facilities. For office access or badge replacements, "
            "please note your building location and floor. A specialist will follow up shortly."
        )
    else:
        return (
            "Hello,\n\n"
            "We have received your support request and assigned it to the appropriate team member. "
            "We will review your inquiry and get back to you as soon as possible."
        )
