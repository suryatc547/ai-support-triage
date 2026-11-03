import json
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..models import database, models
from ..schemas import TicketAssignIn, TicketOut, TicketPageOut, UserOut
from ..services.audit_service import record_audit_event
from ..services.imap_service import fetch_and_process_emails
from ..services.rag import process_ticket_via_rag

router = APIRouter()

DbSession = Annotated[Session, Depends(database.get_db)]


def _parse_findings(raw: str | None) -> list[dict[str, object]]:
    """Decode the JSON-encoded validation findings stored on a ticket.

    Returns an empty list when the value is missing or not valid JSON, so the
    controller never surfaces a raw parsing error to the client.
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _serialize_ticket(t: models.Ticket) -> dict:
    return {
        "id": t.id,
        "subject": t.subject,
        "body": t.body,
        "sender_email": t.sender_email,
        "ticket_type": t.ticket_type,
        "status": t.status,
        "created_at": t.created_at,
        "assignee": (
            {
                "id": t.assignee.id,
                "name": t.assignee.name,
                "email": t.assignee.email,
            }
            if t.assignee
            else None
        ),
        "security_flag": t.security_flag,
        "suspicion_score": t.suspicion_score or 0,
        "validation_findings": _parse_findings(t.validation_findings),
    }


@router.get("/api/tickets", response_model=TicketPageOut)
def get_tickets(
    db: DbSession,
    search: str | None = Query(default=None),
    security: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=100),
):
    query = db.query(models.Ticket)

    if search:
        term = f"%{search}%"
        query = query.filter(
            or_(
                models.Ticket.subject.ilike(term),
                models.Ticket.body.ilike(term),
                models.Ticket.sender_email.ilike(term),
                models.Ticket.ticket_type.ilike(term),
            )
        )

    if security and security != "all":
        quarantine = "quarantined"
        flagged = "flagged"
        if security == "suspicious":
            query = query.filter(models.Ticket.security_flag.isnot(None))
        elif security == "quarantined":
            query = query.filter(models.Ticket.security_flag == quarantine)
        elif security == "flagged":
            query = query.filter(models.Ticket.security_flag == flagged)
        else:
            query = query.filter(models.Ticket.security_flag == security)

    total = query.count()
    tickets = (
        query.order_by(models.Ticket.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    return {
        "items": [_serialize_ticket(t) for t in tickets],
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit,
    }


@router.get("/api/users", response_model=list[UserOut])
def get_users(db: DbSession):
    users = db.query(models.User).all()
    return [{"id": u.id, "name": u.name, "department": u.department} for u in users]


@router.post("/api/sync")
def trigger_sync(background_tasks: BackgroundTasks, db: DbSession):
    # Trigger an IMAP sync via background task
    background_tasks.add_task(fetch_and_process_emails)
    return {"message": "Email sync triggered in background"}


def _get_ticket_or_404(db: Session, ticket_id: int) -> models.Ticket:
    ticket = db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.post("/api/tickets/{ticket_id}/assign", response_model=TicketOut)
def assign_ticket(ticket_id: int, payload: TicketAssignIn, db: DbSession):
    """Manually assign (or unassign) a ticket to a support staff member.

    Only a human triggers this endpoint — unlike the RAG pipeline it overrides
    the ``needs_review`` guard explicitly. Assigning ``null`` clears the
    assignment.
    """
    ticket = _get_ticket_or_404(db, ticket_id)

    if payload.assignee_id is not None:
        assignee = db.query(models.User).filter(models.User.id == payload.assignee_id).first()
        if assignee is None:
            raise HTTPException(status_code=404, detail="Assignee not found")
    else:
        assignee = None

    previous = ticket.assigned_to
    ticket.assigned_to = payload.assignee_id
    record_audit_event(
        db,
        ticket.id,
        "reassigned",
        "system/manual",
        {
            "from": previous,
            "to": payload.assignee_id,
            "assignee_name": assignee.name if assignee else None,
        },
    )
    db.commit()
    db.refresh(ticket)
    return _serialize_ticket(ticket)


@router.post("/api/tickets/{ticket_id}/reanalyze", response_model=TicketOut)
def reanalyze_ticket(ticket_id: int, db: DbSession):
    """Re-run the RAG classifier over a stored ticket's subject/body.

    Updates ``ticket_type`` and, when the analysis is decisive, the assignee.
    Quarantined tickets are never auto-assigned, even by a fresh analysis — the
    guardrail deliberately leaves those to a human.
    """
    ticket = _get_ticket_or_404(db, ticket_id)

    result = process_ticket_via_rag(ticket.subject, ticket.body, db)
    quarantined = ticket.security_flag == "quarantined"

    new_category = result["category"]
    new_assignee = None if quarantined else result["assignee_id"]
    previous = ticket.assigned_to

    ticket.ticket_type = new_category
    ticket.assigned_to = new_assignee

    event_type = "classified" if new_assignee is not None else "needs_review"
    record_audit_event(
        db,
        ticket.id,
        event_type,
        "system/manual",
        {
            "method": "reanalyze",
            "from_assignee_id": previous,
            "assignee_id": new_assignee,
            "category": new_category,
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "quarantined": quarantined,
        },
    )
    db.commit()
    db.refresh(ticket)
    return _serialize_ticket(ticket)
