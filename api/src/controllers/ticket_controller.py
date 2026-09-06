import json
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..models import database, models
from ..schemas import TicketPageOut, UserOut
from ..services.imap_service import fetch_and_process_emails

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
