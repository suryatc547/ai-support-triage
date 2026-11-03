"""Tests for ticket assignment / re-analysis controller endpoints."""

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.controllers.ticket_controller import assign_ticket, reanalyze_ticket
from src.models.models import Base, Ticket, TicketAuditLog, User


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fks(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def _seed_user(session, user_id: int) -> User:
    user = User(id=user_id, name="Finance Admin", email=f"admin{user_id}@example.com")
    session.add(user)
    session.commit()
    return user


def _seed_ticket(
    session,
    ticket_type: str = "needs_review",
    assigned_to: int | None = None,
    security_flag: str | None = None,
) -> Ticket:
    ticket = Ticket(
        subject="Invoice charge",
        body="Please explain the extra charge on invoice INV-20451.",
        sender_email="customer@example.com",
        ticket_type=ticket_type,
        assigned_to=assigned_to,
        security_flag=security_flag,
    )
    session.add(ticket)
    session.commit()
    return ticket


def _audit_events(session, ticket_id: int) -> list[TicketAuditLog]:
    return (
        session.query(TicketAuditLog)
        .filter(TicketAuditLog.ticket_id == ticket_id)
        .order_by(TicketAuditLog.id)
        .all()
    )


class TestAssignTicket:
    def test_assigns_unassigned_ticket(self, db_session):
        _seed_user(db_session, 1)
        ticket = _seed_ticket(db_session)

        result = assign_ticket(ticket.id, type("Payload", (), {"assignee_id": 1})(), db_session)

        assert result["assignee"]["id"] == 1
        db_session.refresh(ticket)
        assert ticket.assigned_to == 1
        events = _audit_events(db_session, ticket.id)
        assert events[-1].event_type == "reassigned"
        assert '"to": 1' in events[-1].details

    def test_unassigns_assigned_ticket_with_null(self, db_session):
        _seed_user(db_session, 1)
        ticket = _seed_ticket(db_session, assigned_to=1)

        result = assign_ticket(ticket.id, type("Payload", (), {"assignee_id": None})(), db_session)

        assert result["assignee"] is None
        db_session.refresh(ticket)
        assert ticket.assigned_to is None

    def test_missing_ticket_raises_404(self, db_session):
        _seed_user(db_session, 1)
        with pytest.raises(HTTPException) as exc:
            assign_ticket(999999, type("Payload", (), {"assignee_id": 1})(), db_session)
        assert exc.value.status_code == 404

    def test_missing_assignee_raises_404(self, db_session):
        ticket = _seed_ticket(db_session)
        with pytest.raises(HTTPException) as exc:
            assign_ticket(ticket.id, type("Payload", (), {"assignee_id": 999999})(), db_session)
        assert exc.value.status_code == 404

    def test_assignment_does_not_change_category(self, db_session):
        _seed_user(db_session, 1)
        ticket = _seed_ticket(db_session, ticket_type="needs_review")
        assign_ticket(ticket.id, type("Payload", (), {"assignee_id": 1})(), db_session)
        db_session.refresh(ticket)
        assert ticket.ticket_type == "needs_review"


class TestReanalyzeTicket:
    def test_reassigns_when_analysis_is_decisive(self, db_session, monkeypatch):
        _seed_user(db_session, 1)
        ticket = _seed_ticket(db_session)

        monkeypatch.setattr(
            "src.controllers.ticket_controller.process_ticket_via_rag",
            lambda subject, body, db: {
                "category": "finance",
                "assignee_id": 1,
                "confidence": 0.91,
                "reasoning": "invoice keywords",
            },
        )

        result = reanalyze_ticket(ticket.id, db_session)

        assert result["ticket_type"] == "finance"
        assert result["assignee"]["id"] == 1
        events = _audit_events(db_session, ticket.id)
        assert events[-1].event_type == "classified"
        assert '"assignee_id": 1' in events[-1].details

    def test_stays_needs_review_when_analysis_is_inconclusive(self, db_session, monkeypatch):
        ticket = _seed_ticket(db_session)

        monkeypatch.setattr(
            "src.controllers.ticket_controller.process_ticket_via_rag",
            lambda subject, body, db: {
                "category": "needs_review",
                "assignee_id": None,
                "confidence": 0.55,
                "reasoning": "low confidence",
            },
        )

        result = reanalyze_ticket(ticket.id, db_session)

        assert result["ticket_type"] == "needs_review"
        assert result["assignee"] is None
        assert not ticket.assigned_to

    def test_quarantined_ticket_is_never_auto_assigned(self, db_session, monkeypatch):
        _seed_user(db_session, 1)
        ticket = _seed_ticket(db_session, security_flag="quarantined")

        monkeypatch.setattr(
            "src.controllers.ticket_controller.process_ticket_via_rag",
            lambda subject, body, db: {
                "category": "finance",
                "assignee_id": 1,
                "confidence": 0.9,
                "reasoning": "decisive",
            },
        )

        result = reanalyze_ticket(ticket.id, db_session)

        assert result["ticket_type"] == "finance"
        assert result["assignee"] is None
        assert not ticket.assigned_to

    def test_missing_ticket_raises_404(self, db_session):
        with pytest.raises(HTTPException) as exc:
            reanalyze_ticket(999999, db_session)
        assert exc.value.status_code == 404
