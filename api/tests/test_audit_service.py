"""Tests for ticket audit log tracking and service."""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.models.models import Base, Ticket, TicketAuditLog
from src.services.audit_service import record_audit_event


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")

    # Match production FK enforcement (models/database.py) so FK-violation
    # failure paths can be exercised in tests.
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


def test_record_audit_event_success(db_session):
    ticket = Ticket(
        subject="Test Ticket",
        body="Audit body",
        sender_email="test@example.com",
    )
    db_session.add(ticket)
    db_session.commit()

    entry = record_audit_event(
        db_session,
        ticket.id,
        event_type="created",
        actor="system/imap",
        details={"sender": "test@example.com"},
    )
    db_session.commit()

    assert entry is not None
    assert entry.id is not None
    assert entry.ticket_id == ticket.id
    assert entry.event_type == "created"
    assert entry.actor == "system/imap"
    assert '"sender": "test@example.com"' in entry.details


def test_ticket_audit_logs_relationship_and_cascade(db_session):
    ticket = Ticket(
        subject="VPN Issue",
        body="Cannot connect",
        sender_email="emp@example.com",
    )
    db_session.add(ticket)
    db_session.commit()

    record_audit_event(db_session, ticket.id, "created", "system/imap")
    record_audit_event(
        db_session,
        ticket.id,
        "classified",
        "system/rag",
        details={"category": "it_support", "assignee_id": 2},
    )
    db_session.commit()

    logs = db_session.query(TicketAuditLog).filter_by(ticket_id=ticket.id).all()
    assert len(logs) == 2
    assert len(ticket.audit_logs) == 2

    # Cascade delete verification
    db_session.delete(ticket)
    db_session.commit()
    logs_after = db_session.query(TicketAuditLog).filter_by(ticket_id=ticket.id).all()
    assert len(logs_after) == 0


def test_record_audit_event_fk_failure_rolls_back_and_keeps_session_usable(db_session):
    """A failed audit write must roll back (no poisoned session) and return None.

    Without the rollback the session is left in a PendingRollbackError state and
    every later interaction throws, silently breaking the sync loop.
    """
    entry = record_audit_event(db_session, 999999, "created", "system/imap")
    assert entry is None

    # The session must still be usable for subsequent work.
    ticket = Ticket(
        subject="Still works",
        body="After a failed audit write",
        sender_email="emp@example.com",
    )
    db_session.add(ticket)
    db_session.commit()
    assert ticket.id is not None

    saved = record_audit_event(db_session, ticket.id, "created", "system/imap")
    db_session.commit()
    assert saved is not None
    assert saved.ticket_id == ticket.id


def test_record_audit_event_missing_ticket_id_returns_none(db_session):
    assert record_audit_event(db_session, 0, "created", "system/imap") is None
    assert record_audit_event(db_session, None, "created", "system/imap") is None
