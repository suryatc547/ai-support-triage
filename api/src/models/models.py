from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime.

    ``datetime.utcnow()`` is deprecated since Python 3.12, so we build an
    explicit UTC timestamp instead. Storing naive timestamps would make the
    ``tzinfo`` comparison in SQLite ambiguous, so we keep the object aware.
    """
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    department = Column(String)
    expertise = Column(Text)  # Rich text description of support areas for BM25 + LLM context

    tickets = relationship("Ticket", back_populates="assignee")


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, index=True, nullable=True)  # dedup key
    subject = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    sender_email = Column(String, nullable=False)
    ticket_type = Column(String)  # Categorized by AI
    status = Column(String, default="open")
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    security_flag = Column(String, nullable=True)  # 'quarantined' | None
    suspicion_score = Column(Integer, default=0)  # 0-100 cumulative guardrail score
    validation_findings = Column(Text, nullable=True)  # JSON-encoded list of findings
    created_at = Column(DateTime, default=_utcnow)

    assignee = relationship("User", back_populates="tickets")
    audit_logs = relationship(
        "TicketAuditLog", back_populates="ticket", cascade="all, delete-orphan"
    )


class TicketAuditLog(Base):
    __tablename__ = "ticket_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(
        Integer, ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event_type = Column(
        String, nullable=False, index=True
    )  # created/quarantined/classified/needs_review/forwarded/agent_trajectory/resolved
    actor = Column(String, nullable=False)  # 'system/imap', 'system/rag', 'system/smtp', 'user/1'
    details = Column(Text, nullable=True)  # JSON-encoded metadata/state changes
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    ticket = relationship("Ticket", back_populates="audit_logs")
