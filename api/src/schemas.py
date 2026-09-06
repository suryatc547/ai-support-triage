"""Pydantic response schemas for the public API.

These models define the exact shape of every JSON response so the API contract
is explicit and validated at runtime (FastAPI serialises through them). They
also produce accurate OpenAPI/Swagger documentation.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class AssigneeOut(BaseModel):
    """A support staff member as exposed on a ticket (without internal fields)."""

    id: int
    name: str
    email: str


class ValidationFindingOut(BaseModel):
    """A single security guardrail finding attached to a ticket."""

    type: str
    severity: str
    message: str


class TicketOut(BaseModel):
    """A ticket as returned by the API."""

    id: int
    subject: str
    body: str
    sender_email: str
    ticket_type: str | None = None
    status: str | None = None
    created_at: datetime
    assignee: AssigneeOut | None = None
    security_flag: str | None = None
    suspicion_score: int = Field(default=0)
    validation_findings: list[ValidationFindingOut] = Field(default_factory=list)


class TicketPageOut(BaseModel):
    """Paginated ticket list response."""

    items: list[TicketOut]
    total: int
    page: int
    limit: int
    pages: int


class UserOut(BaseModel):
    """A support staff member as returned by the users endpoint."""

    id: int
    name: str
    department: str | None = None
