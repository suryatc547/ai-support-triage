"""Tests for the Support Agentic Harness, tools, policy guardrails, and orchestration."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.harness import (
    AgentDecision,
    AgentState,
    check_staff_workload_tool,
    draft_resolution_tool,
    evaluate_policy_rules,
    lookup_customer_history_tool,
    run_support_agent,
    sanitize_input,
    search_kb_tool,
)
from src.models.models import Base, Ticket, TicketAuditLog, User

IT_USER_ID = 2
HR_USER_ID = 3
SECURITY_USER_ID = 4


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed mock staff
    it_user = User(
        id=IT_USER_ID, name="IT Admin", email="it-admin@test.com", department="IT Support"
    )
    hr_user = User(id=HR_USER_ID, name="HR Team", email="hr@test.com", department="Human Resources")
    sec_user = User(
        id=SECURITY_USER_ID,
        name="Security Team",
        email="security@test.com",
        department="Security & Compliance",
    )
    session.add_all([it_user, hr_user, sec_user])
    session.commit()

    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_users(db_session):
    return db_session.query(User).all()


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


class FakeProvider:
    def __init__(self, response_dict: dict):
        self.response_dict = response_dict
        self.invocations = 0
        self.calls: list = []

    def invoke(self, messages):
        self.invocations += 1
        self.calls.append(messages)
        return FakeLLMResponse(json.dumps(self.response_dict))


def test_sanitize_input_masks_pii():
    state = AgentState(
        subject="Payment issue 4242 4242 4242 4242",
        body="My password: superSecret123\nPhone: 555-123-4567",
        sender_email="user@test.com",
    )
    sanitize_input(state)
    assert "4242 4242 4242 4242" not in state.sanitized_subject
    assert "[REDACTED_CREDIT_CARD]" in state.sanitized_subject
    assert "superSecret123" not in state.sanitized_body
    assert "[REDACTED_SECRET]" in state.sanitized_body
    assert "[REDACTED_PHONE]" in state.sanitized_body


def test_search_kb_tool(mock_users):
    results = search_kb_tool(
        query="VPN connectivity laptop issue", department="it_support", users=mock_users
    )
    assert isinstance(results, list)
    assert len(results) > 0
    assert any(
        "it_support" in str(r.get("source", "")).lower() or "vpn" in r.get("snippet", "").lower()
        for r in results
    )


def test_lookup_customer_history_tool(db_session):
    # Insert existing ticket
    t = Ticket(
        subject="Past VPN problem",
        body="VPN disconnected",
        sender_email="regular@company.com",
        ticket_type="it_support",
        status="resolved",
    )
    db_session.add(t)
    db_session.commit()

    history = lookup_customer_history_tool("regular@company.com", db_session)
    assert len(history) == 1
    assert history[0]["ticket_id"] == t.id
    assert history[0]["category"] == "it_support"


def test_check_staff_workload_tool(db_session):
    workloads = check_staff_workload_tool(db_session, department="IT Support")
    assert len(workloads) == 1
    assert workloads[0]["user_id"] == IT_USER_ID
    assert workloads[0]["open_tickets"] == 0


def test_draft_resolution_tool():
    it_draft = draft_resolution_tool("it_support", "My VPN is stuck on connecting", "")
    assert "VPN" in it_draft
    assert "MFA" in it_draft

    hr_draft = draft_resolution_tool("hr", "How do I submit leave?", "")
    assert "HR" in hr_draft
    assert "leave" in hr_draft


def test_policy_guard_security_escalation_override(mock_users):
    state = AgentState(
        subject="Urgent",
        body="My account was hacked and someone else logged in!",
        sender_email="victim@company.com",
    )
    initial_decision = AgentDecision(
        action="assign",
        category="it_support",
        assignee_id=IT_USER_ID,
        confidence=0.90,
        reasoning="User has login issue",
    )

    calibrated = evaluate_policy_rules(state, initial_decision, mock_users)
    assert calibrated.action == "escalate"
    assert calibrated.category == "security_compliance"
    assert calibrated.assignee_id == SECURITY_USER_ID
    assert calibrated.hitl_required is True
    assert any("escalation_trigger" in f for f in calibrated.policy_flags)


def test_policy_guard_low_confidence_gate(mock_users):
    state = AgentState(
        subject="Random ambiguous message",
        body="Stuff isn't working today",
        sender_email="emp@company.com",
    )
    initial_decision = AgentDecision(
        action="assign",
        category="facilities",
        assignee_id=1,
        confidence=0.55,
        reasoning="Weak guess",
    )

    calibrated = evaluate_policy_rules(state, initial_decision, mock_users)
    assert calibrated.action == "needs_review"
    assert calibrated.hitl_required is True
    assert "low_confidence_review" in calibrated.policy_flags
    # Fail-closed: a needs-review decision must never keep a resolved assignment.
    assert calibrated.assignee_id is None
    assert calibrated.category == "needs_review"


def test_search_kb_tool_department_filter_scopes_to_metadata(mock_users):
    """Dept scoping must come from the chunk metadata, not bare substring matches."""
    from src.harness import search_kb_tool

    it_only = search_kb_tool(
        "VPN connectivity laptop issue", department="it_support", users=mock_users, k=10
    )
    assert it_only
    assert all("it_support" in r["source"] for r in it_only)

    hr_only = search_kb_tool(
        "annual leave balance question", department="hr", users=mock_users, k=10
    )
    assert hr_only
    assert all("human_resources" in r["source"] for r in hr_only)


def test_orchestrator_fails_closed_on_missing_confidence(mock_users, db_session):
    fake_llm = FakeProvider(
        {
            "action": "assign",
            "category": "it_support",
            "assignee_id": IT_USER_ID,
            "reasoning": "no confidence emitted",
        }
    )
    decision = run_support_agent(
        subject="VPN is down",
        body="cannot connect to the VPN client",
        sender_email="emp@company.com",
        db=db_session,
        providers=[fake_llm],
        users=mock_users,
    )
    # Missing confidence must not default to a routable value.
    assert decision.confidence == 0.0
    assert decision.hitl_required is True
    assert decision.action == "needs_review"
    assert decision.assignee_id is None


def test_orchestrator_rejects_unknown_action_without_assignment(mock_users, db_session):
    fake_llm = FakeProvider(
        {
            "action": "self_destruct",
            "category": "it_support",
            "assignee_id": IT_USER_ID,
            "confidence": 0.95,
            "reasoning": "off-spec action token",
        }
    )
    decision = run_support_agent(
        subject="Laptop broken",
        body="screen cracked",
        sender_email="emp@company.com",
        db=db_session,
        providers=[fake_llm],
        users=mock_users,
    )
    assert decision.action == "needs_review"
    assert decision.assignee_id is None
    assert decision.policy_flags


def test_run_support_agent_end_to_end_trajectory(db_session, mock_users):
    ticket = Ticket(
        subject="Cannot connect to corporate VPN",
        body="Getting timeout on gateway",
        sender_email="john@company.com",
    )
    db_session.add(ticket)
    db_session.commit()

    fake_llm = FakeProvider(
        {
            "action": "assign",
            "category": "it_support",
            "assignee_id": IT_USER_ID,
            "confidence": 0.95,
            "reasoning": "Standard VPN connectivity request matching IT Support policy",
            "draft_reply": "Please restart your VPN client.",
        }
    )

    decision = run_support_agent(
        subject=ticket.subject,
        body=ticket.body,
        sender_email=ticket.sender_email,
        db=db_session,
        ticket_id=ticket.id,
        providers=[fake_llm],
        users=mock_users,
    )

    assert fake_llm.invocations == 1
    assert decision.category == "it_support"
    assert decision.assignee_id == IT_USER_ID
    assert decision.confidence == 0.95
    assert decision.hitl_required is False
    assert decision.suggested_reply == "Please restart your VPN client."

    # Verify audit log was recorded with full trajectory
    audit_logs = db_session.query(TicketAuditLog).filter_by(ticket_id=ticket.id).all()
    assert len(audit_logs) == 1
    log = audit_logs[0]
    assert log.event_type == "agent_trajectory"
    assert log.actor == "system/agentic_harness"
    payload = json.loads(log.details)
    assert "steps" in payload
    assert len(payload["steps"]) >= 4  # Sanitization, Fast SLM/history, KB, synthesis, guardrails
    assert payload["decision"]["category"] == "it_support"


def test_run_support_agent_security_guardrail_override(db_session, mock_users):
    """Verifies that the guardrail overrides LLM misclassification on security triggers."""
    fake_llm = FakeProvider(
        {
            "action": "assign",
            "category": "it_support",
            "assignee_id": IT_USER_ID,
            "confidence": 0.88,
            "reasoning": "Password issue",
            "draft_reply": "Reset your password",
        }
    )

    decision = run_support_agent(
        subject="Security incident",
        body="Someone else logged in and we suspect a data breach!",
        sender_email="alert@company.com",
        db=db_session,
        providers=[fake_llm],
        users=mock_users,
    )

    # Policy guardrail must force escalation to security despite LLM returning IT
    assert decision.action == "escalate"
    assert decision.category == "security_compliance"
    assert decision.assignee_id == SECURITY_USER_ID
    assert decision.hitl_required is True
    assert any("escalation_trigger" in f for f in decision.policy_flags)


def test_orchestrator_prompt_redacts_sender_and_history(db_session, mock_users):
    """No raw customer PII may reach the remote provider prompt."""
    db_session.add(
        Ticket(
            subject="Refund for jane.doe@example.com card 4242 4242 4242 4242",
            body="historical ticket",
            sender_email="jane.doe@example.com",
            ticket_type="finance_procurement",
            status="resolved",
        )
    )
    db_session.commit()

    fake_llm = FakeProvider(
        {
            "action": "assign",
            "category": "it_support",
            "assignee_id": IT_USER_ID,
            "confidence": 0.95,
            "reasoning": "VPN issue",
        }
    )
    run_support_agent(
        subject="Refund question",
        body="Where is my refund?",
        sender_email="jane.doe@example.com",
        db=db_session,
        providers=[fake_llm],
        users=mock_users,
    )

    assert fake_llm.calls, "provider was never invoked"
    system_prompt = fake_llm.calls[0][0][1]
    assert "jane.doe@example.com" not in system_prompt
    assert "[REDACTED_EMAIL]" in system_prompt
    assert "4242 4242 4242 4242" not in system_prompt
    # The staff directory in the prompt must not carry work email addresses.
    for user in mock_users:
        assert user.email not in system_prompt


def test_orchestrator_ignores_non_string_fast_intent(monkeypatch, db_session, mock_users):
    """Off-spec local-model output must not crash the fast-path fallback."""

    class FakeSLM:
        def invoke(self, messages):
            return FakeLLMResponse(json.dumps({"category": ["it_support"], "confidence": 0.99}))

    monkeypatch.setattr("src.harness.orchestrator._get_local_llm", lambda: FakeSLM())

    fake_llm = FakeProvider(
        {
            "action": "assign",
            "category": "it_support",
            "assignee_id": IT_USER_ID,
            "confidence": 0.95,
            "reasoning": "VPN issue",
        }
    )
    decision = run_support_agent(
        subject="VPN is down",
        body="cannot connect to the VPN client",
        sender_email="emp@company.com",
        db=db_session,
        providers=[fake_llm],
        users=mock_users,
    )

    assert decision.category == "it_support"
    assert decision.assignee_id == IT_USER_ID
