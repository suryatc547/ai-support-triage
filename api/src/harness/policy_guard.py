"""Deterministic policy guardrails and safety checks for the Agentic Harness."""

import re

from ..services.pii_service import redact_pii
from .state import AgentDecision, AgentState

# Escalation triggers that mandate immediate routing to the Security Team
_SECURITY_ESCALATION_REGEX = re.compile(
    r"(?i)\b("
    r"account\s+hacked|someone\s+else\s+logged\s+in|suspicious\s+login|"
    r"phishing|malware|ransomware|data\s+breach|unauthorized\s+access|"
    r"credential\s+theft|exfiltration|compromised\s+account"
    r")\b"
)

# High-risk financial approval keywords requiring human sign-off
_FINANCIAL_APPROVAL_REGEX = re.compile(
    r"(?i)\b(wire\s+transfer|payout|bank\s+details|tax\s+refund|invoice\s+payment)\b"
)

# Below this confidence the harness never auto-assigns: the decision is forced
# to human review. Kept distinct from rag.py's 0.70 RAG gate on purpose — the
# harness additionally blocks risky *actions*, not just routing confidence.
_HITL_CONFIDENCE_THRESHOLD = 0.75


def sanitize_input(state: AgentState) -> None:
    """Mask sensitive PII and secrets in the subject and body before agent processing."""
    state.sanitized_subject = redact_pii(state.subject or "")
    state.sanitized_body = redact_pii(state.body or "")


def evaluate_policy_rules(state: AgentState, decision: AgentDecision, users: list) -> AgentDecision:
    """Enforce deterministic company policies and safety constraints onto the agent decision.

    Guarantees that security breaches always route to Security & Compliance,
    and flags low-confidence or high-risk actions for Human-in-the-Loop (HITL) review.
    """
    combined_text = f"{state.subject}\n{state.body}"
    policy_flags = list(decision.policy_flags)

    # 1. Mandatory Security Escalation Trigger Check
    security_match = _SECURITY_ESCALATION_REGEX.search(combined_text)
    if security_match:
        trigger_word = security_match.group(0)
        policy_flags.append(f"escalation_trigger:{trigger_word.lower()}")

        # Locate the Security Team user ID
        security_user = next(
            (
                u
                for u in users
                if "security" in (getattr(u, "department", "") or "").lower()
                or "security" in (getattr(u, "name", "") or "").lower()
            ),
            None,
        )

        decision.action = "escalate"
        decision.category = "security_compliance"
        decision.assignee_id = getattr(security_user, "id", None)
        decision.confidence = 1.0
        decision.reasoning = (
            f"Mandatory security escalation triggered by detected keyword: '{trigger_word}'."
        )
        decision.hitl_required = True

    # 2. Financial approvals over critical operations require HITL
    if _FINANCIAL_APPROVAL_REGEX.search(combined_text):
        policy_flags.append("financial_compliance_review")
        decision.hitl_required = True

    # 3. Confidence Threshold Gatekeeper (< 0.75 forces human review)
    if decision.confidence < _HITL_CONFIDENCE_THRESHOLD:
        policy_flags.append("low_confidence_review")
        decision.hitl_required = True
        if decision.action not in ("escalate", "quarantine"):
            # Fail-closed: strip the assignment and mark the category for human
            # triage (mirrors rag.py's `_clean_llm_result`, which nulls
            # `assignee_id` below its own threshold). Leaving the assignee set
            # on a `needs_review` decision would let a consumer auto-assign a
            # decision the harness explicitly wanted vetted.
            decision.action = "needs_review"
            decision.category = "needs_review"
            decision.assignee_id = None

    # 4. If action is quarantine or escalate, ensure HITL is flagged
    if decision.action in ("quarantine", "escalate"):
        decision.hitl_required = True

    decision.policy_flags = policy_flags
    return decision
