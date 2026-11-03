"""Tiered Orchestrator and Reasoning Engine for the Support Agentic Harness."""

import json
import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from ..models import models
from ..services.audit_service import record_audit_event
from ..services.pii_service import redact_emails, redact_pii
from ..services.rag import (
    _get_local_llm,
    invoke_with_retry,
    provider_label,
    select_llm,
)
from .policy_guard import evaluate_policy_rules, sanitize_input
from .state import AgentDecision, AgentState, ToolCall
from .tools import (
    check_staff_workload_tool,
    draft_resolution_tool,
    lookup_customer_history_tool,
    search_kb_tool,
)

logger = logging.getLogger(__name__)

_ORCHESTRATOR_PROMPT = """You are an autonomous enterprise support agent and triage specialist.

Analyze the customer inquiry along with gathered tool findings (knowledge base policies,
past customer history, and staff workload) to make a calibrated routing decision.

--- CUSTOMER INQUIRY ---
Subject: {subject}
Body: {body}
Sender: {sender}

--- TOOL FINDINGS: KNOWLEDGE BASE ---
{kb_context}

--- TOOL FINDINGS: CUSTOMER HISTORY ---
{history_context}

--- AVAILABLE STAFF & WORKLOAD ---
{staff_context}

Respond ONLY with a valid JSON object with the following keys:
- "action": "assign" (standard routing) or "auto_resolve" (if standard inquiry)
- "category": a short lowercase string (e.g. "it_support", "hr", "finance", "security")
- "assignee_id": the integer ID of the best matched staff member, or null
- "confidence": a float between 0.0 and 1.0 representing confidence
- "reasoning": a one-sentence justification for the decision
- "draft_reply": a polite first-contact response draft, or null
"""


def run_support_agent(
    subject: str,
    body: str,
    sender_email: str,
    db: Session | None = None,
    ticket_id: int | None = None,
    providers: list[Any] | None = None,
    users: list[Any] | None = None,
) -> AgentDecision:
    """Execute the Agentic Harness loop over an inbound support ticket."""
    start_time = time.time()
    state = AgentState(
        subject=subject,
        body=body,
        sender_email=sender_email,
        ticket_id=ticket_id,
    )

    # 1. PII Redaction & Sanitization
    sanitize_input(state)
    state.add_step(
        "Sanitized inbound inquiry; masked sensitive PII and secrets prior to tool dispatch"
    )

    # Resolve users list
    if users is None and db is not None:
        users = db.query(models.User).all()
    users = users or []

    # 2. Phase 1: Fast-Tier Triage check via local SLM (modelm)
    local_slm = _get_local_llm()
    if local_slm is not None:
        t0 = time.time()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            slm_resp = local_slm.invoke(
                [
                    SystemMessage(content="Classify support ticket into category and assignee."),
                    HumanMessage(
                        content=f"Subject: {state.sanitized_subject}\nBody: {state.sanitized_body}"
                    ),
                ]
            )
            raw_data = json.loads(slm_resp.content) if isinstance(slm_resp.content, str) else {}
            # Model output is untrusted: keep only a non-empty string category so
            # a list/dict verdict can never reach ``.lower()`` further down.
            raw_category = raw_data.get("category")
            state.fast_intent = (
                raw_category.strip()
                if isinstance(raw_category, str) and raw_category.strip()
                else None
            )
            state.fast_confidence = float(raw_data.get("confidence", 0.0))
            duration = (time.time() - t0) * 1000.0
            state.add_step(
                f"Fast SLM intent: '{state.fast_intent}' (conf {state.fast_confidence:.2f})",
                ToolCall(
                    "modelm_fast_triage", {"subject": state.sanitized_subject}, raw_data, duration
                ),
            )
        except Exception as exc:
            logger.debug("Fast-tier SLM skipped: %s", exc)

    # 3. Phase 2: Tool Invocations
    # Tool A: Customer History Lookup
    t0 = time.time()
    history = lookup_customer_history_tool(state.sender_email, db)
    state.add_step(
        f"Retrieved {len(history)} previous ticket(s) for sender {redact_pii(state.sender_email)}",
        ToolCall(
            "lookup_customer_history",
            {"email": state.sender_email},
            history,
            (time.time() - t0) * 1000.0,
        ),
    )

    # Tool B: Knowledge Base Search
    t0 = time.time()
    kb_results = search_kb_tool(
        query=f"{state.sanitized_subject}\n{state.sanitized_body}",
        department=state.fast_intent,
        users=users,
    )
    state.add_step(
        f"Retrieved {len(kb_results)} relevant knowledge base policy document(s)",
        ToolCall(
            "search_kb",
            {"query": state.sanitized_subject, "dept": state.fast_intent},
            kb_results,
            (time.time() - t0) * 1000.0,
        ),
    )

    # Tool C: Staff Workload Check
    t0 = time.time()
    workloads = check_staff_workload_tool(db, department=state.fast_intent)
    state.add_step(
        f"Inspected active staff workloads for {len(workloads)} user(s)",
        ToolCall(
            "check_staff_workload",
            {"department": state.fast_intent},
            workloads,
            (time.time() - t0) * 1000.0,
        ),
    )

    # 4. Phase 3: Synthesis & Calibrated Reasoning
    provider_chain = providers if providers is not None else select_llm()

    kb_context_str = (
        # KB chunks resolve ``{{contact_email}}`` to a staff address; the remote
        # prompt only needs the policy text, so strip addresses here.
        redact_emails("\n\n".join(f"[{r['source']}]\n{r['snippet']}" for r in kb_results))
        if kb_results
        else "No specific policy matched."
    )
    history_str = (
        "\n".join(
            f"- #{h['ticket_id']}: {redact_pii(h['subject'])} "
            f"(type={h['category']}, status={h['status']})"
            for h in history
        )
        if history
        else "No prior ticket history."
    )
    staff_str = (
        # No staff email here: this string is sent to the remote provider, and
        # routing only needs the numeric ID. Keep contact details server-side.
        "\n".join(f"- User #{u.id}: {u.name} ({u.department})" for u in users)
        if users
        else "No registered staff."
    )

    decision = None
    if provider_chain:
        for llm in provider_chain:
            try:
                prompt_text = _ORCHESTRATOR_PROMPT.format(
                    subject=state.sanitized_subject,
                    body=state.sanitized_body,
                    sender=redact_pii(state.sender_email),
                    kb_context=kb_context_str,
                    history_context=history_str,
                    staff_context=staff_str,
                )
                messages = [
                    ("system", prompt_text),
                    ("human", "Execute agent reasoning and emit decision JSON."),
                ]
                resp = invoke_with_retry(llm, messages)
                content = resp.content.strip()
                if content.startswith("```json"):
                    content = content[7:].rstrip("` \n")
                elif content.startswith("```"):
                    content = content[3:].rstrip("` \n")
                parsed = json.loads(content)

                cat = str(parsed.get("category", "general")).strip().lower()
                assignee_val = parsed.get("assignee_id")
                valid_assignee = (
                    assignee_val
                    if isinstance(assignee_val, int) and any(u.id == assignee_val for u in users)
                    else None
                )
                # Fail-closed confidence: missing, null, or non-numeric confidence
                # becomes 0.0 so the policy guard's HITL gate (and the consumer)
                # treats it as a needs-review decision — never an auto-assignment.
                raw_conf = parsed.get("confidence")
                try:
                    conf = float(raw_conf) if raw_conf is not None else 0.0
                except (TypeError, ValueError):
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))
                reason = str(parsed.get("reasoning", "Multi-step agent routing"))[:250]
                action = str(parsed.get("action", "assign")).strip().lower()
                # Whitelist the agent action; anything else degrades to
                # needs_review so an off-spec response cannot trigger an
                # unvetted side effect (e.g. auto_resolve) downstream — and the
                # assignment is stripped so nothing auto-assigns on it.
                agent_flags: list[str] = []
                if action not in ("assign", "auto_resolve", "escalate", "quarantine"):
                    action = "needs_review"
                    valid_assignee = None
                    agent_flags.append("invalid_action")
                draft = parsed.get("draft_reply")

                decision = AgentDecision(
                    action=action,
                    category=cat,
                    assignee_id=valid_assignee,
                    confidence=conf,
                    reasoning=reason,
                    suggested_reply=draft,
                    policy_flags=agent_flags,
                )
                state.add_step(
                    f"Agent synthesis via {provider_label(llm)}: {cat} (conf {conf:.2f})"
                )
                break
            except Exception as exc:
                logger.debug("Provider %s failed in harness: %s", provider_label(llm), exc)

    # Fallback to local heuristic / fast intent if no remote provider succeeded
    if decision is None:
        target_user = None
        if state.fast_intent and users:
            target_user = next(
                (
                    u
                    for u in users
                    if state.fast_intent.lower() in (getattr(u, "department", "") or "").lower()
                ),
                None,
            )

        draft = draft_resolution_tool(
            state.fast_intent or "general",
            state.sanitized_body,
            kb_context_str,
        )
        assigned_name = getattr(target_user, "name", "unassigned")
        decision = AgentDecision(
            action="assign" if target_user else "needs_review",
            category=state.fast_intent or "general",
            assignee_id=getattr(target_user, "id", None),
            confidence=state.fast_confidence if state.fast_confidence > 0 else 0.60,
            reasoning=f"Fast-path heuristic assignment to {assigned_name}",
            suggested_reply=draft,
        )
        state.add_step("Applied fast-path heuristic synthesis")

    # 5. Phase 4: Deterministic Policy Guardrails
    decision = evaluate_policy_rules(state, decision, users)
    state.decision = decision
    state.add_step(
        f"Guardrails verified: hitl={decision.hitl_required}, flags={decision.policy_flags}"
    )

    # 6. Phase 5: Persist Trajectory in Audit Log
    if db is not None and ticket_id is not None:
        try:
            record_audit_event(
                db=db,
                ticket_id=ticket_id,
                event_type="agent_trajectory",
                actor="system/agentic_harness",
                details={
                    "duration_ms": round((time.time() - start_time) * 1000.0, 1),
                    "steps": state.trajectory_summary(),
                    "decision": decision.to_dict(),
                },
            )
            db.commit()
        except Exception as exc:
            logger.warning("Failed to record agent trajectory audit: %s", exc)

    return decision
