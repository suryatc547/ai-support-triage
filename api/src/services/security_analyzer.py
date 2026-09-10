"""Optional LLM-assisted security classification (hybrid guardrail).

The deterministic guardrails in :mod:`validation_service` form the security
baseline and never depend on a model. For genuinely ambiguous cases — a mail
that was *flagged* but not *quarantined* — this module optionally consults the
LLM provider chain to decide whether the mail looks like a phishing or social
engineering attempt.

It is strictly best-effort and one-directional:
- Any LLM failure is swallowed and the deterministic verdict is kept.
- The LLM can only *escalate* a ``flag`` to ``quarantine``; it can never reduce
  a hard block or make a blocked mail look safe.
"""

import json
import logging

from . import rag

logger = logging.getLogger(__name__)


def _refine_prompt() -> str:
    return (
        "You are an email security classifier. Decide whether the following "
        "incoming email is likely a phishing or social-engineering attempt.\n\n"
        "Respond ONLY with a JSON object with exactly one key:\n"
        '- "suspicious": true or false (a JSON boolean).\n\n'
        "Be conservative: only mark true when there is a clear social-engineering "
        "or phishing signal (urgency, credential/bank/OTP requests, spoofed "
        "trusted identity, suspicious links, impersonation). Prefer false for "
        "ordinary support requests."
    )


def analyze_suspicious(
    subject: str,
    body: str,
    sender_email: str,
    providers: list[rag.LLMProvider] | None = None,
) -> bool:
    """Return True if the LLM believes the mail is a phishing/social-engineering attempt.

    Best-effort: returns False (safe) on any provider failure so the
    deterministic guardrail verdict stands.

    :param providers: Optional provider chain. Defaults to ``rag.select_llm()``.
    """
    providers = providers if providers is not None else rag.select_llm()
    if not providers:
        logger.debug("No LLM provider configured; skipping LLM security refinement")
        return False

    messages = [
        ("system", _refine_prompt()),
        (
            "human",
            f"Subject: {subject}\nSender: {sender_email}\n\n{body}",
        ),
    ]

    for llm in providers:
        try:
            response = rag.invoke_with_retry(llm, messages)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:].rstrip("` \n")
            elif content.startswith("```"):
                content = content[3:].rstrip("` \n")
            parsed = json.loads(content)
            return bool(parsed.get("suspicious", False))
        except Exception as e:  # noqa: BLE001 - best-effort, never block on LLM
            logger.warning("Security LLM %s failed: %s", rag.provider_label(llm), e)
            continue

    return False


def refine_result(
    result,
    subject: str,
    body: str,
    sender_email: str,
    providers: list[rag.LLMProvider] | None = None,
):
    """Return a refined validation result, escalating a `flag` to `quarantine`
    if the LLM judges the mail suspicious.

    The deterministic result is returned unchanged unless the LLM explicitly
    flags the mail as suspicious. Subject/body/sender are passed explicitly so
    this helper stays free of hidden state.
    """
    from .validation_service import Finding, ValidationResult

    if result.verdict != "flag":
        return result

    try:
        suspicious = analyze_suspicious(subject, body, sender_email, providers=providers)
    except Exception:  # noqa: BLE001 - best-effort
        return result

    if not suspicious:
        return result

    findings = list(result.findings) + [
        Finding(
            "llm_suspicious",
            "high",
            "LLM classified the email as a likely phishing/social-engineering attempt.",
        )
    ]
    logger.warning("LLM flagged mail as suspicious; escalated flag -> quarantine")
    return ValidationResult(
        verdict="quarantine",
        score=result.score + 60,
        findings=findings,
        is_trusted=result.is_trusted,
        quarantined_reason="LLM flagged as phishing/social engineering",
    )
