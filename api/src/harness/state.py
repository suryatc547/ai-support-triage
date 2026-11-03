"""State schema and trajectory data models for the Support Agentic Harness."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolCall:
    """Record of a tool invocation within an agent step."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    duration_ms: float = 0.0


@dataclass
class AgentStep:
    """A single step / turn in the agent reasoning trajectory."""

    thought: str
    tool_call: ToolCall | None = None
    timestamp: str = field(default_factory=_utcnow_iso)


@dataclass
class AgentDecision:
    """Final calibrated decision produced by the agentic harness."""

    action: str  # "assign" | "auto_resolve" | "escalate" | "quarantine" | "needs_review"
    category: str
    assignee_id: int | None
    confidence: float
    reasoning: str
    suggested_reply: str | None = None
    hitl_required: bool = False
    policy_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """Working state container passed through the agentic harness execution."""

    subject: str
    body: str
    sender_email: str
    ticket_id: int | None = None
    sanitized_subject: str = ""
    sanitized_body: str = ""
    fast_intent: str | None = None
    fast_confidence: float = 0.0
    trajectory: list[AgentStep] = field(default_factory=list)
    decision: AgentDecision | None = None

    def add_step(self, thought: str, tool_call: ToolCall | None = None) -> AgentStep:
        step = AgentStep(thought=thought, tool_call=tool_call)
        self.trajectory.append(step)
        return step

    def trajectory_summary(self) -> list[dict[str, Any]]:
        """Serialize trajectory steps for persistence in audit logs."""
        summary = []
        for step in self.trajectory:
            entry = {
                "thought": step.thought,
                "timestamp": step.timestamp,
            }
            if step.tool_call:
                entry["tool"] = {
                    "name": step.tool_call.tool_name,
                    "arguments": step.tool_call.arguments,
                    "result": step.tool_call.result,
                    "duration_ms": step.tool_call.duration_ms,
                }
            summary.append(entry)
        return summary
