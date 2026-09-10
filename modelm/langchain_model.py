"""LangChain BaseChatModel adapter for the local custom support classifier.

Allows the custom domain-trained model to participate natively in LangChain
chains and provider fallback sequences.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from .model import CustomTicketClassifier

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"

# Mapping from category to typical keywords/aliases for assignee matching
CATEGORY_TO_DEPARTMENT = {
    "it_support": ("it support", "it admin", "it team"),
    "finance_procurement": ("finance & procurement", "finance", "procurement", "admin team"),
    "human_resources": ("human resources", "hr", "hr team"),
    "security_compliance": ("security & compliance", "security", "security team"),
    "facilities_admin": ("facilities & admin", "facilities", "facilities team"),
}


def _name_tokens(text: str) -> set[str]:
    """Lowercased, punctuation-stripped word tokens used for boundary-safe matching."""
    return set(re.findall(r"\b[a-z0-9_-]+\b", (text or "").lower()))


class LocalSupportChatModel(BaseChatModel):
    """A native LangChain ChatModel powered by our custom local model.

    Runs entirely on CPU in < 5ms with ~15MB RAM.
    """

    model_name: str = "custom-support-local-v1"
    weights_path: str = Field(default_factory=lambda: str(DEFAULT_WEIGHTS_PATH))
    # Minimum confidence for an assignment. Low-confidence predictions are left
    # category "unclassified" (unassigned) for manual review rather than guessed.
    confidence_threshold: float = 0.40
    temperature: float = 1.0

    _classifier: CustomTicketClassifier | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._classifier is None:
            path = Path(self.weights_path)
            if not path.exists():
                logger.info("Weights not found at %s. Training local model...", path)
                from .train import train_and_save

                self._classifier, _ = train_and_save(path)
            else:
                self._classifier = CustomTicketClassifier.load(path)

    @property
    def _llm_type(self) -> str:
        return "custom-local-support-model"

    def _extract_ticket_text(
        self, messages: Sequence[BaseMessage | tuple[str, str]]
    ) -> tuple[str, str, str]:
        """Extract subject, body, and user_context from prompt messages."""
        full_text = ""
        for m in messages:
            if isinstance(m, tuple):
                full_text += f"\n{m[1]}"
            elif hasattr(m, "content"):
                full_text += f"\n{m.content}"

        # Extract Subject and Body from standard RAG prompt template without
        # leaking subsequent sections (e.g., ROUTING POLICY CONTEXT).
        subject_match = re.search(r"Subject:\s*(.*?)(?:\n|$)", full_text, re.IGNORECASE)
        body_match = re.search(
            r"Body:\s*([\s\S]*?)(?:\n\n---|\n---|\n\nAnalyze|\nAnalyze|\n\nRoute|\nRoute|$)",
            full_text,
            re.IGNORECASE,
        )

        subject = subject_match.group(1).strip() if subject_match else ""
        body = body_match.group(1).strip() if body_match else ""

        if not subject and not body:
            # Fallback to entire text
            body = full_text.strip()

        return subject, body, full_text

    def _find_assignee_id(self, category: str, full_prompt: str) -> int | None:
        """Find the best matching user ID from the user_context injected into the prompt.

        Matching is order-independent and precedence-based: exact department
        match wins, then exact team/name, then a multi-word alias substring, and
        finally a word-boundary token match. This avoids substring aliases like
        ``"it"`` matching unrelated teams (e.g. ``"facilities team"``) so the
        routing outcome no longer depends on the order staff are listed in.
        """
        aliases = CATEGORY_TO_DEPARTMENT.get(category, (category.replace("_", " "),))
        if isinstance(aliases, str):
            aliases = (aliases,)
        aliases = [a.strip().lower() for a in aliases if a.strip()]

        blocks = re.split(r"(?=ID:\s*\d+)", full_prompt, flags=re.IGNORECASE)
        candidates: list[dict[str, Any]] = []
        for block in blocks:
            id_match = re.search(r"ID:\s*(\d+)", block, re.IGNORECASE)
            if not id_match:
                continue
            try:
                uid = int(id_match.group(1))
            except ValueError:
                continue
            name_m = re.search(r"Name:\s*([^,\n]+)", block, re.IGNORECASE)
            dept_m = re.search(r"Department:\s*([^\n]+)", block, re.IGNORECASE)
            candidates.append(
                {
                    "id": uid,
                    "name": name_m.group(1).strip().lower() if name_m else "",
                    "dept": dept_m.group(1).strip().lower() if dept_m else "",
                }
            )

        def match_score(cand: dict[str, Any]) -> int:
            """Rank a candidate by how precisely its dept/name matches an alias."""
            name, dept = cand["name"], cand["dept"]
            for a in aliases:
                if dept and dept == a:
                    return 4
            for a in aliases:
                if name and name == a:
                    return 3
            for a in aliases:
                if dept and len(a.split(" ")) > 1 and a in dept:
                    return 2
            for a in aliases:
                alias_tokens = _name_tokens(a)
                if not alias_tokens:
                    continue
                if alias_tokens <= _name_tokens(dept) or alias_tokens <= _name_tokens(name):
                    return 1
            return 0

        best = max(candidates, key=match_score, default=None)
        if best is not None and match_score(best) > 0:
            return best["id"]

        # Fallback query to SQLite database if available
        try:
            from api.src.models.database import SessionLocal
            from api.src.models.models import User

            db = SessionLocal()
            try:
                best_user = None
                best_rank = 0
                for u in db.query(User).all():
                    rank = match_score(
                        {
                            "name": (u.name or "").lower(),
                            "dept": (u.department or "").lower(),
                        }
                    )
                    if rank > best_rank:
                        best_rank = rank
                        best_user = u
                if best_user is not None:
                    return best_user.id
            finally:
                db.close()
        except (ImportError, RuntimeError, OSError) as e:
            logger.debug("DB fallback lookup skipped: %s", e)

        return None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_loaded()
        assert self._classifier is not None

        subject, body, full_prompt = self._extract_ticket_text(messages)
        query = f"{subject}\n{body}".strip()

        category, confidence = self._classifier.predict(query, temperature=self.temperature)

        if confidence < self.confidence_threshold:
            category = "unclassified"
            assignee_id = None
        else:
            assignee_id = self._find_assignee_id(category, full_prompt)

        payload = {
            "category": category,
            "assignee_id": assignee_id,
        }
        json_output = json.dumps(payload)

        logger.info(
            "Local model predicted category '%s' (confidence: %.2f, assignee_id: %s)",
            category,
            confidence,
            assignee_id,
        )

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json_output))])
