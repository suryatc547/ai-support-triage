import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from ..models import models
from .pii_service import redact_emails, redact_pii

logger = logging.getLogger(__name__)


class InvocationResult(Protocol):
    """Minimal contract for an LLM call result (a string content payload)."""

    content: str


class LLMProvider(Protocol):
    """The single interface the routing chain needs from any provider.

    LangChain chat models satisfy this protocol structurally; test doubles do
    too, so the pipeline can be exercised without importing any concrete
    provider class (Dependency Inversion).
    """

    def invoke(self, messages) -> InvocationResult: ...


# Path to policy documents — located inside src/knowledge_base/
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

# Retry settings for transient 429 (rate limit) throttling on the primary provider.
_PRIMARY_RETRIES = 3
_PRIMARY_BACKOFF_SECONDS = 2.0

# Remote providers are constructed with bounded request timeouts. A dead/broken
# network must fail fast and fall through the chain — never hold the IMAP sync
# session open on a single hung `llm.invoke` (which would also leave every
# unprocessed mail un-read and re-hang on the next sync).
_REMOTE_REQUEST_TIMEOUT_SECONDS = 45

# Common English filler words that carry no routing signal. Filtering them out of
# both the query and the index docs avoids spurious BM25 matches (e.g. a generic
# "how does this work" matching prose words inside the policy documents).
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "of",
        "on",
        "or",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
        "hello",
        "thanks",
        "thank",
        "hi",
        "dear",
        "please",
        "kindly",
        "regards",
    ]
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, and tokenize query/doc, dropping stopwords."""
    tokens = re.findall(r"\b[a-z0-9_-]+\b", (text or "").lower())
    return [tok for tok in tokens if tok not in _STOPWORDS]


_SYSTEM_PROMPT = """You are an intelligent enterprise support ticket routing system.

Analyze the incoming support email below and:
1. Identify the category of support needed.
2. Select the best support staff member to assign the ticket to.

Use the routing policy context and the staff list to make your decision.

--- INCOMING TICKET ---
Subject: {subject}
Body: {body}

--- ROUTING POLICY CONTEXT ---
{policy_context}

--- AVAILABLE SUPPORT STAFF ---
{user_context}

Respond ONLY with a valid JSON object with exactly four keys:
- "category": a short lowercase string (e.g., "finance", "it_support", "hr",
  "security", "facilities")
- "assignee_id": the integer ID of the best matched staff member, or null if no match
- "confidence": a float between 0.0 and 1.0 representing your confidence in this
  routing decision
- "reasoning": a brief one-sentence explanation of your routing decision
"""

_HUMAN_PROMPT = "Route the following support ticket. Return ONLY the JSON object."


def _get_primary_llm() -> LLMProvider | None:
    """Instantiate the primary Gemini LLM if an API key is configured."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"),
        temperature=0,
        timeout=_REMOTE_REQUEST_TIMEOUT_SECONDS,
        max_retries=2,
    )


def _get_fallback_llms() -> list[LLMProvider]:
    """Build the ordered list of fallback LLM providers.

    OpenRouter is served through the OpenAI-compatible client with a custom
    base URL; Groq uses its own langchain integration. Each entry returns None
    if its API key is missing so unavailable providers are skipped.
    """
    fallbacks = []

    if os.environ.get("OPENROUTER_API_KEY"):
        fallbacks.append(
            ChatOpenAI(
                model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
                temperature=0,
                request_timeout=_REMOTE_REQUEST_TIMEOUT_SECONDS,
                max_retries=2,
            )
        )

    if os.environ.get("GROQ_API_KEY"):
        fallbacks.append(
            ChatGroq(
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.environ["GROQ_API_KEY"],
                temperature=0,
                request_timeout=_REMOTE_REQUEST_TIMEOUT_SECONDS,
                max_retries=2,
            )
        )

    return fallbacks


def invoke_with_retry(llm: LLMProvider, prompt_values):
    """Invoke an LLM, retrying on 429 rate limits with a short backoff.

    Any non-429 failure (or exhaustion) is re-raised so the caller can fall
    through to the next provider.
    """
    for attempt in range(_PRIMARY_RETRIES):
        try:
            return llm.invoke(prompt_values)
        except Exception as e:  # noqa: BLE001 - providers surface errors differently
            status = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            if status == 429 and attempt < _PRIMARY_RETRIES - 1:
                backoff = _PRIMARY_BACKOFF_SECONDS * (attempt + 1)
                logger.warning(
                    "LLM rate-limited (429); retrying in %.1fs (attempt %d of %d)",
                    backoff,
                    attempt + 2,
                    _PRIMARY_RETRIES,
                )
                time.sleep(backoff)
                continue
            raise


def provider_label(llm: LLMProvider) -> str:
    """Return a `provider - model` label for a langchain chat model for logging."""
    cls = type(llm).__name__
    if "GoogleGenerativeAI" in cls:
        provider = "gemini"
    elif "Groq" in cls:
        provider = "groq"
    elif "OpenAI" in cls or "OpenRouter" in cls:
        provider = "openrouter"
    else:
        provider = cls.lower()
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    return f"{provider} - {model}"


def _get_local_llm() -> LLMProvider | None:
    """Instantiate the custom local LangChain chat model from modelm/ if enabled."""
    if os.environ.get("USE_LOCAL_MODEL", "true").lower() in ("false", "0", "no"):
        return None
    try:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from modelm.langchain_model import LocalSupportChatModel

        return LocalSupportChatModel()
    except Exception:
        logger.exception("Failed to load local support model from modelm")
        return None


def select_llm() -> list[LLMProvider]:
    """Return an ordered list of configured LLM providers (primary + fallbacks)."""
    providers = []
    use_local_first = os.environ.get("USE_LOCAL_MODEL_FIRST", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    local_llm = _get_local_llm()

    if use_local_first and local_llm is not None:
        providers.append(local_llm)

    primary = _get_primary_llm()
    if primary is not None:
        providers.append(primary)
    providers.extend(_get_fallback_llms())

    if not use_local_first and local_llm is not None:
        providers.append(local_llm)

    return providers


def _fallback_bm25(users, subject, body, threshold=1.0):
    """Assign using the best BM25 match when no LLM provider is configured.

    Uses both the support staff expertise and the knowledge_base routing policy
    documents (linked to each team) to build the search index. Applies a
    confidence threshold so weak or non-matching queries are left unassigned
    rather than guessed into an arbitrary team.
    """
    if not users:
        return {
            "category": "unclassified",
            "assignee_id": None,
            "confidence": 0.0,
            "reasoning": "No support staff available",
        }

    # Build an index of (text, assignee_id) entries from staff expertise plus
    # the knowledge_base policy docs (matched to a team via their contact email).
    # Escalation-trigger sections are excluded: they describe when to route
    # AWAY from the document's own team (e.g. IT's Escalation Triggers list
    # "account hacked" / "phishing"), so indexing them under that team steals
    # keyword matches from the real destination team.
    indexed: list[tuple[str, int]] = []
    for d in _build_user_docs(users):
        indexed.append((d.page_content, d.metadata["id"]))
    for pd in _load_knowledge_base_docs(users):
        section = (pd.metadata or {}).get("section") or ""
        if "escalation" in section.lower():
            continue
        user = _match_kb_doc_to_user(users, pd)
        if user is not None:
            indexed.append((pd.page_content, user.id))

    tokenized = [_tokenize(text) for (text, _id) in indexed]
    if not tokenized:
        return {
            "category": "unclassified",
            "assignee_id": None,
            "confidence": 0.0,
            "reasoning": "Empty BM25 index",
        }

    bm25 = BM25Okapi(tokenized)
    query = _tokenize(f"{subject}\n{body}")
    scores = bm25.get_scores(query)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_idx] < threshold:
        return {
            "category": "unclassified",
            "assignee_id": None,
            "confidence": 0.0,
            "reasoning": f"Best BM25 score {scores[best_idx]:.2f} below threshold {threshold}",
        }

    # Aggregate the per-document scores by team (assignee id): a team's user doc
    # and its own KB policy chunks are evidence *for* that team, not rivals.
    # Routing ambiguity is between teams, so the calibration below compares the
    # best team's score against the strongest *other* team.
    team_scores: dict[int, float] = {}
    for i, score in enumerate(scores):
        team = indexed[i][1]
        if score > team_scores.get(team, 0.0):
            team_scores[team] = score

    best_team, confidence = _team_routing_confidence(team_scores)

    return {
        "category": "general",
        "assignee_id": best_team,
        "confidence": round(confidence, 3),
        "reasoning": f"BM25 keyword match (score {scores[best_idx]:.2f})",
    }


def _team_routing_confidence(team_scores: dict[int, float]) -> tuple[int, float]:
    """Pick the best team and calibrate a [0, 1] routing confidence.

    ``team_scores`` maps an assignee id to the strongest BM25 score across that
    team's documents. Confidence is the ratio ``best / (best + runner_up)``
    comparing the winning team against the strongest *other* team: one decisive
    team maps to ``1.0``, two equally good teams map to ``0.5`` (below the
    needs-review threshold — a coin flip must go to a human). Normalising by the
    best score itself would always return ``1.0`` and carry no signal.
    """
    if not team_scores:
        return 0, 0.0
    best_team = max(team_scores, key=team_scores.get)
    best = team_scores[best_team]
    runner_up = max((s for t, s in team_scores.items() if t != best_team), default=0.0)
    if runner_up <= 0:
        return best_team, 1.0
    return best_team, min(1.0, best / (best + runner_up))


def _match_kb_doc_to_user(users, doc) -> models.User | None:
    """Link a knowledge_base routing policy doc to the matching support User.

    Matches by department, team name, or contact email. Section chunks produced
    by :func:`_chunk_by_headings` carry their team breadcrumb in metadata, which
    is consulted first so every chunk links to its team instead of only the
    intro chunk.
    """
    team = (doc.metadata or {}).get("team") or ""
    dept = (doc.metadata or {}).get("department") or ""

    if dept:
        for u in users:
            if (getattr(u, "department", None) or "").strip().lower() == dept.strip().lower():
                return u
    if team:
        for u in users:
            if (getattr(u, "name", None) or "").strip().lower() == team.strip().lower():
                return u

    # Fallback: regex over raw page_content (works for preamble/intro chunks
    # that still carry the full ## Team / **Department** lines).
    dept_match = re.search(r"\*\*Department:\*\*\s*(.+)", doc.page_content, re.IGNORECASE)
    if dept_match:
        dept_name = dept_match.group(1).strip().lower()
        for u in users:
            if (getattr(u, "department", None) or "").strip().lower() == dept_name:
                return u

    team_match = re.search(r"##\s*Team:\s*(.+)", doc.page_content, re.IGNORECASE)
    if team_match:
        team_name = team_match.group(1).strip().lower()
        for u in users:
            if (getattr(u, "name", None) or "").strip().lower() == team_name:
                return u

    match = re.search(r"contact:\s*\**\s*([\w.+-]+@[\w.-]+)", doc.page_content, re.IGNORECASE)
    if match:
        contact_email = match.group(1).strip().lower()
        for u in users:
            if (getattr(u, "email", None) or "").strip().lower() == contact_email:
                return u
    return None


# Cache of loaded policy docs, invalidated whenever a KB file changes on disk
# (mtime) or the resolved user email list changes. This eliminates per-ticket
# disk I/O and file reads for the common, static case.
_KB_DOCS_CACHE: dict = {"mtime": 0.0, "user_key": "", "docs": None}


def _kb_docs_invalidated(mtime: float, user_key: str) -> bool:
    """Return True when the cached KB docs no longer match the current inputs."""
    cached = _KB_DOCS_CACHE["docs"]
    return (
        cached is None or _KB_DOCS_CACHE["mtime"] != mtime or _KB_DOCS_CACHE["user_key"] != user_key
    )


def _load_knowledge_base_docs(users: list | None = None) -> list[Document]:
    """Load all markdown routing policy documents from the knowledge_base directory.

    Replaces placeholder variables like `{{contact_email}}` with the matching
    department or team user's email directly from the database users list.

    Results are cached between calls; the cache is invalidated when any KB file's
    mtime changes or the resolved user -> email mapping changes. This avoids
    re-reading and re-parsing static policy files for every ticket.
    """
    docs = []
    if not KNOWLEDGE_BASE_DIR.exists():
        return docs

    files = [f for f in KNOWLEDGE_BASE_DIR.iterdir() if f.is_file()]
    if not files:
        return docs

    user_key = ",".join(sorted(getattr(u, "email", "") or "" for u in users or []))
    mtime = max(f.stat().st_mtime for f in files)

    if not _kb_docs_invalidated(mtime, user_key):
        return _KB_DOCS_CACHE["docs"]

    users_by_dept = {}
    users_by_team = {}
    if users:
        for u in users:
            dept = getattr(u, "department", None)
            if dept:
                users_by_dept[dept.strip().lower()] = getattr(u, "email", "")
            name = getattr(u, "name", None)
            if name:
                users_by_team[name.strip().lower()] = getattr(u, "email", "")

    for md_file in KNOWLEDGE_BASE_DIR.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")

        dept_match = re.search(r"\*\*Department:\*\*\s*(.+)", content, re.IGNORECASE)
        dept_name = dept_match.group(1).strip().lower() if dept_match else ""

        team_match = re.search(r"##\s*Team:\s*(.+)", content, re.IGNORECASE)
        team_name = team_match.group(1).strip().lower() if team_match else ""

        resolved_email = users_by_dept.get(dept_name) or users_by_team.get(team_name) or ""

        content = content.replace("{{contact_email}}", resolved_email)

        # Section-level chunking: split by markdown `## ` headings so BM25 and
        # the LLM retrieve the specific relevant section (e.g. `## Escalation
        # Triggers`) instead of a whole team policy file. The parent document
        # team/department breadcrumbs are preserved in each chunk's metadata.
        docs.extend(_chunk_by_headings(content, md_file.name, team_name, dept_name))

    _KB_DOCS_CACHE["mtime"] = mtime
    _KB_DOCS_CACHE["user_key"] = user_key
    _KB_DOCS_CACHE["docs"] = docs
    return docs


def _chunk_by_headings(content: str, source: str, team: str, dept: str = "") -> list[Document]:
    """Split a markdown policy document into heading-delimited chunks.

    A document like ``## Scope of Support`` / ``## Escalation Triggers`` /
    ``## Keywords`` becomes multiple Documents, each tagged with the source file
    and the team/department breadcrumbs so downstream retrieval can scope
    matches precisely.
    """
    chunks: list[Document] = []
    sections = re.split(r"(?m)^(##\s+.+)$", content)

    # Retrieved via regex split: preamble runs before the first `## ` heading.
    preamble = sections[0].strip() if sections and sections[0].strip() else ""
    if preamble:
        chunks.append(
            Document(
                page_content=preamble,
                metadata={
                    "source": source,
                    "section": "preamble",
                    "team": team,
                    "department": dept,
                },
            )
        )

    # Header/body pairs (heading, body) — index from 1 so preamble is skipped.
    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        body = sections[i + 1].strip() if i + 1 < len(sections) else ""
        if not heading:
            continue
        chunk_text = f"{heading}\n{body}" if body else heading
        chunks.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "source": source,
                    "section": heading,
                    "team": team,
                    "department": dept,
                },
            )
        )
    return chunks


def _build_user_docs(users: list) -> list[Document]:
    """Build BM25-indexable documents from User records using their expertise field.

    Staff emails are deliberately omitted: these documents are rendered into the
    remote LLM prompt (``user_context``), and routing only needs the ID/name.
    """
    docs = []
    for u in users:
        expertise_text = u.expertise or f"Support for {u.department} inquiries."
        content = f"Name: {u.name}\nDepartment: {u.department}\nExpertise: {expertise_text}"
        docs.append(Document(page_content=content, metadata={"id": u.id, "name": u.name}))
    return docs


# Cache of tokenized BM25 corpora keyed by a content hash, so per-ticket
# re-tokenization of the (cached) policy documents is skipped.
_BM25_TOKEN_CACHE: dict[str, list] = {}


def _bm25_search(docs: list[Document], query: str, k: int = 5) -> list[Document]:
    """Return the top ``k`` documents most relevant to ``query`` via BM25.

    Local replacement for the sunset ``langchain_community`` BM25Retriever.
    ``rank_bm25`` is already a direct project dependency and the tokeniser is
    shared with :func:`_fallback_bm25`, so stopword noise is filtered the same
    way in both paths. Documents with zero similarity are dropped so clearly
    unrelated mail does not inject noise into the LLM prompt context.

    Tokenized corpora are memoized by a cheap content hash: the knowledge-base
    documents come from :func:`_load_knowledge_base_docs`, which is itself
    cached, so repeated queries over the same corpus avoid re-tokenization.
    """
    tokenized = [_tokenize(doc.page_content) for doc in docs]
    query_tokens = _tokenize(query)

    if not tokenized or not query_tokens:
        return []

    corpus_key = hash(tuple(" ".join(toks) for toks in tokenized))
    bm25 = _BM25_TOKEN_CACHE.get(corpus_key)
    if bm25 is None:
        bm25 = BM25Okapi(tokenized)
        _BM25_TOKEN_CACHE[corpus_key] = bm25

    scores = bm25.get_scores(query_tokens)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked[:k] if scores[i] > 0]


# Below 0.70 confidence, a ticket is safe-fallback to human triage.
_CONFIDENCE_NEEDS_REVIEW_THRESHOLD = 0.70

# Categories the routing pipeline can actually assign. A model verdict outside
# this set (e.g. "unclassified") means the model found no team fit — BM25 must
# not be allowed to override that and invent one from thin keyword overlap.
_KB_ROUTING_CATEGORIES = frozenset(
    {
        "it_support",
        "finance_procurement",
        "human_resources",
        "security_compliance",
        "facilities_admin",
    }
)


def _clean_llm_result(result: dict, users: list) -> dict:
    """Sanitise the LLM's routing response before it touches the database.

    The LLM is a remote, non-deterministic component — its JSON can be off-spec
    (missing keys, wrong types, or an assignee id that does not exist). This
    helper guards the DB against such output:

    - ``category`` is coerced to a short lowercase string.
    - ``assignee_id`` is kept only when it is an integer that refers to a real
      support staff member; anything else becomes ``None`` (unassigned).
    - ``confidence`` is coerced to a float in [0.0, 1.0]. When it is below
      :data:`_CONFIDENCE_NEEDS_REVIEW_THRESHOLD`, the ticket is treated as
      unassigned and marked ``needs_review`` for human triage.
    - ``reasoning`` is coerced to a short string.
    """
    category = result.get("category")
    if not isinstance(category, str) or not category.strip():
        category = "general"
    else:
        category = category.strip().lower()[:50]

    assignee_id = result.get("assignee_id")
    is_valid_assignee = (
        # bool is a subclass of int in Python, so exclude it explicitly.
        not isinstance(assignee_id, bool)
        and isinstance(assignee_id, int)
        and any(u.id == assignee_id for u in users)
    )
    if not is_valid_assignee:
        assignee_id = None

    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.0
    confidence = max(0.0, min(float(confidence), 1.0))

    reasoning = result.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    reasoning = reasoning.strip()[:500]

    if confidence < _CONFIDENCE_NEEDS_REVIEW_THRESHOLD:
        assignee_id = None
        category = "needs_review"

    return {
        "category": category,
        "assignee_id": assignee_id,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def process_ticket_via_rag(
    subject: str, body: str, db, providers: list[LLMProvider] | None = None
) -> dict:
    """
    Classify an incoming support ticket and assign it to the most relevant support
    staff member using a hybrid BM25 + LLM RAG pipeline.

    :param providers: Optional provider chain to use. Defaults to the providers
        configured via environment variables (``select_llm``). Accepting this as
        a parameter keeps the pipeline testable without real API keys.
    :return: A dict with keys:
        - category (str): Short ticket category label.
        - assignee_id (int | None): DB ID of the matched User, or None.
        - confidence (float): Routing confidence 0.0–1.0.
        - reasoning (str): LLM or BM25 rationale.
    """
    # 1. Load support staff from DB
    users = db.query(models.User).all()

    # 2. Load routing policy documents from knowledge_base/
    policy_docs = _load_knowledge_base_docs(users)

    # 3. PII redaction: scrub sensitive data before any remote LLM call.
    #    Original unredacted text stays in the database.
    redacted_subject = redact_pii(subject)
    redacted_body = redact_pii(body)
    query = f"{redacted_subject}\n{redacted_body}"

    # 4. Build BM25 retrieval context
    user_docs = _build_user_docs(users) if users else []

    if users:
        relevant_policy_docs = _bm25_search(policy_docs, query, k=3) if policy_docs else []
        relevant_user_docs = _bm25_search(user_docs, query, k=3) if user_docs else []

        # If no user doc specifically matched the query, provide all staff docs so
        # the LLM always has the complete support staff directory to choose from.
        target_user_docs = relevant_user_docs if relevant_user_docs else user_docs

        user_context = "\n\n".join(
            f"ID: {d.metadata['id']}, Name: {d.metadata['name']}, Info: {d.page_content}"
            for d in target_user_docs
            if "id" in d.metadata
        )
        policy_context = "\n\n---\n\n".join(
            d.page_content for d in relevant_policy_docs if "source" in d.metadata
        )
    else:
        user_context = ""
        policy_context = ""

    # KB docs resolve ``{{contact_email}}`` to a real staff address; that text
    # goes to the remote provider, which only needs the routing policy. Strip
    # addresses (and only addresses) so policy prose stays intact.
    user_context = redact_emails(user_context)
    policy_context = redact_emails(policy_context)

    providers = providers if providers is not None else select_llm()
    if not providers:
        logger.info("No LLM provider configured. Falling back to BM25 routing.")
        return _fallback_bm25(users, redacted_subject, redacted_body)

    last_error = None
    for llm in providers:
        try:
            context_values = {
                "subject": redacted_subject,
                "body": redacted_body,
                "policy_context": policy_context,
                "user_context": user_context,
            }
            messages = [
                ("system", _SYSTEM_PROMPT.format(**context_values)),
                ("human", _HUMAN_PROMPT),
            ]
            response = invoke_with_retry(llm, messages)

            content = response.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```json"):
                content = content[7:].rstrip("` \n")
            elif content.startswith("```"):
                content = content[3:].rstrip("` \n")

            result = json.loads(content)
            logger.info("%s: Success", provider_label(llm))
            raw_category = str(result.get("category") or "").strip().lower()[:50]
            cleaned = _clean_llm_result(result, users)

            # The local model's softmax confidence is systematically compressed
            # (good matches commonly land in the 0.4–0.7 band), so a clean
            # keyword routing can still be vetoed by the 0.70 needs-review gate.
            # When the model is not confident enough to drive an assignment, let
            # the deterministic BM25 router (staff expertise + KB policies)
            # adjudicate before giving up. It only overrides when BOTH hold:
            #  1. the model still named a routable team category (it found a
            #     real signal, just could not express confidence in it), and
            #  2. BM25's own routing confidence is itself decisive (>= 0.70).
            # A ticket the model called "unclassified" — or that BM25 can only
            # weakly split between two teams — is a genuine coin flip and must
            # go to a human instead of being force-assigned.
            if (
                cleaned["confidence"] < _CONFIDENCE_NEEDS_REVIEW_THRESHOLD
                and raw_category in _KB_ROUTING_CATEGORIES
            ):
                bm25_fallback = _fallback_bm25(users, redacted_subject, redacted_body)
                if (
                    bm25_fallback["assignee_id"] is not None
                    and bm25_fallback["confidence"] >= _CONFIDENCE_NEEDS_REVIEW_THRESHOLD
                ):
                    return bm25_fallback
            return cleaned
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning("%s: Failed - %s", provider_label(llm), e)

    if last_error is not None:
        # All configured LLM providers failed at runtime. Instead of guessing a
        # team via BM25 (which fabricates weak/no-match assignments), leave the
        # ticket unassigned so a human can route it correctly.
        logger.warning(
            "All LLM providers failed. Leaving ticket unassigned for manual review. Last error: %s",
            last_error,
        )

    return {
        "category": "unclassified",
        "assignee_id": None,
        "confidence": 0.0,
        "reasoning": "No LLM provider produced a result",
    }
