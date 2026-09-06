import json
import logging
import os
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from ..models import models

logger = logging.getLogger(__name__)

# Path to policy documents — located inside src/knowledge_base/
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

# Retry settings for transient 429 (rate limit) throttling on the primary provider.
_PRIMARY_RETRIES = 3
_PRIMARY_BACKOFF_SECONDS = 2.0

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
    """Lowercase and split a document/query, dropping routing-noise stopwords."""
    return [tok for tok in text.lower().split() if tok not in _STOPWORDS]


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

Respond ONLY with a valid JSON object with exactly two keys:
- "category": a short lowercase string (e.g., "finance", "it_support", "hr",
  "security", "facilities")
- "assignee_id": the integer ID of the best matched staff member, or null if no match
"""

_HUMAN_PROMPT = "Route the following support ticket. Return ONLY the JSON object."


def _get_primary_llm():
    """Instantiate the primary Gemini LLM if an API key is configured."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"), temperature=0
    )


def _get_fallback_llms():
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
            )
        )

    if os.environ.get("GROQ_API_KEY"):
        fallbacks.append(
            ChatGroq(
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.environ["GROQ_API_KEY"],
                temperature=0,
            )
        )

    return fallbacks


def invoke_with_retry(llm, prompt_values):
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
                time.sleep(_PRIMARY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise


def provider_label(llm) -> str:
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


def select_llm():
    """Return an ordered list of configured LLM providers (primary + fallbacks)."""
    providers = []
    primary = _get_primary_llm()
    if primary is not None:
        providers.append(primary)
    providers.extend(_get_fallback_llms())
    return providers


def _fallback_bm25(users, subject, body, threshold=1.0):
    """Assign using the best BM25 match when no LLM provider is configured.

    Uses both the support staff expertise and the knowledge_base routing policy
    documents (linked to each team) to build the search index. Applies a
    confidence threshold so weak or non-matching queries are left unassigned
    rather than guessed into an arbitrary team.
    """
    if not users:
        return {"category": "unclassified", "assignee_id": None}

    # Build an index of (text, assignee_id) entries from staff expertise plus
    # the knowledge_base policy docs (matched to a team via their contact email).
    indexed: list[tuple[str, int]] = []
    for d in _build_user_docs(users):
        indexed.append((d.page_content, d.metadata["id"]))
    for pd in _load_knowledge_base_docs():
        user = _match_kb_doc_to_user(users, pd)
        if user is not None:
            indexed.append((pd.page_content, user.id))

    tokenized = [_tokenize(text) for (text, _id) in indexed]
    if not tokenized:
        return {"category": "unclassified", "assignee_id": None}

    bm25 = BM25Okapi(tokenized)
    query = _tokenize(f"{subject}\n{body}")
    scores = bm25.get_scores(query)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_idx] < threshold:
        return {"category": "unclassified", "assignee_id": None}

    return {"category": "general", "assignee_id": indexed[best_idx][1]}


def _match_kb_doc_to_user(users, doc) -> models.User | None:
    """Link a knowledge_base routing policy doc to the matching support User.

    Policy docs declare a `Contact:` line such as `contact: it-admin@test.com`;
    we match it against the User's email address.
    """
    import re

    match = re.search(r"contact:\s*\**\s*([\w.+-]+@[\w.-]+)", doc.page_content, re.IGNORECASE)
    if not match:
        return None
    contact_email = match.group(1).strip().lower()
    for u in users:
        if (u.email or "").strip().lower() == contact_email:
            return u
    return None


def _load_knowledge_base_docs() -> list[Document]:
    """Load all markdown routing policy documents from the knowledge_base directory."""
    docs = []
    if not KNOWLEDGE_BASE_DIR.exists():
        return docs
    for md_file in KNOWLEDGE_BASE_DIR.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={"source": md_file.name}))
    return docs


def _build_user_docs(users: list) -> list[Document]:
    """Build BM25-indexable documents from User records using their expertise field."""
    docs = []
    for u in users:
        expertise_text = u.expertise or f"Support for {u.department} inquiries."
        content = (
            f"Name: {u.name}\n"
            f"Email: {u.email}\n"
            f"Department: {u.department}\n"
            f"Expertise: {expertise_text}"
        )
        docs.append(Document(page_content=content, metadata={"id": u.id, "name": u.name}))
    return docs


def _bm25_search(docs: list[Document], query: str, k: int = 5) -> list[Document]:
    """Return the top ``k`` documents most relevant to ``query`` via BM25.

    Local replacement for the sunset ``langchain_community`` BM25Retriever.
    ``rank_bm25`` is already a direct project dependency and the tokeniser is
    shared with :func:`_fallback_bm25`, so stopword noise is filtered the same
    way in both paths. Documents with zero similarity are dropped so clearly
    unrelated mail does not inject noise into the LLM prompt context.
    """
    tokenized = [_tokenize(doc.page_content) for doc in docs]
    query_tokens = _tokenize(query)

    if not tokenized or not query_tokens:
        return []

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked[:k] if scores[i] > 0]


def _clean_llm_result(result: dict, users: list) -> dict:
    """Sanitise the LLM's routing response before it touches the database.

    The LLM is a remote, non-deterministic component — its JSON can be off-spec
    (missing keys, wrong types, or an assignee id that does not exist). This
    helper guards the DB against such output:

    - ``category`` is coerced to a short lowercase string.
    - ``assignee_id`` is kept only when it is an integer that refers to a real
      support staff member; anything else becomes ``None`` (unassigned).
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

    return {"category": category, "assignee_id": assignee_id}


def process_ticket_via_rag(subject: str, body: str, db) -> dict:
    """
    Classify an incoming support ticket and assign it to the most relevant support
    staff member using a hybrid BM25 + LLM RAG pipeline.

    Returns a dict with keys:
        - category (str): Short ticket category label.
        - assignee_id (int | None): DB ID of the matched User, or None.
    """
    # 1. Load support staff from DB
    users = db.query(models.User).all()

    # 2. Load routing policy documents from knowledge_base/
    policy_docs = _load_knowledge_base_docs()

    # 3. Build the combined BM25 retrieval context
    user_docs = _build_user_docs(users) if users else []
    all_docs = user_docs + policy_docs

    query = f"{subject}\n{body}"

    if all_docs and users:
        relevant_docs = _bm25_search(all_docs, query, k=5)
        user_context = "\n\n".join(
            f"ID: {d.metadata['id']}, Name: {d.metadata['name']}, Info: {d.page_content}"
            for d in relevant_docs
            if "id" in d.metadata
        )
        policy_context = "\n\n---\n\n".join(
            d.page_content for d in relevant_docs if "source" in d.metadata
        )
    else:
        user_context = ""
        policy_context = ""

    providers = select_llm()
    if not providers:
        logger.info("No LLM provider configured. Falling back to BM25 routing.")
        return _fallback_bm25(users, subject, body)

    last_error = None
    for llm in providers:
        try:
            context_values = {
                "subject": subject,
                "body": body,
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
            return _clean_llm_result(result, users)
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

    return {"category": "unclassified", "assignee_id": None}
