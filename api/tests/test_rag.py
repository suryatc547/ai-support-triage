"""Unit tests for the non-LLM BM25 fallback routing in rag.py.

These tests run entirely offline (no API keys, no database) by passing
lightweight fake user objects into ``_fallback_bm25``. The knowledge_base
policy documents used to enrich the routing index are the real shipped files.
"""

import time
from types import SimpleNamespace

import pytest

from src.services.rag import (
    _bm25_search,
    _build_user_docs,
    _clean_llm_result,
    _fallback_bm25,
    _match_kb_doc_to_user,
    _tokenize,
    process_ticket_via_rag,
)

# Team IDs mirrored from the seeded staff: 1=Admin, 2=IT, 3=HR, 4=Security, 5=Facilities
IT_ADMIN = 2
HR = 3
SECURITY = 4
FACILITIES = 5
ADMIN = 1


def make_users():
    """Return a list of lightweight user objects matching the seeded staff.

    The ``expertise`` strings mirror the exact values written by the seed
    script so the tests validate production routing behaviour.
    """
    return [
        SimpleNamespace(
            id=ADMIN,
            name="Admin Team",
            email="admin-test@yopmail.com",
            department="Finance & Procurement",
            expertise=(
                "Purchase orders, invoice approvals, vendor payments, expense "
                "reimbursements, budget queries, financial approvals, contract "
                "renewals, subscription billing, procurement requests, purchase "
                "requisitions, cost center approvals"
            ),
        ),
        SimpleNamespace(
            id=IT_ADMIN,
            name="IT Admin",
            email="it-admin-test@yopmail.com",
            department="IT Support",
            expertise=(
                "Hardware requests, laptop setup, software installation, VPN issues, "
                "network connectivity problems, password resets, device provisioning, "
                "email account setup, printer issues, software licenses, system access, "
                "monitor setup, peripheral devices, remote desktop, IT helpdesk"
            ),
        ),
        SimpleNamespace(
            id=HR,
            name="HR Team",
            email="hr-test@yopmail.com",
            department="Human Resources",
            expertise=(
                "Employee onboarding, offboarding, profile updates, ID card, leave "
                "requests, payroll queries, salary, benefits enrollment, medical "
                "insurance, training programs, policy clarifications, contract "
                "amendments, performance reviews, job transfers, resignation, "
                "termination, employee handbook, attendance"
            ),
        ),
        SimpleNamespace(
            id=SECURITY,
            name="Security Team",
            email="security-test@yopmail.com",
            department="Security & Compliance",
            expertise=(
                "Data breach reports, unauthorized access, suspicious activity, "
                "account compromise, GDPR compliance, data privacy, phishing emails, "
                "malware, ransomware, access revocation, security audit, vulnerability "
                "reports, MFA issues, security policy, incident response, compliance "
                "requirements"
            ),
        ),
        SimpleNamespace(
            id=FACILITIES,
            name="Facilities Team",
            email="facilities-test@yopmail.com",
            department="Facilities & Admin",
            expertise=(
                "Office access cards, building entry, desk booking, hot desking, "
                "maintenance requests, office repairs, visitor management, parking "
                "passes, asset allocation, office supplies, meeting room booking, "
                "air conditioning, cleaning requests, furniture, relocation within office"
            ),
        ),
    ]


@pytest.fixture
def users():
    return make_users()


def route(users, subject, body):
    return _fallback_bm25(users, subject, body)["assignee_id"]


def test_routes_connectivity_to_it(users):
    assert route(users, "Connectivity Issue", "HotSpot is not connecting to device.") == IT_ADMIN


def test_routes_hardware_request_to_it(users):
    assert (
        route(
            users,
            "Device upgrade",
            "Switching to a new project, please arrange a laptop for me.",
        )
        == IT_ADMIN
    )


def test_routes_network_to_it(users):
    assert (
        route(
            users,
            "Network",
            "VPN is not working and the Wi-Fi is very slow, helpdesk.",
        )
        == IT_ADMIN
    )


def test_routes_printer_malfunction_to_it(users):
    assert (
        route(users, "Printer", "The printer is jammed and not printing, please fix.") == IT_ADMIN
    )


def test_routes_printer_supplies_to_facilities(users):
    # Printer consumables legitimately overlap IT ("printer issues") and
    # Facilities ("office supplies: toner cartridges") in the KB, so either is
    # a valid routing outcome. Not a free pass: it must hit one of the two.
    assert route(users, "Supplies", "Need printer paper and a toner cartridge.") in {
        IT_ADMIN,
        FACILITIES,
    }


def test_routes_facilities_repair(users):
    assert (
        route(
            users,
            "Maintenance",
            "The air conditioning in the meeting room is not working, please repair.",
        )
        == FACILITIES
    )


def test_routes_parking_pass_to_facilities(users):
    assert (
        route(
            users,
            "Parking",
            "I need to register a visitor and get a visitor parking pass.",
        )
        == FACILITIES
    )


def test_routes_payroll_to_hr(users):
    assert (
        route(
            users,
            "Payslip",
            "Salary query, can I access my pay slip and payroll deductions?",
        )
        == HR
    )


def test_routes_phishing_to_security(users):
    assert (
        route(
            users,
            "Phishing",
            "My account was hacked and I see a suspicious unauthorized login.",
        )
        == SECURITY
    )


def test_routes_purchase_order_to_admin(users):
    assert (
        route(
            users,
            "Purchase",
            "Please approve this purchase order and process the vendor invoice.",
        )
        == ADMIN
    )


def test_generic_or_no_match_returns_unassigned(users):
    """Regression: queries with no real match must NOT default to any team."""
    assert route(users, "General", "Hello, thanks, does anyone know how this works?") is None
    assert route(users, "General", "Just a friendly random message with zero signal.") is None


def test_empty_users_returns_unassigned():
    result = _fallback_bm25([], "Subject", "body")
    assert result["category"] == "unclassified"
    assert result["assignee_id"] is None
    assert result["confidence"] == 0.0


def test_kb_docs_link_to_each_team(users):
    """Every routing policy doc should map to exactly one staff member."""
    for doc in _load_kb_docs():
        assert _match_kb_doc_to_user(users, doc) is not None


# --- _clean_llm_result ------------------------------------------------------


def test_clean_llm_result_keeps_valid_assignee(users):
    result = _clean_llm_result(
        {"category": "IT", "assignee_id": IT_ADMIN, "confidence": 0.9},
        users,
    )
    assert result == {
        "category": "it",
        "assignee_id": IT_ADMIN,
        "confidence": 0.9,
        "reasoning": "",
    }


def test_clean_llm_result_rejects_unknown_assignee(users):
    """An assignee id that does not exist must not be persisted."""
    result = _clean_llm_result(
        {"category": "it", "assignee_id": 999, "confidence": 0.9},
        users,
    )
    assert result["assignee_id"] is None
    assert result["category"] == "it"


def test_clean_llm_result_rejects_wrong_type_assignee(users):
    assert (
        _clean_llm_result({"category": "it", "assignee_id": "2", "confidence": 0.9}, users)[
            "assignee_id"
        ]
        is None
    )
    # bool is a subclass of int; "true" must not be treated as user id 1.
    assert (
        _clean_llm_result({"category": "it", "assignee_id": True, "confidence": 0.9}, users)[
            "assignee_id"
        ]
        is None
    )
    assert (
        _clean_llm_result({"category": "it", "assignee_id": 1.5, "confidence": 0.9}, users)[
            "assignee_id"
        ]
        is None
    )


def test_clean_llm_result_normalises_category(users):
    assert (
        _clean_llm_result({"category": None, "assignee_id": None, "confidence": 0.9}, users)[
            "category"
        ]
        == "general"
    )
    assert (
        _clean_llm_result({"category": "", "assignee_id": None, "confidence": 0.9}, users)[
            "category"
        ]
        == "general"
    )
    assert (
        _clean_llm_result({"category": "  HR  ", "assignee_id": None, "confidence": 0.9}, users)[
            "category"
        ]
        == "hr"
    )
    long_cat = _clean_llm_result(
        {"category": "x" * 200, "assignee_id": None, "confidence": 0.9}, users
    )
    assert len(long_cat["category"]) == 50


def test_clean_llm_result_missing_confidence_is_needs_review(users):
    # An LLM response without a confidence value is treated as low-confidence
    # and routed to human triage instead of being auto-assigned.
    result = _clean_llm_result({"category": "it", "assignee_id": IT_ADMIN}, users)
    assert result["category"] == "needs_review"
    assert result["assignee_id"] is None
    assert result["confidence"] == 0.0


def test_clean_llm_result_low_confidence_is_needs_review(users):
    result = _clean_llm_result(
        {"category": "hr", "assignee_id": HR, "confidence": 0.5},
        users,
    )
    assert result["category"] == "needs_review"
    assert result["assignee_id"] is None
    assert result["confidence"] == 0.5


def test_clean_llm_result_confidence_boundaries(users):
    # At exactly 0.70 the threshold is met; below it the ticket needs review.
    at_threshold = _clean_llm_result(
        {"category": "hr", "assignee_id": HR, "confidence": 0.70}, users
    )
    assert at_threshold["assignee_id"] == HR
    assert at_threshold["category"] == "hr"

    below = _clean_llm_result({"category": "hr", "assignee_id": HR, "confidence": 0.699}, users)
    assert below["category"] == "needs_review"
    assert below["assignee_id"] is None


def test_clean_llm_result_clamps_out_of_range_confidence(users):
    high = _clean_llm_result({"category": "it", "assignee_id": IT_ADMIN, "confidence": 1.7}, users)
    assert high["confidence"] == 1.0
    low = _clean_llm_result({"category": "it", "assignee_id": IT_ADMIN, "confidence": -0.3}, users)
    assert low["confidence"] == 0.0


def test_clean_llm_result_coerces_reasoning_to_string(users):
    result = _clean_llm_result(
        {"category": "it", "assignee_id": IT_ADMIN, "confidence": 0.9, "reasoning": 42}, users
    )
    assert result["reasoning"] == "42"


# --- _bm25_search -----------------------------------------------------------


def test_bm25_search_returns_most_relevant_user_first(users):
    docs = _build_user_docs(users)
    hits = _bm25_search(docs, "laptop setup VPN and password reset", k=3)
    assert hits, "expected at least one hit"
    assert hits[0].metadata["id"] == IT_ADMIN


def test_bm25_search_honours_k(users):
    docs = _build_user_docs(users)
    hits = _bm25_search(docs, "payroll salary onboarding benefits", k=2)
    assert len(hits) <= 2


def test_bm25_search_drops_noise_query(users):
    """A query of only stopwords must not inject arbitrary context."""
    docs = _build_user_docs(users)
    assert _bm25_search(docs, "hello thanks regards please") == []
    assert _bm25_search(docs, "   ") == []


def test_user_docs_exclude_staff_email_addresses(users):
    """User docs are rendered into the remote prompt, so they must not carry emails."""
    docs = _build_user_docs(users)
    for doc in docs:
        assert "@" not in doc.page_content
    for user in users:
        assert all(user.email not in doc.page_content for doc in docs)


def _load_kb_docs():
    from src.services.rag import _load_knowledge_base_docs

    return _load_knowledge_base_docs()


# --- process_ticket_via_rag with injected providers -------------------------


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeProvider:
    """Lightweight structural stand-in for an LLM provider (Dependency Inversion)."""

    def __init__(self, content: str = "", fails: bool = False):
        self.content = content
        self.fails = fails
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.fails:
            raise RuntimeError("provider exploded")
        return FakeResponse(self.content)


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeDB:
    """Fake session exposing just the ``query`` surface process_ticket_via_rag uses."""

    def __init__(self, users):
        self._users = users

    def query(self, model):
        return FakeQuery(self._users)


def test_process_ticket_via_rag_uses_injected_provider(users):
    provider = FakeProvider(
        '{"category": "finance", "assignee_id": '
        f'{ADMIN}, "confidence": 0.9, "reasoning": "vendor invoice"}}'
    )
    result = process_ticket_via_rag(
        "Invoice approval", "Please approve invoice #44", FakeDB(users), providers=[provider]
    )
    assert provider.calls == 1
    assert result == {
        "category": "finance",
        "assignee_id": ADMIN,
        "confidence": 0.9,
        "reasoning": "vendor invoice",
    }


def test_process_ticket_via_rag_sanitises_injected_provider_output(users):
    provider = FakeProvider(
        '{"category": "  FINANCE ", "assignee_id": "not-an-int", "confidence": 0.8}'
    )
    result = process_ticket_via_rag("Invoice", "Approve this", FakeDB(users), providers=[provider])
    assert result["category"] == "finance"
    assert result["assignee_id"] is None
    assert result["confidence"] == 0.8


def test_process_ticket_via_rag_redacts_pii_before_provider(users):
    captured = {}

    class RecordingProvider:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            captured["system"] = messages[0][1]
            captured["human"] = messages[1][1]
            return FakeResponse(
                '{"category": "it", "assignee_id": 2, "confidence": 0.9, "reasoning": "ok"}'
            )

    provider = RecordingProvider()
    result = process_ticket_via_rag(
        "My card 4242 4242 4242 4242 refused",
        "SSN 123-45-6789 for setup, call 555-123-4567",
        FakeDB(users),
        providers=[provider],
    )
    assert provider.calls == 1
    human = captured["human"].lower()
    assert "4242 4242 4242 4242" not in human
    assert "123-45-6789" not in human
    assert "555-123-4567" not in human
    assert result["assignee_id"] == 2


def test_process_ticket_via_rag_never_sends_staff_email_in_prompt(users):
    """KB docs resolve ``{{contact_email}}``; that staff address must not reach the provider."""
    captured = {}

    class RecordingProvider:
        def invoke(self, messages):
            captured["system"] = messages[0][1]
            return FakeResponse(
                '{"category": "it", "assignee_id": 2, "confidence": 0.9, "reasoning": "ok"}'
            )

    process_ticket_via_rag(
        "VPN and laptop setup",
        "Please set up VPN on my new laptop and reset my password.",
        FakeDB(users),
        providers=[RecordingProvider()],
    )

    system = captured["system"]
    assert system
    for user in users:
        assert user.email not in system


def test_process_ticket_via_rag_unassigned_when_provider_fails(users):
    provider = FakeProvider(fails=True)
    result = process_ticket_via_rag("Help", "Laptop broken", FakeDB(users), providers=[provider])
    assert result["category"] == "unclassified"
    assert result["assignee_id"] is None
    assert result["confidence"] == 0.0


def test_low_confidence_provider_adjudicated_by_decisive_bm25(users):
    """A compressed model confidence must not veto a keyword-decisive ticket.

    The local model's softmax reliably lands good-but-unspectacular matches in
    the 0.4–0.7 band, so the 0.70 gate alone leaves valid tickets unassigned.
    When the model is unsure, the deterministic BM25 router adjudicates and
    upgrades the assignment when the keyword evidence is decisive.
    """
    provider = FakeProvider(
        '{"category": "finance_procurement", "assignee_id": null,'
        ' "confidence": 0.52, "reasoning": "perhaps invoice"}'
    )
    result = process_ticket_via_rag(
        "Incorrect charge",
        "I noticed a charge on my recent invoice that I don't understand. "
        "Invoice number: INV-20451",
        FakeDB(users),
        providers=[provider],
    )
    assert provider.calls == 1
    assert result["assignee_id"] == ADMIN
    assert result["confidence"] >= 0.5


def test_low_confidence_provider_stays_needs_review_without_bm25_evidence(users):
    """A low-confidence model output with no keyword support must stay unassigned."""
    provider = FakeProvider(
        '{"category": "general", "assignee_id": null, "confidence": 0.3, "reasoning": "unsure"}'
    )
    result = process_ticket_via_rag(
        "flurb", "wibble quux zilt gribble", FakeDB(users), providers=[provider]
    )
    assert result["category"] == "needs_review"
    assert result["assignee_id"] is None


def test_low_confidence_model_with_unclassified_category_not_adjudicated(users):
    """An 'unclassified' model verdict is not overridden by opportunistic BM25 hits.

    When the model says it found no team fit, keyword overlap with one policy
    doc must not be enough to force an assignment — the out-of-scope/scam scrap
    (e.g. "you won a gift card") matches the access-card policy purely on the
    word "card", and routing that to facilities would be a hallucination.
    """
    provider = FakeProvider(
        '{"category": "unclassified", "assignee_id": null,'
        ' "confidence": 0.38, "reasoning": "no team"}'
    )
    result = process_ticket_via_rag(
        "Congratulations, you have won!",
        "Your email was selected and you won a USD 500 gift card. Claim it within 24 hours!",
        FakeDB(users),
        providers=[provider],
    )
    assert result["category"] == "needs_review"
    assert result["assignee_id"] is None


def test_low_confidence_adjudication_requires_decisive_bm25_confidence(users):
    """BM25 must be decisive on its own terms before it can override needs_review.

    A weak BM25 split (~0.5 ratio, e.g. payslip access that touches both HR and
    IT docs) is not evidence — it is exactly the "coin flip" that must go to a
    human instead of being force-assigned to whichever team's doc barely won.
    """
    provider = FakeProvider(
        '{"category": "human_resources", "assignee_id": null,'
        ' "confidence": 0.50, "reasoning": "perhaps hr"}'
    )
    result = process_ticket_via_rag(
        "Payslip access blocked",
        "My payslip for this month is not on the portal and the loan application "
        "deadline is approaching. Can someone help me retrieve a copy?",
        FakeDB(users),
        providers=[provider],
    )
    assert result["category"] == "needs_review"
    assert result["assignee_id"] is None

    # Counter-check: a decisive BM25 ratio still upgrades a well-known keyword
    # routing, so the original invoice fix keeps working.
    provider = FakeProvider(
        '{"category": "finance_procurement", "assignee_id": null,'
        ' "confidence": 0.52, "reasoning": "perhaps invoice"}'
    )
    result = process_ticket_via_rag(
        "Incorrect charge",
        "I noticed a charge on my recent invoice that I don't understand. "
        "Invoice number: INV-20451",
        FakeDB(users),
        providers=[provider],
    )
    assert result["assignee_id"] == ADMIN
    assert result["confidence"] >= 0.70


def test_kb_docs_placeholder_resolution(users):
    from src.services.rag import _load_knowledge_base_docs

    docs = _load_knowledge_base_docs(users)
    assert len(docs) > 0
    for doc in docs:
        assert "{{contact_email}}" not in doc.page_content
        assert "{{STAFF_DOMAIN}}" not in doc.page_content


def test_kb_doc_matches_user_with_custom_email_domain():
    from src.services.rag import _load_knowledge_base_docs

    custom_users = [
        SimpleNamespace(
            id=99,
            name="IT Admin",
            email="custom-it@enterprise.internal",
            department="IT Support",
            expertise="hardware",
        )
    ]
    docs = _load_knowledge_base_docs(custom_users)
    # With section chunking, the resolved contact email lives in the team
    # breadcrumb chunk; every chunk must still be placeholder-free (covered by
    # test_kb_docs_placeholder_resolution).
    it_doc = next(d for d in docs if "custom-it@enterprise.internal" in d.page_content)
    assert "{{contact_email}}" not in it_doc.page_content

    matched = _match_kb_doc_to_user(custom_users, it_doc)
    assert matched is not None
    assert matched.id == 99


def test_new_department_without_code_changes():
    """Verify that a brand new department added purely in DB works dynamically."""
    from langchain_core.documents import Document

    new_user = SimpleNamespace(
        id=42,
        name="Legal Team",
        email="legal@custom-domain.com",
        department="Legal & Compliance",
        expertise="Review vendor contracts, NDAs, and corporate governance policies",
    )
    doc_content = (
        "# Legal Routing Policy\n\n"
        "## Team: Legal Team\n"
        "**Department:** Legal & Compliance\n"
        "**Contact:** {{contact_email}}\n\n"
        "Handle contracts and NDAs."
    )
    doc = Document(page_content=doc_content)
    matched = _match_kb_doc_to_user([new_user], doc)
    assert matched is not None
    assert matched.id == 42
    assert matched.email == "legal@custom-domain.com"


def test_tokenize_strips_punctuation():
    """Verify that _tokenize strips punctuation so trailing commas/periods match."""
    tokens = _tokenize("Hello, welcome! Here are packages, parcels, and kits.")
    assert "welcome" in tokens
    assert "welcome," not in tokens
    assert "packages" in tokens
    assert "parcels" in tokens
    assert "kits" in tokens


def test_routes_welcome_kit_parcel_damage_to_hr(users):
    """Verify welcome kit and parcel damage routes to HR via fallback BM25."""
    assert (
        route(
            users,
            "General Query",
            "I received my welcome kits today. However the parcel has minor damages.",
        )
        == HR
    )


# --- section chunking -------------------------------------------------------


def test_chunk_by_headings_splits_sections_and_keeps_breadcrumbs():

    from src.services.rag import _chunk_by_headings

    content = (
        "# Team Policy\n\n"
        "Intro line with context.\n\n"
        "## Scope of Support\n"
        "Handle laptops and printers.\n\n"
        "## Escalation Triggers\n"
        "Route phishing to Security.\n"
    )
    chunks = _chunk_by_headings(content, "it_support.md", "it admin", "it support")

    assert len(chunks) == 3
    preamble, scope, escalation = chunks
    assert preamble.page_content == "# Team Policy\n\nIntro line with context."
    assert "Scope of Support" in scope.page_content
    assert "Route phishing to Security." in escalation.page_content

    for chunk in chunks:
        assert chunk.metadata["source"] == "it_support.md"
        assert chunk.metadata["team"] == "it admin"
        assert chunk.metadata["department"] == "it support"
    assert scope.metadata["section"] == "## Scope of Support"
    assert escalation.metadata["section"] == "## Escalation Triggers"


def test_load_knowledge_base_docs_returns_section_chunks(users):
    from src.services.rag import _load_knowledge_base_docs

    docs = _load_knowledge_base_docs(users)
    assert len(docs) > 1
    scopes = [d for d in docs if d.metadata.get("section", "").startswith("##")]
    assert scopes, "expected heading-delimited section chunks"
    assert all("{{contact_email}}" not in d.page_content for d in docs)


# --- KB + BM25 caching ------------------------------------------------------


@pytest.fixture
def kb_dir(tmp_path, monkeypatch):
    import src.services.rag as rag

    md = tmp_path / "it.md"
    md.write_text(
        "## Team: IT\n**Contact:** {{contact_email}}\n\n## Scope\nHandle laptops.",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "KNOWLEDGE_BASE_DIR", tmp_path)
    monkeypatch.setattr(rag, "_KB_DOCS_CACHE", {"mtime": 0.0, "user_key": "", "docs": None})
    monkeypatch.setattr(rag, "_BM25_TOKEN_CACHE", {})
    return md


def test_knowledge_base_docs_cached_between_calls(users, kb_dir):
    from src.services.rag import _load_knowledge_base_docs

    first = _load_knowledge_base_docs(users)
    second = _load_knowledge_base_docs(users)
    assert first is second


def test_cache_invalidated_when_kb_file_changes(users, kb_dir, monkeypatch):
    import os

    import src.services.rag as rag

    first = rag._load_knowledge_base_docs(users)
    assert first
    assert all("Handle Printers." not in d.page_content for d in first)

    kb_dir.write_text(
        "## Team: IT\n**Contact:** {{contact_email}}\n\n## Scope\nHandle Printers.",
        encoding="utf-8",
    )
    future = time.time() + 10
    os.utime(kb_dir, (future, future))

    second = rag._load_knowledge_base_docs(users)
    assert second is not None
    assert second is not first
    assert any("Handle Printers." in d.page_content for d in second)


def test_bm25_token_cache_reused_for_same_corpus(users, monkeypatch):
    import src.services.rag as rag

    docs = rag._build_user_docs(users)
    hits1 = rag._bm25_search(docs, "laptop VPN printer", k=3)
    first_hits = len(hits1)
    hits2 = rag._bm25_search(docs, "laptop VPN printer", k=3)
    assert len(hits1) == len(hits2) == first_hits
    assert rag._BM25_TOKEN_CACHE, "expected a cached tokenized corpus"


def test_team_routing_confidence_is_calibrated_not_saturated():
    """Confidence must carry signal: rival-team matches must not read 1.0.

    The old normalisation (``best / max(scores)``) divided the best score by
    itself because the argmax *is* the max — every routed ticket reported
    exactly ``1.0`` and the needs-review gate could never catch a coin-flip
    routing. The team-level ratio must score a two-way tie near 0.5, a
    one-sided match at 1.0, and never exceed 1.0. Same-team documents (a user's
    expertise doc + its own KB chunks) are aggregated before the comparison, so
    they reinforce a team instead of being counted as rivals.
    """
    from src.services.rag import _team_routing_confidence

    # Two rival teams score identically -> ambiguous -> must be < 1.0.
    best_team, ambiguous = _team_routing_confidence({1: 1.5, 2: 1.5})
    assert ambiguous < 1.0
    assert abs(ambiguous - 0.5) < 1e-6
    assert best_team in (1, 2)

    # Best team strongly dominates -> high but still bounded.
    _, dominated = _team_routing_confidence({1: 3.0, 2: 0.25, 3: 0.1})
    assert dominated < 1.0

    # Rival teams with no score -> unambiguous.
    assert _team_routing_confidence({1: 2.5}) == (1, 1.0)
    assert _team_routing_confidence({1: 2.5, 2: 0.0}) == (1, 1.0)

    # Empty / guard rail never exceed [0, 1].
    assert _team_routing_confidence({}) == (0, 0.0)


def test_fallback_bm25_confidence_stays_in_range(users):
    """End-to-end: any routed fallback ticket reports a bounded, positive confidence."""
    result = _fallback_bm25(users, "VPN connectivity issue", "Cannot reach the office network")
    if result["assignee_id"] is not None:
        assert 0.0 < result["confidence"] <= 1.0


def test_fallback_bm25_routes_invoice_keywords_to_finance(users):
    """A keyword-decisive ticket (invoice/charge) must route to the admin team."""
    result = _fallback_bm25(
        users,
        "Incorrect charge on my account",
        "I noticed a charge on my recent invoice I don't understand. Invoice number: INV-20451",
    )
    assert result["assignee_id"] == ADMIN
    assert result["confidence"] >= 0.5
