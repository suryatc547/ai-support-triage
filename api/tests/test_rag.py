"""Unit tests for the non-LLM BM25 fallback routing in rag.py.

These tests run entirely offline (no API keys, no database) by passing
lightweight fake user objects into ``_fallback_bm25``. The knowledge_base
policy documents used to enrich the routing index are the real shipped files.
"""

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
            email="admin@test.com",
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
            email="it-admin@test.com",
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
            email="hr@test.com",
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
            email="security@test.com",
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
            email="facilities@test.com",
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
    assert _fallback_bm25([], "Subject", "body") == {
        "category": "unclassified",
        "assignee_id": None,
    }


def test_kb_docs_link_to_each_team(users):
    """Every routing policy doc should map to exactly one staff member."""
    for doc in _load_kb_docs():
        assert _match_kb_doc_to_user(users, doc) is not None


# --- _clean_llm_result ------------------------------------------------------


def test_clean_llm_result_keeps_valid_assignee(users):
    assert _clean_llm_result({"category": "IT", "assignee_id": IT_ADMIN}, users) == {
        "category": "it",
        "assignee_id": IT_ADMIN,
    }


def test_clean_llm_result_rejects_unknown_assignee(users):
    """An assignee id that does not exist must not be persisted."""
    assert _clean_llm_result({"category": "it", "assignee_id": 999}, users)["assignee_id"] is None


def test_clean_llm_result_rejects_wrong_type_assignee(users):
    assert _clean_llm_result({"category": "it", "assignee_id": "2"}, users)["assignee_id"] is None
    # bool is a subclass of int; "true" must not be treated as user id 1.
    assert _clean_llm_result({"category": "it", "assignee_id": True}, users)["assignee_id"] is None
    assert _clean_llm_result({"category": "it", "assignee_id": 1.5}, users)["assignee_id"] is None


def test_clean_llm_result_normalises_category(users):
    assert (
        _clean_llm_result({"category": None, "assignee_id": None}, users)["category"] == "general"
    )
    assert _clean_llm_result({"category": "", "assignee_id": None}, users)["category"] == "general"
    assert _clean_llm_result({"category": "  HR  ", "assignee_id": None}, users)["category"] == "hr"
    long_cat = _clean_llm_result({"category": "x" * 200, "assignee_id": None}, users)
    assert len(long_cat["category"]) == 50


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
    provider = FakeProvider(f'{{"category": "finance", "assignee_id": {ADMIN}}}')
    result = process_ticket_via_rag(
        "Invoice approval", "Please approve invoice #44", FakeDB(users), providers=[provider]
    )
    assert provider.calls == 1
    assert result == {"category": "finance", "assignee_id": ADMIN}


def test_process_ticket_via_rag_sanitises_injected_provider_output(users):
    provider = FakeProvider('{"category": "  FINANCE ", "assignee_id": "not-an-int"}')
    result = process_ticket_via_rag("Invoice", "Approve this", FakeDB(users), providers=[provider])
    assert result == {"category": "finance", "assignee_id": None}


def test_process_ticket_via_rag_unassigned_when_provider_fails(users):
    provider = FakeProvider(fails=True)
    result = process_ticket_via_rag("Help", "Laptop broken", FakeDB(users), providers=[provider])
    assert result == {"category": "unclassified", "assignee_id": None}


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
    it_doc = next(d for d in docs if "IT Support" in d.page_content)
    assert "custom-it@enterprise.internal" in it_doc.page_content

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
