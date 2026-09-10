"""Unit tests for the custom domain model in modelm."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from modelm.langchain_model import LocalSupportChatModel
from modelm.model import CustomTicketClassifier


def test_custom_classifier_fit_and_predict():
    texts = [
        "VPN connection keeps dropping and cannot connect to Wi-Fi",
        "Laptop screen is flickering and mouse is not working",
        "Need approval for vendor invoice PO-1029",
        "Expense reimbursement claim for client lunch travel",
        "Questions regarding annual medical health insurance policy",
        "Salary payslip discrepancy for this month",
        "Suspicious phishing email asking for login credentials",
        "Report malware alert on database server",
        "Office access badge RFID keycard is not opening door",
        "Air conditioning AC in boardroom is not cooling properly",
    ]
    labels = [
        "it_support",
        "it_support",
        "finance_procurement",
        "finance_procurement",
        "human_resources",
        "human_resources",
        "security_compliance",
        "security_compliance",
        "facilities_admin",
        "facilities_admin",
    ]

    classifier = CustomTicketClassifier()
    classifier.fit(texts, labels, epochs=300, lr=2.0)

    cat, prob = classifier.predict("My VPN client is disconnected from Wi-Fi")
    assert cat == "it_support"
    assert prob > 0.30

    cat_fin, prob_fin = classifier.predict("Please approve my vendor invoice")
    assert cat_fin == "finance_procurement"
    assert prob_fin > 0.30


def test_local_support_chat_model_invoke():
    model = LocalSupportChatModel()

    system_prompt = (
        "--- AVAILABLE SUPPORT STAFF ---\n"
        "ID: 2, Name: IT Admin, Info: Hardware, VPN, WiFi, Laptop\n"
        "ID: 1, Name: Admin Team, Info: Purchase orders, invoices\n"
    )
    user_prompt = "Subject: VPN issue\n\nBody: I cannot connect to the company VPN."

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    result = model.invoke(messages)
    assert hasattr(result, "content")
    payload = json.loads(result.content)
    assert payload["category"] == "it_support"
    assert payload["assignee_id"] == 2


def test_local_support_chat_model_unclassified_on_nonsense():
    model = LocalSupportChatModel(confidence_threshold=0.80)
    messages = [HumanMessage(content="xyz abcd 1234 nonesuch")]
    result = model.invoke(messages)
    payload = json.loads(result.content)
    assert payload["category"] == "unclassified"
    assert payload["assignee_id"] is None


def test_fallback_chain_handles_429_with_local_model():
    """Simulate 429 on cloud provider and verify local model succeeds."""
    from api.src.services.rag import invoke_with_retry

    # Mock cloud provider that throws 429
    failing_cloud_llm = MagicMock()
    err_429 = RuntimeError("Rate limit exceeded")
    err_429.status_code = 429
    failing_cloud_llm.invoke.side_effect = err_429

    local_model = LocalSupportChatModel()

    prompt = [
        (
            "system",
            "ID: 2, Name: IT Admin, Info: VPN\n\nSubject: VPN\nBody: VPN broken",
        ),
        ("human", "Route ticket"),
    ]

    # Cloud LLM fails with 429
    with pytest.raises(RuntimeError):
        invoke_with_retry(failing_cloud_llm, prompt)

    # Chain advances to local model which succeeds seamlessly
    res = invoke_with_retry(local_model, prompt)
    payload = json.loads(res.content)
    assert payload["category"] == "it_support"
    assert payload["assignee_id"] == 2


def test_extract_ticket_text_does_not_bleed_rag_context():
    """Verify _extract_ticket_text does not leak subsequent sections into body."""
    local_model = LocalSupportChatModel()
    prompt = (
        "--- INCOMING TICKET ---\n"
        "Subject: Welcome kit\n"
        "Body: My welcome kit parcel has minor damages.\n\n"
        "--- ROUTING POLICY CONTEXT ---\n"
        "# Security & Compliance — Routing Policy\n"
        "breach, hack, phishing, malware, ransomware\n\n"
        "--- AVAILABLE SUPPORT STAFF ---\n"
        "ID: 3, Name: HR Team, Info: Onboarding"
    )
    subject, body, _ = local_model._extract_ticket_text([("system", prompt)])
    assert subject == "Welcome kit"
    assert body == "My welcome kit parcel has minor damages."
    assert "phishing" not in body
    assert "ROUTING POLICY" not in body


def test_local_model_routes_welcome_kit_damage_to_hr():
    """Verify welcome kit parcel damage routes to human_resources and HR assignee."""
    local_model = LocalSupportChatModel()
    prompt = [
        (
            "system",
            (
                "--- INCOMING TICKET ---\n"
                "Subject: General Query\n"
                "Body: I received my welcome kits today. However the parcel has minor damages.\n\n"
                "--- AVAILABLE SUPPORT STAFF ---\n"
                "ID: 3, Name: HR Team, Info: Onboarding, welcome kits\n"
            ),
        ),
        ("human", "Route ticket"),
    ]
    res = local_model.invoke(prompt)
    payload = json.loads(res.content)
    assert payload["category"] == "human_resources"
    assert payload["assignee_id"] == 3


def test_adamw_optimizer_convergence():
    """Verify AdamW optimizer converges rapidly and tracks validation history."""
    texts = [
        "Cannot connect to office Wi-Fi",
        "Laptop charger is broken",
        "Please approve vendor invoice PO-102",
        "Expense report for travel pending",
        "When is the annual appraisal cycle?",
        "Updating home address in employee portal",
        "Suspicious phishing link in email",
        "Account access revoked for contractor",
        "Office air conditioning broken in room 3",
        "Desk chair is unstable and broken",
    ]
    labels = [
        "it_support",
        "it_support",
        "finance_procurement",
        "finance_procurement",
        "human_resources",
        "human_resources",
        "security_compliance",
        "security_compliance",
        "facilities_admin",
        "facilities_admin",
    ]
    clf = CustomTicketClassifier()
    history = clf.fit(
        texts,
        labels,
        val_texts=texts,
        val_labels=labels,
        epochs=120,
        lr=0.08,
        optimizer="adamw",
    )
    assert "train_loss" in history
    assert history["train_loss"][0] > history["train_loss"][-1]
    assert history["val_acc"][-1] == 1.0


def test_stratified_split():
    """Verify stratified_split preserves class representation and set separation."""
    from modelm.train import stratified_split

    texts = [f"Sample text {i}" for i in range(50)]
    labels = ["cat_a"] * 20 + ["cat_b"] * 20 + ["cat_c"] * 10

    train_t, train_l, val_t, val_l = stratified_split(texts, labels, test_size=0.2, seed=42)

    assert len(train_t) + len(val_t) == 50
    assert len(train_l) == len(train_t)
    assert len(val_l) == len(val_t)

    # Check all classes present in validation set
    assert set(val_l) == {"cat_a", "cat_b", "cat_c"}


def test_compute_classification_metrics():
    """Verify precision, recall, f1, and accuracy computations."""
    from modelm.train import compute_classification_metrics

    cats = ["it", "hr", "finance"]
    y_true = ["it", "it", "hr", "hr", "finance"]
    y_pred = ["it", "hr", "hr", "hr", "finance"]

    metrics = compute_classification_metrics(y_true, y_pred, cats)
    assert metrics["accuracy"] == 0.8
    assert "it" in metrics["per_class"]
    assert "confusion_matrix" in metrics
    assert metrics["macro_f1"] > 0.70


def test_dataset_deduplication():
    """Verify deduplicate_dataset eliminates duplicates and near-duplicates."""
    from modelm.dataset import deduplicate_dataset

    pairs = [
        ("VPN connection is dropping constantly", "it_support"),
        ("vpn connection is dropping constantly!", "it_support"),  # exact duplicate (punctuation)
        ("VPN connection is dropping constantly.", "it_support"),  # exact duplicate (casing)
        ("Need invoice approval for contractor", "finance_procurement"),
    ]
    clean = deduplicate_dataset(pairs, threshold=0.85)
    assert len(clean) == 2
    assert clean[0][1] == "it_support"
    assert clean[1][1] == "finance_procurement"


def test_assignee_matching_is_order_independent():
    """Verify routing does not depend on staff list order (no substring-alias bugs)."""
    model = LocalSupportChatModel()
    facility_first = [
        (
            "system",
            (
                "--- INCOMING TICKET ---\n"
                "Subject: VPN broken\n"
                "Body: Cannot connect to corporate VPN from home.\n\n"
                "--- AVAILABLE SUPPORT STAFF ---\n"
                "ID: 5, Name: Facilities Team, Info: office access cards, desks\n"
                "ID: 2, Name: IT Admin, Info: Hardware, VPN, WiFi\n"
            ),
        ),
        ("human", "Route ticket"),
    ]
    it_first = [
        (
            "system",
            (
                "ID: 2, Name: IT Admin, Info: Hardware, VPN, WiFi\n"
                "ID: 5, Name: Facilities Team, Info: office access cards, desks"
            ),
        ),
        ("human", "subject: vpn broken body: cannot connect to corporate vpn"),
    ]

    payload_first = json.loads(model.invoke(facility_first).content)
    payload_second = json.loads(model.invoke(it_first).content)

    assert payload_first["category"] == "it_support"
    assert payload_second["category"] == "it_support"
    # Same ticket must map to the same team regardless of listing order.
    assert payload_first["assignee_id"] == payload_second["assignee_id"] == 2


def test_temperature_scaling():
    """Verify temperature parameter softens logits when high and sharpens when low."""
    clf = CustomTicketClassifier(categories=["it_support", "hr"])
    clf.vocab = {"vpn": 0, "leave": 1}
    clf.idf = [1.0, 1.0]
    # Simple manual weights
    clf.weights = np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=np.float32)
    clf.bias = np.zeros(2, dtype=np.float32)

    sharp_probs = clf.predict_proba("vpn", temperature=0.2)
    soft_probs = clf.predict_proba("vpn", temperature=5.0)

    # Low temperature should be sharper (closer to 1.0 for top class)
    assert sharp_probs["it_support"] > soft_probs["it_support"]
    # High temperature should be softer (closer to uniform 0.5)
    assert soft_probs["it_support"] < 0.80
