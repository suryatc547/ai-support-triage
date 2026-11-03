"""Tests for PII redaction service."""

from src.services.pii_service import redact_emails, redact_pii


def test_redact_empty_and_none():
    assert redact_pii("") == ""
    assert redact_pii(None) == ""


def test_redact_credit_cards():
    text = "Please refund to my card 4242 4242 4242 4242 or 5500000000000004 immediately."
    redacted = redact_pii(text)
    assert "4242 4242 4242 4242" not in redacted
    assert "5500000000000004" not in redacted
    assert "[REDACTED_CREDIT_CARD]" in redacted


def test_redact_credit_card_rejects_invalid_luhn():
    # 4111 2222 3333 4444 and 1234-5678-9012-3456 are credit-card-shaped but
    # fail the Luhn checksum — they must be preserved, not redacted.
    text = "Order reference 4111 2222 3333 4444 details: 1234-5678-9012-3456 thanks."
    redacted = redact_pii(text)
    assert "4111 2222 3333 4444" in redacted
    assert "1234-5678-9012-3456" in redacted
    assert "[REDACTED_CREDIT_CARD]" not in redacted


def test_preserves_credit_card_lookalike_reference():
    # A long numeric reference that fails Luhn must not be partially consumed
    # by the phone regex either.
    text = "Reference #1234-5678-9012-3456 is your order."
    redacted = redact_pii(text)
    assert "1234-5678-9012-3456" in redacted
    assert "[REDACTED_CREDIT_CARD]" not in redacted
    assert "[REDACTED_PHONE]" not in redacted


def test_redact_ssn():
    text = "My employee SSN is 123-45-6789 for tax purposes."
    redacted = redact_pii(text)
    assert "123-45-6789" not in redacted
    assert "[REDACTED_SSN]" in redacted


def test_redact_email_addresses():
    text = "Please reply to jane.doe+support@example.co.uk or bob@corp.example.com about ticket 42."
    redacted = redact_pii(text)
    assert "jane.doe+support@example.co.uk" not in redacted
    assert "bob@corp.example.com" not in redacted
    assert redacted.count("[REDACTED_EMAIL]") == 2


def test_preserves_non_email_at_mentions():
    # A bare @handle or user@host token without a dotted TLD is not an address.
    text = "Ping @support or open user@localhost for details."
    assert redact_pii(text) == text


def test_redact_emails_only_touches_addresses():
    text = "Contact jane@example.com; rotate the API key: abc123 quarterly."
    assert redact_emails(text) == (
        "Contact [REDACTED_EMAIL]; rotate the API key: abc123 quarterly."
    )


def test_redact_passwords_and_tokens():
    text = (
        "Here is the credential: password: mySecretP@ssword! "
        "and api_key: sk_live_987654321 and token=xyz123456789."
    )
    redacted = redact_pii(text)
    assert "mySecretP@ssword!" not in redacted
    assert "sk_live_987654321" not in redacted
    assert "xyz123456789" not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_redact_multi_word_secret_value():
    # Value with interior spacing used to leak everything after the first token.
    text = "password = sup3r s3cret value"
    redacted = redact_pii(text)
    assert "sup3r s3cret value" not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_preserves_innocent_keyword_sentence():
    # Keywords like "password" without a `:`/`=` must not trigger redaction.
    text = "Please visit the password reset tool or check the API documentation."
    assert redact_pii(text) == text


def test_redact_bearer_token():
    text = "Authorization header: Bearer abcdef1234567890_long_token_string_here"
    redacted = redact_pii(text)
    assert "abcdef1234567890_long_token_string_here" not in redacted
    assert "Bearer [REDACTED_TOKEN]" in redacted


def test_redact_phone_numbers():
    text = "Call me back at +1-555-123-4567 or (800) 555-0199."
    redacted = redact_pii(text)
    assert "555-123-4567" not in redacted
    assert "555-0199" not in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_preserves_long_numeric_sequence():
    # An 11+ digit account number must not be partially redacted as a phone.
    text = "Account: 55512345678. Invoice #819-2024 due soon."
    redacted = redact_pii(text)
    assert "55512345678" in redacted
    assert "[REDACTED_PHONE]" not in redacted


def test_preserves_innocent_text():
    text = "Ticket #42: Need help connecting to the office VPN network on my laptop."
    assert redact_pii(text) == text
