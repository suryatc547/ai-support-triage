"""Unit tests for the deterministic email guardrail (validation_service)."""

from email.message import EmailMessage
from unittest.mock import patch

import pytest

from src.configs import security_config as cfg
from src.services.validation_service import validate_email


@pytest.fixture(autouse=True)
def mock_trusted_domains():
    with patch.object(cfg, "TRUSTED_DOMAINS", ["test.com"]):
        yield


def make_message(
    sender="user@test.com",
    subject="Need help",
    body="Password reset please.",
    attachments=None,
):
    msg = EmailMessage()
    msg["From"] = sender
    msg["Subject"] = subject
    msg.set_content(body)
    if attachments:
        for filename, data in attachments:
            if "." in filename:
                ext = filename.rsplit(".", 1)[1]
                subtype = "octet-stream"
                if ext in ("txt", "pdf"):
                    subtype = ext
                msg.add_attachment(
                    data,
                    maintype="application",
                    subtype=subtype,
                    filename=filename,
                )
            else:
                msg.add_attachment(
                    data,
                    maintype="application",
                    subtype="octet-stream",
                    filename=filename,
                )
    return msg


def test_trusted_internal_sender_passes():
    result = validate_email(make_message(sender="employee@test.com"))
    assert result.verdict == "pass"
    assert result.is_trusted is True
    assert result.score == 0


def test_subdomain_of_trusted_domain_passes():
    result = validate_email(make_message(sender="user@it.test.com"))
    assert result.verdict == "pass"
    assert result.is_trusted is True


def test_disposable_domain_quarantines():
    result = validate_email(make_message(sender="attacker@mailinator.com"))
    assert result.verdict == "quarantine"
    assert result.quarantined_reason is not None
    assert any(f.type == "disposable_domain" for f in result.findings)


def test_spam_domain_quarantines():
    result = validate_email(make_message(sender="spam@spamgourmet.com"))
    assert result.verdict == "quarantine"
    assert any(f.type == "spam_domain" for f in result.findings)


def test_typosquat_domain_flags():
    result = validate_email(make_message(sender="admin@t3st.com"))
    assert result.verdict == "flag"
    assert any(f.type == "typosquat" for f in result.findings)


def test_dangerous_attachment_quarantines():
    result = validate_email(
        make_message(sender="external@gmail.com", attachments=[("invoice.pdf.exe", b"MZ...")])
    )
    assert result.verdict == "quarantine"
    assert any(f.type == "dangerous_attachment" for f in result.findings)


def test_macro_office_attachment_flags():
    result = validate_email(
        make_message(sender="external@gmail.com", attachments=[("form.xlsm", b"PK...")])
    )
    assert result.verdict == "flag"
    assert any(f.type == "risky_attachment" for f in result.findings)


def test_archive_attachment_flags():
    result = validate_email(
        make_message(sender="external@gmail.com", attachments=[("bundle.zip", b"PK...")])
    )
    assert result.verdict == "flag"
    assert any(f.type == "risky_attachment" for f in result.findings)


def test_double_extension_flags():
    result = validate_email(
        make_message(sender="external@gmail.com", attachments=[("report.pdf.js", b"alert(1)")])
    )
    assert result.verdict == "quarantine"  # .js is a dangerous extension too
    assert any(f.type == "dangerous_attachment" for f in result.findings)


def test_clear_external_support_passes():
    result = validate_email(make_message(sender="customer@gmail.com"))
    assert result.verdict == "pass"
    assert result.score == 0


def test_oversized_attachment_flags():
    big = b"x" * (25 * 1024 * 1024 + 1)
    result = validate_email(
        make_message(sender="external@gmail.com", attachments=[("big.iso", big)])
    )
    # .iso is an archive-type extension -> already flagged; oversized adds low.
    assert any(f.type == "oversized_attachment" for f in result.findings)


def test_malformed_sender_flags():
    result = validate_email(make_message(sender="not-an-address"))
    assert result.verdict == "flag"
    assert any(f.type == "malformed_sender" for f in result.findings)


def test_external_sender_never_trusted():
    result = validate_email(make_message(sender="customer@attacker.com"))
    assert result.is_trusted is False
