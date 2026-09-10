from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.services.email_service import build_forward_message, forward_ticket_email


def _mock_ticket(**kwargs):
    defaults = {
        "id": 42,
        "subject": "Help with VPN connection",
        "body": "I cannot connect to the company VPN since morning.",
        "sender_email": "employee@example.com",
        "ticket_type": "it_support",
        "status": "open",
        "security_flag": None,
        "suspicion_score": 0,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _mock_user(**kwargs):
    defaults = {
        "id": 2,
        "name": "IT Admin",
        "email": "it-admin@test.com",
        "department": "IT Support",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_forward_message_headers_and_content():
    ticket = _mock_ticket()
    assignee = _mock_user()
    from_addr = "support-bot@example.com"

    msg = build_forward_message(ticket, assignee, from_addr)

    assert msg["Subject"] == "Fwd: [Ticket #42] Help with VPN connection"
    assert msg["From"] == "support-bot@example.com"
    assert msg["To"] == "it-admin@test.com"
    assert msg["Reply-To"] == "employee@example.com"

    content = msg.get_content()
    assert "Ticket ID   : #42" in content
    assert "Department  : IT Support" in content
    assert "Assigned To : IT Admin <it-admin@test.com>" in content
    assert "Category    : it_support" in content
    assert "I cannot connect to the company VPN since morning." in content


def test_build_forward_message_includes_security_flag():
    ticket = _mock_ticket(security_flag="flagged", suspicion_score=25)
    assignee = _mock_user()

    msg = build_forward_message(ticket, assignee, "support@example.com")
    content = msg.get_content()
    assert "Security    : flagged (score 25)" in content


def test_forward_ticket_email_missing_assignee_or_email():
    ticket = _mock_ticket()
    assert forward_ticket_email(ticket, None) is False
    assert forward_ticket_email(ticket, SimpleNamespace(email=None)) is False


def test_forward_ticket_email_missing_smtp_credentials(monkeypatch):
    monkeypatch.setattr(
        "src.services.email_service.smtp_config",
        {"host": "smtp.gmail.com", "port": 587, "user": None, "password": None},
    )
    ticket = _mock_ticket()
    assignee = _mock_user()

    with patch("smtplib.SMTP") as mock_smtp:
        result = forward_ticket_email(ticket, assignee)
        assert result is False
        mock_smtp.assert_not_called()


def test_forward_ticket_email_starttls_success(monkeypatch):
    monkeypatch.setattr(
        "src.services.email_service.smtp_config",
        {
            "host": "smtp.gmail.com",
            "port": 587,
            "user": "support@gmail.com",
            "password": "secret-password",
        },
    )
    ticket = _mock_ticket()
    assignee = _mock_user()

    mock_server = MagicMock()
    with patch("smtplib.SMTP", return_value=mock_server) as mock_smtp_cls:
        mock_server.__enter__.return_value = mock_server
        result = forward_ticket_email(ticket, assignee)

        assert result is True
        mock_smtp_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=30)
        assert mock_server.starttls.called
        mock_server.login.assert_called_once_with("support@gmail.com", "secret-password")
        mock_server.send_message.assert_called_once()
        sent_msg = mock_server.send_message.call_args[0][0]
        assert sent_msg["To"] == "it-admin@test.com"


def test_forward_ticket_email_ssl_success(monkeypatch):
    monkeypatch.setattr(
        "src.services.email_service.smtp_config",
        {
            "host": "smtp.gmail.com",
            "port": 465,
            "user": "support@gmail.com",
            "password": "secret-password",
        },
    )
    ticket = _mock_ticket()
    assignee = _mock_user()

    mock_server = MagicMock()
    with patch("smtplib.SMTP_SSL", return_value=mock_server) as mock_ssl_cls:
        mock_server.__enter__.return_value = mock_server
        result = forward_ticket_email(ticket, assignee)

        assert result is True
        mock_ssl_cls.assert_called_once()
        mock_server.login.assert_called_once_with("support@gmail.com", "secret-password")
        mock_server.send_message.assert_called_once()


def test_forward_ticket_email_smtp_failure_is_non_fatal(monkeypatch):
    monkeypatch.setattr(
        "src.services.email_service.smtp_config",
        {
            "host": "smtp.gmail.com",
            "port": 587,
            "user": "support@gmail.com",
            "password": "secret-password",
        },
    )
    ticket = _mock_ticket()
    assignee = _mock_user()

    with patch("smtplib.SMTP", side_effect=Exception("Connection refused")):
        result = forward_ticket_email(ticket, assignee)
        assert result is False
