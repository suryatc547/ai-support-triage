"""Tests for trace-ID-aware logging configuration and middleware."""

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.logging_config import (
    DATE_FORMAT,
    LOG_FORMAT,
    RequestIdFilter,
    new_trace_id,
    request_id_var,
    sanitise_trace_id,
    setup_logging,
    trace_context,
)
from src.middleware import TraceIdMiddleware

setup_logging()

_APP_LOGGER_NAMES = {"app.request", "test.endpoint"}


class CaptureHandler(logging.Handler):
    """Snapshot log records, tagging each with the active trace ID via its own filter."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.addFilter(RequestIdFilter())
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured():
    handler = CaptureHandler()
    logging.getLogger().addHandler(handler)
    try:
        yield handler
    finally:
        logging.getLogger().removeHandler(handler)


# ---------------------------------------------------------------------------
# Trace ID helpers
# ---------------------------------------------------------------------------


def test_new_trace_id_has_prefix_and_length():
    tid = new_trace_id("req")
    assert tid.startswith("req-")
    assert len(tid) == 4 + 12
    assert len(new_trace_id()) > 0


def test_sanitise_accepts_safe_external_id():
    assert sanitise_trace_id("client-abc_123:XYZ") == "client-abc_123:XYZ"


def test_sanitise_strips_surrounding_whitespace():
    assert sanitise_trace_id("  track-9  ") == "track-9"


def test_sanitise_rejects_unsafe_external_id():
    tid = sanitise_trace_id("bad id with\ninjection!")
    assert tid.startswith("req-")


def test_sanitise_rejects_oversized_external_id():
    assert sanitise_trace_id("x" * 100).startswith("req-")


def test_sanitise_handles_none_and_empty():
    assert sanitise_trace_id(None).startswith("req-")
    assert sanitise_trace_id("   ").startswith("req-")


# ---------------------------------------------------------------------------
# RequestIdFilter + format
# ---------------------------------------------------------------------------


def _record(message: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("test", logging.INFO, "file.py", 1, message, (), None)


def test_log_format_prefixes_request_id():
    assert "%(request_id)s" in LOG_FORMAT


def test_filter_uses_active_context():
    with trace_context("trace-1"):
        record = _record()
        assert RequestIdFilter().filter(record) is True
        assert record.request_id == "trace-1"


def test_filter_defaults_to_dash():
    record = _record()
    RequestIdFilter().filter(record)
    assert record.request_id == "-"


def test_trace_context_restores_previous_value():
    with trace_context("outer"):
        with trace_context("inner"):
            assert request_id_var.get() == "inner"
        assert request_id_var.get() == "outer"
    assert request_id_var.get() == ""


def test_setup_logging_attaches_filter_to_handlers_once():
    setup_logging()
    setup_logging()
    root = logging.getLogger()
    assert root.handlers
    for handler in root.handlers:
        matches = [f for f in handler.filters if isinstance(f, RequestIdFilter)]
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# Middleware end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_logger():
    app = FastAPI()
    app.add_middleware(TraceIdMiddleware)

    @app.get("/hello")
    def hello(name: str = "world"):
        logging.getLogger("test.endpoint").info("handling hello for %s", name)
        return {"message": "hi"}

    return app


def _app_records(handler):
    return [r for r in handler.records if r.name in _APP_LOGGER_NAMES]


def _format_messages(records):
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    return [formatter.format(r) for r in records]


def test_middleware_prefixes_logs_and_echoes_header(app_with_logger, captured):
    with TestClient(app_with_logger) as client:
        response = client.get("/hello?name=support", headers={"X-Request-ID": "track-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "track-123"

    app_records = _app_records(captured)
    assert app_records
    for rec in app_records:
        assert rec.request_id == "track-123"
    for line in _format_messages(app_records):
        assert line.startswith("[track-123]")

    messages = [rec.getMessage() for rec in app_records]
    start = next(i for i, m in enumerate(messages) if m == "START GET /hello?name=support")
    end = next(
        i for i, m in enumerate(messages) if m.startswith("END GET /hello?name=support -> 200 in ")
    )
    assert start < end
    assert "handling hello for support" in messages


def test_middleware_generates_id_when_header_missing(app_with_logger, captured):
    with TestClient(app_with_logger) as client:
        response = client.get("/hello")

    trace_id = response.headers["x-request-id"]
    assert trace_id.startswith("req-")
    app_records = _app_records(captured)
    assert app_records
    assert {rec.request_id for rec in app_records} == {trace_id}


def test_middleware_sanitises_unsafe_header(app_with_logger):
    with TestClient(app_with_logger) as client:
        response = client.get("/hello", headers={"X-Request-ID": "bad\ninjection!"})
    assert response.headers["x-request-id"].startswith("req-")


def test_context_restores_after_request(app_with_logger):
    with trace_context("outer"):
        with TestClient(app_with_logger) as client:
            client.get("/hello", headers={"X-Request-ID": "track-9"})
        assert request_id_var.get() == "outer"
    assert request_id_var.get() == ""
