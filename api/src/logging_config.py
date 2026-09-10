"""Centralised logging configuration with per-request trace IDs.

Every log line emitted while a trace ID is active — set by
:class:`src.middleware.TraceIdMiddleware` for HTTP requests, or inside
``imap_service.fetch_and_process_emails`` for IMAP sync runs — is prefixed with
that ID so a full request/run can be followed from start to finish in the logs.
Records produced outside any active context use ``-`` as the ID so the prefix
column is always present and unambiguous.
"""

import logging
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

# Active trace ID for the current execution context; empty when outside a request.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

LOG_FORMAT = "[%(request_id)s] %(levelname)-8s %(asctime)s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_MAX_TRACE_ID_LENGTH = 64
_SAFE_TRACE_ID = re.compile(rf"^[A-Za-z0-9._:+=;-]{{1,{_MAX_TRACE_ID_LENGTH}}}$")


class RequestIdFilter(logging.Filter):
    """Attach the active trace ID to every log record.

    Records produced outside any request/sync context get ``-`` so the prefix
    stays present and unambiguous.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        return True


def new_trace_id(prefix: str = "req") -> str:
    """Generate a short unique trace ID, e.g. ``req-a1b2c3d4e5f6``."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def sanitise_trace_id(value: str | None) -> str:
    """Sanitise an externally supplied trace ID.

    The value is accepted only when it matches the safe-character whitelist
    with a bounded length; otherwise a fresh ``req-*`` ID is generated. This
    keeps untrusted client input (no spaces, newlines, or control characters)
    out of the log prefix.
    """
    if value:
        candidate = value.strip()
        if _SAFE_TRACE_ID.fullmatch(candidate):
            return candidate
    return new_trace_id()


@contextmanager
def trace_context(trace_id: str):
    """Run a block with ``trace_id`` active, restoring the previous value after."""
    token = request_id_var.set(trace_id)
    try:
        yield
    finally:
        request_id_var.reset(token)


def setup_logging(level: int = logging.INFO) -> None:
    """Replace the root logger config with the trace-ID-aware format.

    The :class:`RequestIdFilter` is attached to the root *handlers* so every
    record they format carries a ``request_id`` attribute. Handlers are the
    correct hook here: a record raised on a descendant logger (e.g. ``src.*``,
    ``httpx``) reaches the root handlers without ever passing through the root
    logger's own filter chain (``Logger.callHandlers`` walks ancestors only).
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(RequestIdFilter())
