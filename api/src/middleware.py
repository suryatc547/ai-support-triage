"""ASGI middleware for request trace IDs and API key authentication."""

import json
import logging
import secrets
import time

from starlette.types import ASGIApp, Receive, Scope, Send

from .configs import config
from .logging_config import sanitise_trace_id, trace_context

logger = logging.getLogger("app.request")

_API_PREFIX = "/api/"
_EXEMPT_PATHS = {"/docs", "/openapi.json", "/redoc", "/health"}


def _header_value(scope: Scope, name: bytes) -> str | None:
    """Return the first value of a request header (case-insensitive), if any."""
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


class TraceIdMiddleware:
    """Assign a trace ID to every HTTP request and correlate its log lines.

    - Reuses a trusted ``X-Request-ID`` header when present (sanitised);
      otherwise generates a ``req-<12 hex>`` ID.
    - Emits ``START`` / ``END`` log lines with method, path, status, and
      duration around the request so a full request can be followed in logs.
    - Echoes the ID back via the ``X-Request-ID`` response header so clients
      and debugging tooling can correlate downstream work.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        trace_id = sanitise_trace_id(_header_value(scope, b"x-request-id"))
        method = scope.get("method", "")
        path = scope.get("path", "")
        query = scope.get("query_string", b"")
        target = f"{path}?{query.decode('latin-1')}" if query else path

        start = time.perf_counter()
        status: list[int] = []

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status.append(message["status"])
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", trace_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        with trace_context(trace_id):
            logger.info("START %s %s", method, target)
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                code = status[0] if status else 0
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("END %s %s -> %d in %.1fms", method, target, code, elapsed_ms)


class APIKeyMiddleware:
    """Enforce an API key on ``/api/*`` requests when keys are configured.

    Reads ``X-API-Key`` and compares it in constant time against the configured
    keys. API keys are loaded from ``API_KEYS`` / ``API_KEY`` env vars; when
    none are configured the middleware is a pass-through (local/dev mode).

    Interactive docs (``/docs``, ``/openapi.json``, ``/redoc``, ``/health``)
    stay open so the API remains explorable.
    """

    def __init__(self, app: ASGIApp, api_keys: tuple[str, ...] | None = None):
        self.app = app
        self._keys = api_keys if api_keys is not None else config.api_keys

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._keys:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith(_API_PREFIX) or path in _EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        provided = _header_value(scope, b"x-api-key")
        if provided is not None and any(
            secrets.compare_digest(provided, key) for key in self._keys
        ):
            await self.app(scope, receive, send)
            return

        await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps({"detail": "Invalid or missing API key"}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})
