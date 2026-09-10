"""Tests for API key authentication middleware and key config parsing."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.configs.config import _parse_api_keys
from src.middleware import APIKeyMiddleware


def test_parse_api_keys_splits_and_trims():
    assert _parse_api_keys("key-one, key-two ,key-three") == ("key-one", "key-two", "key-three")
    assert _parse_api_keys("") == ()
    assert _parse_api_keys(None) == ()
    assert _parse_api_keys("  ,  ,") == ()


@pytest.fixture
def protected_app():
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, api_keys=("secret-1", "secret-2"))

    @app.get("/api/tickets")
    def tickets():
        return {"items": []}

    @app.get("/api/users")
    def users():
        return [{"id": 1}]

    @app.get("/docs")
    def docs():
        return {"kind": "docs"}

    @app.get("/openapi.json")
    def openapi():
        return {"kind": "openapi"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def test_missing_key_rejected(protected_app):
    with TestClient(protected_app) as client:
        response = client.get("/api/tickets")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_wrong_key_rejected(protected_app):
    with TestClient(protected_app) as client:
        response = client.get("/api/tickets", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_valid_key_accepted(protected_app):
    with TestClient(protected_app) as client:
        response = client.get("/api/tickets", headers={"X-API-Key": "secret-1"})
    assert response.status_code == 200

    with TestClient(protected_app) as client:
        response = client.get("/api/users", headers={"X-API-Key": "secret-2"})
    assert response.status_code == 200


def test_exempt_paths_require_no_key(protected_app):
    with TestClient(protected_app) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/health").status_code == 200


def test_disabled_when_no_keys_configured():
    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, api_keys=())

    @app.get("/api/tickets")
    def tickets():
        return {"items": []}

    with TestClient(app) as client:
        assert client.get("/api/tickets").status_code == 200
