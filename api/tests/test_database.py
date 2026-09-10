"""Tests for SQLite connection pragma harding in database.py."""

import sqlite3

from sqlalchemy import create_engine, event, text

from src.models import database

_BUSY_TIMEOUT_MS = 5000


def test_pragmas_enforce_foreign_keys_and_busy_timeout():
    connection = sqlite3.connect(":memory:")
    try:
        database._apply_sqlite_pragmas(connection, None)
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == _BUSY_TIMEOUT_MS
    finally:
        connection.close()


def test_pragmas_switch_journal_mode_to_wal(tmp_path):
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    engine = create_engine(url)
    event.listen(engine, "connect", database._apply_sqlite_pragmas)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    finally:
        engine.dispose()
