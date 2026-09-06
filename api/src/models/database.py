from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from .models import Base

_REPO_ROOT = Path(__file__).resolve().parents[2]
SQLALCHEMY_DATABASE_URL = f"sqlite:///{(_REPO_ROOT / 'support.db').as_posix()}"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _run_lightweight_migrations():
    """Additively migrate the SQLite schema to match the current models.

    SQLAlchemy's `create_all` only creates missing tables — it never adds
    columns to existing ones. This performs idempotent `ALTER TABLE` statements
    so existing databases stay in sync without dropping data.
    """
    inspector = inspect(engine)
    tables = {t: {c["name"] for c in inspector.get_columns(t)} for t in inspector.get_table_names()}

    if "users" in tables and "expertise" not in tables["users"]:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN expertise TEXT"))

    if "tickets" in tables:
        cols = tables["tickets"]
        if "message_id" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE tickets ADD COLUMN message_id VARCHAR"))
        # Unique index (nullable so existing rows without a message_id are allowed;
        # SQLite permits multiple NULLs in a unique index).
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ix_tickets_message_id "
                    "ON tickets (message_id)"
                )
            )
        if "security_flag" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE tickets ADD COLUMN security_flag VARCHAR"))
        if "suspicion_score" not in cols:
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE tickets ADD COLUMN suspicion_score INTEGER DEFAULT 0")
                )
        if "validation_findings" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE tickets ADD COLUMN validation_findings TEXT"))


def init_db():
    Base.metadata.create_all(bind=engine)
    _run_lightweight_migrations()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
