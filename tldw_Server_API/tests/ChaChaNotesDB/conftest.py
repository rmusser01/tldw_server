import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def client_id() -> str:
    return "test_client_001"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provides a temporary path for the database file for each test."""
    return tmp_path / "test_db.sqlite"


@pytest.fixture(scope="function")
def db_instance(db_path: Path, client_id: str) -> Iterator[CharactersRAGDB]:
    """Creates a DB instance for each test, ensuring a fresh database."""
    current_db_path = Path(db_path)

    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{current_db_path}{suffix}")
        if path.exists():
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - best-effort fixture cleanup
                print(f"Warning: Could not unlink {path}: {exc}")

    db: CharactersRAGDB | None = None
    try:
        db = CharactersRAGDB(current_db_path, client_id)
        yield db
    finally:
        if db:
            db.close_connection()
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{current_db_path}{suffix}")
            if path.exists():
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass


@pytest.fixture
def mem_db_instance(client_id: str) -> Iterator[CharactersRAGDB]:
    """Creates an in-memory DB instance."""
    db = CharactersRAGDB(":memory:", client_id)
    yield db
    db.close_connection()


def get_current_utc_timestamp_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _create_sample_card_data(
    name_suffix: str = "",
    client_id_override: str | None = None,
) -> dict[str, object]:
    return {
        "name": f"Test Character {name_suffix}",
        "description": "A test character.",
        "personality": "Testy",
        "scenario": "A test scenario.",
        "image": b"testimagebytes",
        "first_message": "Hello, test!",
        "alternate_greetings": json.dumps(["Hi", "Hey"]),
        "tags": json.dumps(["test", "sample"]),
        "extensions": json.dumps({"custom_field": "value"}),
        "client_id": client_id_override,
    }
