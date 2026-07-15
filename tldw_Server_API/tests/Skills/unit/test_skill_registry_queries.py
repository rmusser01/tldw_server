from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def skills_db(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(db_path=tmp_path / "ChaChaNotes.db", client_id="skills_registry_query_test")
    yield db
    db.close_connection()


def _insert_skill(
    db: CharactersRAGDB,
    name: str,
    *,
    description: str = "General skill",
    argument_hint: str | None = None,
    user_invocable: bool = True,
    allowed_tools: list[str] | None = None,
    model: str | None = None,
    context: str = "inline",
) -> str:
    return db.insert_skill_registry(
        {
            "name": name,
            "description": description,
            "argument_hint": argument_hint,
            "user_invocable": user_invocable,
            "allowed_tools": allowed_tools,
            "model": model,
            "context": context,
            "directory_path": f"/tmp/skills/{name}",
            "file_hash": f"hash-{name}",
        }
    )


def test_skill_registry_can_find_a_row_by_uuid(skills_db: CharactersRAGDB) -> None:
    skill_uuid = _insert_skill(skills_db, "uuid-lookup")

    row = skills_db.get_skill_registry_by_uuid(skill_uuid)

    assert row is not None
    assert row["name"] == "uuid-lookup"
    assert skills_db.get_skill_registry_by_uuid("missing-uuid") is None


def test_skill_registry_filters_and_sorts_before_pagination(skills_db: CharactersRAGDB) -> None:
    for index in range(12):
        _insert_skill(skills_db, f"alpha-{index:02d}", context="inline")
    _insert_skill(
        skills_db,
        "beta-first",
        allowed_tools=["Read"],
        model="gpt-4o",
        context="fork",
    )
    _insert_skill(
        skills_db,
        "beta-second",
        allowed_tools=["Grep"],
        model="gpt-4o",
        context="fork",
    )

    rows = skills_db.list_skill_registry(
        context="fork",
        has_tools=True,
        model="gpt-4o",
        sort="name",
        order="desc",
        limit=1,
        offset=0,
    )

    assert [row["name"] for row in rows] == ["beta-second"]
    assert skills_db.count_skill_registry(
        context="fork",
        has_tools=True,
        model="gpt-4o",
    ) == 2


def test_skill_registry_explicit_user_invocable_filter_can_find_hidden(
    skills_db: CharactersRAGDB,
) -> None:
    _insert_skill(skills_db, "visible", user_invocable=True)
    _insert_skill(skills_db, "hidden", user_invocable=False)

    rows = skills_db.list_skill_registry(user_invocable=False, limit=10, offset=0)

    assert [row["name"] for row in rows] == ["hidden"]
    assert skills_db.count_skill_registry(user_invocable=False) == 1


def test_skill_registry_rejects_unapproved_sort_fields(skills_db: CharactersRAGDB) -> None:
    _insert_skill(skills_db, "visible")

    with pytest.raises(InputError, match="Unsupported skill registry sort field"):
        skills_db.list_skill_registry(sort="directory_path")


def test_restore_skill_registry_reactivates_soft_deleted_row(skills_db: CharactersRAGDB) -> None:
    _insert_skill(skills_db, "restorable", description="Before")

    skills_db.mark_skill_registry_deleted("restorable", expected_version=1)
    assert skills_db.get_skill_registry("restorable", include_deleted=False) is None

    skills_db.restore_skill_registry(
        "restorable",
        {"description": "After"},
        expected_version=2,
    )

    restored = skills_db.get_skill_registry("restorable", include_deleted=False)
    assert restored is not None
    assert restored["description"] == "After"
    assert not restored["deleted"]
    assert restored["version"] == 3


def test_bulk_mark_skill_registry_deleted_rolls_back_on_version_conflict(
    skills_db: CharactersRAGDB,
) -> None:
    """A stale item in a bulk delete must not soft-delete earlier rows."""
    _insert_skill(skills_db, "bulk-atomic-a")
    _insert_skill(skills_db, "bulk-atomic-b")
    skills_db.update_skill_registry(
        "bulk-atomic-b",
        {"description": "newer"},
        expected_version=1,
    )

    with pytest.raises(ConflictError):
        skills_db.bulk_mark_skill_registry_deleted(
            [
                ("bulk-atomic-a", 1),
                ("bulk-atomic-b", 1),
            ]
        )

    assert skills_db.get_skill_registry("bulk-atomic-a", include_deleted=False) is not None
    assert skills_db.get_skill_registry("bulk-atomic-b", include_deleted=False) is not None


def test_bulk_mark_skill_registry_deleted_allows_unknown_versions(
    skills_db: CharactersRAGDB,
) -> None:
    """Rows with omitted versions remain compatible with unversioned deletes."""
    _insert_skill(skills_db, "bulk-unversioned")
    skills_db.update_skill_registry(
        "bulk-unversioned",
        {"description": "changed after list"},
        expected_version=1,
    )

    deleted = skills_db.bulk_mark_skill_registry_deleted([("bulk-unversioned", None)])

    assert [item["name"] for item in deleted] == ["bulk-unversioned"]
    assert skills_db.get_skill_registry("bulk-unversioned", include_deleted=False) is None
    deleted_row = skills_db.get_skill_registry("bulk-unversioned", include_deleted=True)
    assert deleted_row is not None
    assert deleted_row["version"] == 3


def test_bulk_mark_skill_registry_deleted_rejects_missing_directory_path(
    skills_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed rows fail closed instead of persisting the string ``None``."""
    _insert_skill(skills_db, "bulk-missing-path")
    original_row_to_dict = skills_db._skill_row_to_dict

    def _without_directory_path(row):  # noqa: ANN001, ANN202
        item = original_row_to_dict(row)
        if item is not None and item["name"] == "bulk-missing-path":
            item["directory_path"] = None
        return item

    monkeypatch.setattr(skills_db, "_skill_row_to_dict", _without_directory_path)

    with pytest.raises(InputError, match="directory path"):
        skills_db.bulk_mark_skill_registry_deleted([("bulk-missing-path", 1)])

    assert skills_db.get_skill_registry("bulk-missing-path", include_deleted=False) is not None


def test_deleted_skill_registry_lists_archive_path_and_purges_by_version(
    skills_db: CharactersRAGDB,
) -> None:
    """Trash queries return only deleted rows and permanent deletion is versioned."""
    _insert_skill(skills_db, "active-skill")
    _insert_skill(skills_db, "trashed-skill")
    archive_path = "/tmp/skills/.trash/trashed-uuid"

    skills_db.mark_skill_registry_deleted(
        "trashed-skill",
        expected_version=1,
        directory_path=archive_path,
    )

    rows = skills_db.list_deleted_skill_registry(limit=10, offset=0)
    assert [row["name"] for row in rows] == ["trashed-skill"]
    assert rows[0]["directory_path"] == archive_path
    assert skills_db.count_deleted_skill_registry() == 1

    with pytest.raises(ConflictError):
        skills_db.purge_skill_registry("trashed-skill", expected_version=1)

    skills_db.purge_skill_registry("trashed-skill", expected_version=2)
    assert skills_db.get_skill_registry("trashed-skill", include_deleted=True) is None
    assert skills_db.get_skill_registry("active-skill", include_deleted=False) is not None
