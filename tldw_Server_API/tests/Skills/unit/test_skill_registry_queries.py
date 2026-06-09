from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


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
) -> None:
    db.insert_skill_registry(
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

    with pytest.raises(ValueError, match="Unsupported skill registry sort field"):
        skills_db.list_skill_registry(sort="directory_path")
