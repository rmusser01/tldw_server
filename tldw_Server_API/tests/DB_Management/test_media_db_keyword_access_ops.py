from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


pytestmark = pytest.mark.unit


def _load_keyword_access_ops_module():
    module_name = "tldw_Server_API.app.core.DB_Management.media_db.runtime.keyword_access_ops"
    assert importlib.util.find_spec(module_name) is not None
    return importlib.import_module(module_name)


def _make_media_db(tmp_path: Path) -> MediaDatabase:
    db_path = tmp_path / "Media_DB_v2.db"
    return MediaDatabase(db_path=str(db_path), client_id="keyword-access-tests")


def test_keyword_access_helpers_rebind_on_media_database() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase as NativeMediaDatabase

    keyword_access_ops_module = _load_keyword_access_ops_module()

    assert NativeMediaDatabase.add_keyword is keyword_access_ops_module.add_keyword
    assert (
        NativeMediaDatabase.fetch_media_for_keywords
        is keyword_access_ops_module.fetch_media_for_keywords
    )


def test_add_keyword_forwards_to_keywords_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keyword_access_ops_module = _load_keyword_access_ops_module()

    repo_factory_calls: list[object] = []
    add_calls: list[tuple[str, object | None]] = []
    db = SimpleNamespace()
    conn = object()
    expected_result = (7, "kw-uuid")

    class _FakeRepo:
        def add(self, keyword: str, conn=None):
            add_calls.append((keyword, conn))
            return expected_result

    monkeypatch.setattr(
        keyword_access_ops_module.KeywordsRepository,
        "from_legacy_db",
        classmethod(
            lambda cls, _db: (
                repo_factory_calls.append(_db),
                _FakeRepo(),
            )[1]
        ),
    )

    result = keyword_access_ops_module.add_keyword(db, " Alpha ", conn=conn)

    assert repo_factory_calls == [db]
    assert add_calls == [(" Alpha ", conn)]
    assert result == expected_result


def test_fetch_media_for_keywords_rejects_non_list_input() -> None:
    keyword_access_ops_module = _load_keyword_access_ops_module()

    with pytest.raises(TypeError, match="Input 'keywords' must be a list of strings."):
        keyword_access_ops_module.fetch_media_for_keywords(SimpleNamespace(), "alpha")


def test_fetch_media_for_keywords_returns_empty_for_empty_or_blank_inputs() -> None:
    keyword_access_ops_module = _load_keyword_access_ops_module()

    assert keyword_access_ops_module.fetch_media_for_keywords(SimpleNamespace(), []) == {}
    assert keyword_access_ops_module.fetch_media_for_keywords(
        SimpleNamespace(),
        ["", "   ", None],
    ) == {}


def test_fetch_media_for_keywords_normalizes_groups_and_filters_trash(
    tmp_path: Path,
) -> None:
    keyword_access_ops_module = _load_keyword_access_ops_module()
    db = _make_media_db(tmp_path)

    media_a, _, _ = db.add_media_with_keywords(
        title="Alpha Beta Active",
        media_type="doc",
        content="A body",
        keywords=["Alpha", "beta"],
    )
    media_b, _, _ = db.add_media_with_keywords(
        title="Alpha Trash",
        media_type="doc",
        content="B body",
        keywords=["alpha"],
    )
    media_c, _, _ = db.add_media_with_keywords(
        title="Beta Active",
        media_type="doc",
        content="C body",
        keywords=["beta"],
    )
    db.mark_as_trash(media_b)

    without_trash = keyword_access_ops_module.fetch_media_for_keywords(
        db,
        [" alpha ", "BETA", "alpha", " "],
        include_trash=False,
    )
    with_trash = keyword_access_ops_module.fetch_media_for_keywords(
        db,
        [" alpha ", "BETA", "alpha", " "],
        include_trash=True,
    )

    assert list(without_trash.keys()) == ["alpha", "beta"]
    assert [item["id"] for item in without_trash["alpha"]] == [media_a]
    assert {item["id"] for item in without_trash["beta"]} == {media_a, media_c}
    assert {item["title"] for item in without_trash["beta"]} == {
        "Alpha Beta Active",
        "Beta Active",
    }

    assert {item["id"] for item in with_trash["alpha"]} == {media_a, media_b}
    assert {item["id"] for item in with_trash["beta"]} == {media_a, media_c}


def test_fetch_media_for_keywords_keys_by_identity_and_keeps_unknown_keywords_empty(
    tmp_path: Path,
) -> None:
    keyword_access_ops_module = _load_keyword_access_ops_module()
    db = _make_media_db(tmp_path)

    media_id, _, _ = db.add_media_with_keywords(
        title="Display casing",
        media_type="doc",
        content="D body",
        keywords=["Alpha"],
    )

    result = keyword_access_ops_module.fetch_media_for_keywords(db, ["ALPHA", "missing"])

    assert list(result) == ["alpha", "missing"]
    assert [item["id"] for item in result["alpha"]] == [media_id]
    assert result["missing"] == []
