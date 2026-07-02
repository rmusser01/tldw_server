from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs.importers.local import DocsImportService
from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore

pytestmark = pytest.mark.unit


def _service(tmp_path: Path, root: Path) -> tuple[DocsImportService, DocsCatalogStore]:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root.resolve(),))
    return DocsImportService(settings=settings, store=store), store


def test_import_file_creates_local_file_source_and_link(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nSQLite sync source.\n", encoding="utf-8")
    service, store = _service(tmp_path, root)
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    result = service.import_path(scope=scope, path=guide, keywords=("setup",), collection_names=("Docs",))

    sources = store.list_sources(scope=scope)
    links = store.source_document_links(scope=scope, source_id=sources[0]["id"])
    assert result["source"]["source_type"] == "local_file"  # nosec B101
    assert sources[0]["source_type"] == "local_file"  # nosec B101
    assert sources[0]["metadata"]["default_keywords"] == ["setup"]  # nosec B101
    assert links[0]["document_id"] == result["documents"][0]["id"]  # nosec B101
    assert links[0]["status"] == "active"  # nosec B101


def test_import_directory_creates_one_directory_source_with_file_items(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nSQLite A.\n", encoding="utf-8")
    (root / "b.md").write_text("# B\n\nSQLite B.\n", encoding="utf-8")
    service, store = _service(tmp_path, root)
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    service.import_path(scope=scope, path=root, keywords=("shared",), collection_names=("Docs",))

    sources = store.list_sources(scope=scope)
    links = store.source_document_links(scope=scope, source_id=sources[0]["id"])
    assert len(sources) == 1  # nosec B101
    assert sources[0]["source_type"] == "local_directory"  # nosec B101
    assert sorted(link["source_item_uri"].rsplit("/", 1)[-1] for link in links) == ["a.md", "b.md"]  # nosec B101
