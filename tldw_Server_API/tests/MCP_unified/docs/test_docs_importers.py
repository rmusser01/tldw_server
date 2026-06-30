from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs.errors import DocsError
from mcp_unified.docs.importers.local import DocsImportService
from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore


def _service(
    tmp_path: Path,
    root: Path,
    *,
    max_import_file_bytes: int = 2_000_000,
) -> tuple[DocsImportService, DocsCatalogStore]:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings(
        db_path=tmp_path / "docs.db",
        trusted_roots=(root.resolve(),),
        max_import_file_bytes=max_import_file_bytes,
    )
    return DocsImportService(settings=settings, store=store), store


def test_import_markdown_extracts_heading_chunks_keywords_and_collection(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "guide.md"
    path.write_text("# Install\n\nUse sqlite FTS for local docs.\n", encoding="utf-8")
    service, store = _service(tmp_path, root)
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    result = service.import_path(
        scope=scope,
        path=path,
        keywords=("setup",),
        collection_names=("Project Docs",),
    )

    assert result["status"] == "created"  # nosec B101
    assert result["documents"][0]["title"] == "Install"  # nosec B101
    assert result["documents"][0]["chunks"] >= 1  # nosec B101
    search_results = store.search_chunks(
        scope=scope,
        query="sqlite",
        limit=10,
        filters={"collection": "Project Docs", "keywords": ("setup",)},
    )
    document = store.get_document(scope, result["documents"][0]["id"], mode="full")
    assert [match["title"] for match in search_results] == ["Install"]  # nosec B101
    assert document["sections"][0]["heading"] == "Install"  # nosec B101


def test_import_static_html_without_web_dependencies_or_script_text(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "page.html"
    path.write_text(
        (
            "<html><body><h1>API</h1><p>Search docs with FTS.</p>"
            "<script>hiddenScriptText()</script><style>.hidden{display:none}</style>"
            "<noscript>hidden fallback</noscript></body></html>"
        ),
        encoding="utf-8",
    )
    service, store = _service(tmp_path, root)
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    result = service.import_path(
        scope=scope,
        path=path,
        keywords=(),
        collection_names=(),
    )

    document = store.get_document(scope, result["documents"][0]["id"], mode="full")
    assert result["status"] == "created"  # nosec B101
    assert result["documents"][0]["title"] == "API"  # nosec B101
    assert "Search docs with FTS." in document["text"]  # nosec B101
    assert "hiddenScriptText" not in document["text"]  # nosec B101
    assert "hidden fallback" not in document["text"]  # nosec B101


def test_import_rejects_path_outside_trusted_roots(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside.md"
    root.mkdir()
    outside.write_text("# Outside\n", encoding="utf-8")
    service, _store = _service(tmp_path, root)

    with pytest.raises(DocsError) as excinfo:
        service.import_path(
            scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
            path=outside,
            keywords=(),
            collection_names=(),
        )

    assert excinfo.value.code == "path_scope_denied"  # nosec B101


def test_import_rejects_symlink_escape_from_trusted_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside.md"
    root.mkdir()
    outside.write_text("# Outside\n", encoding="utf-8")
    link = root / "escape.md"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    service, _store = _service(tmp_path, root)

    with pytest.raises(DocsError) as excinfo:
        service.import_path(
            scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
            path=link,
            keywords=(),
            collection_names=(),
        )

    assert excinfo.value.code == "path_scope_denied"  # nosec B101


def test_import_rejects_files_larger_than_configured_limit(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "large.md"
    path.write_text("# Large\n\nToo much text.\n", encoding="utf-8")
    service, _store = _service(tmp_path, root, max_import_file_bytes=8)

    with pytest.raises(DocsError) as excinfo:
        service.import_path(
            scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
            path=path,
            keywords=(),
            collection_names=(),
        )

    assert excinfo.value.code == "import_file_too_large"  # nosec B101


def test_import_rejects_unsupported_file_format(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "data.bin"
    path.write_bytes(b"binary")
    service, _store = _service(tmp_path, root)

    with pytest.raises(DocsError) as excinfo:
        service.import_path(
            scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
            path=path,
            keywords=(),
            collection_names=(),
        )

    assert excinfo.value.code == "unsupported_import_format"  # nosec B101
