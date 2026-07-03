from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs.errors import DocsError
from mcp_unified.docs.importers.base import chunks_from_text
from mcp_unified.docs.importers.html import parse_html_document
from mcp_unified.docs.importers.local import DocsImportService
from mcp_unified.docs.importers.markdown import parse_markdown
from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore

pytestmark = pytest.mark.unit


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


def test_import_directory_skips_unsupported_files(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "guide.md").write_text("# Guide\n\nSQLite docs.\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"png")
    service, _store = _service(tmp_path, root)

    result = service.import_path(
        scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
        path=root,
        keywords=(),
        collection_names=(),
    )

    assert result["status"] == "created"  # nosec B101
    assert [document["title"] for document in result["documents"]] == ["Guide"]  # nosec B101


def test_import_directory_materializes_metadata_iterables_for_each_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.md").write_text("# A\n\nSQLite docs.\n", encoding="utf-8")
    (root / "b.md").write_text("# B\n\nSQLite docs.\n", encoding="utf-8")
    service, store = _service(tmp_path, root)
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    result = service.import_path(
        scope=scope,
        path=root,
        keywords=(keyword for keyword in ("shared",)),
        collection_names=(collection for collection in ("Project Docs",)),
    )

    search_results = store.search_chunks(
        scope=scope,
        query="SQLite",
        limit=10,
        filters={"collection": "Project Docs", "keywords": ("shared",)},
    )
    assert result["status"] == "created"  # nosec B101
    assert sorted(document["title"] for document in result["documents"]) == ["A", "B"]  # nosec B101
    assert sorted(match["title"] for match in search_results) == ["A", "B"]  # nosec B101


def test_docs_error_initializes_exception_message_args() -> None:
    error = DocsError(code="example", message="Human readable", details={"field": "value"})

    assert error.args == ("Human readable",)  # nosec B101
    assert str(error) == "example: Human readable"  # nosec B101


def test_chunks_from_text_rejects_non_progressing_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunks_from_text("abcdef", max_chars=4, overlap=4)


def test_markdown_parser_ignores_fenced_headings_and_keeps_first_heading_title(tmp_path: Path) -> None:
    path = tmp_path / "Guide.md"
    parsed = parse_markdown(
        path,
        "```md\n# Not a heading\n```\n# Guide\n\n## Details\n",
        "markdown",
    )

    assert parsed.title == "Guide"  # nosec B101
    assert [section.heading for section in parsed.sections] == ["Guide", "Details"]  # nosec B101
    assert parsed.sections[0].end_char == parsed.sections[1].start_char  # nosec B101
    assert parsed.sections[1].end_char == len("```md\n# Not a heading\n```\n# Guide\n\n## Details\n")  # nosec B101


def test_markdown_parser_requires_matching_fence_style(tmp_path: Path) -> None:
    path = tmp_path / "Guide.md"
    parsed = parse_markdown(path, "```md\n# Still fenced\n~~~\n# Also fenced\n```\n# Real\n", "markdown")

    assert parsed.title == "Real"  # nosec B101
    assert [section.heading for section in parsed.sections] == ["Real"]  # nosec B101


def test_static_html_preserves_inline_text_breaks_and_nested_headings() -> None:
    parsed = parse_html_document(
        text="<h1>First <h2>Second</h2></h1><p>Hello <strong>world</strong>.</p><p>Next<br>line</p>",
        title_hint="fallback",
        canonical_uri="memory://inline",
    )

    assert [section.heading for section in parsed.sections] == ["First", "Second"]  # nosec B101
    assert "Hello world." in parsed.text  # nosec B101
    assert "Next\nline" in parsed.text  # nosec B101


def test_static_html_preserves_whitespace_between_inline_elements() -> None:
    parsed = parse_html_document(
        text="<p><span>Hello</span> <span>World</span></p>",
        title_hint="fallback",
        canonical_uri="memory://inline-spacing",
    )

    assert "Hello World" in parsed.text  # nosec B101
    assert "HelloWorld" not in parsed.text  # nosec B101


def test_static_html_preserves_div_breaks_and_preformatted_text() -> None:
    parsed = parse_html_document(
        text="<div>First</div><div>Second</div><pre>def main():\n    return 1\n</pre>",
        title_hint="fallback",
        canonical_uri="memory://pre",
    )

    assert "First\nSecond" in parsed.text  # nosec B101
    assert "def main():\n    return 1" in parsed.text  # nosec B101


def test_import_wraps_non_utf8_files_in_docs_error(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "latin1.md"
    path.write_bytes("caf\xe9".encode("latin-1"))
    service, _store = _service(tmp_path, root)

    with pytest.raises(DocsError) as excinfo:
        service.import_path(
            scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
            path=path,
            keywords=(),
            collection_names=(),
        )

    assert excinfo.value.code == "import_file_decode_error"  # nosec B101
    assert excinfo.value.details["path"] == str(path.resolve())  # nosec B101


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
