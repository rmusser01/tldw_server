# Standalone MCP Docs Stage 4A Source Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded `docs.sync_source` support for already-known local and URL docs sources in the standalone MCP docs corpus.

**Architecture:** Extend the runtime-neutral `mcp_unified.docs` package with explicit source registry tables, source-population hooks in import/URL ingest paths, and a deterministic `DocsSourceSyncService`. Keep sync execution synchronous, SQLite + FTS5 only, and isolated from `tldw_Server_API`; the built-in `tldw_server` MCP module remains a thin host adapter.

**Tech Stack:** Python 3.10+, dataclasses, stdlib `sqlite3`, `pathlib`, existing Stage 1 local importers, existing Stage 2 URL policy/fetch/extraction seams, pytest with fake resolver/transport objects, Bandit for touched Python paths.

---

## Source References

- Design spec: `Docs/superpowers/specs/2026-07-01-standalone-mcp-docs-stage4a-sync-source-design.md`
- Stage 1 plan: `Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md`
- Stage 2 plan: `Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-url-acquisition-implementation-plan.md`
- Stage 3 plan: `Docs/superpowers/plans/2026-07-01-standalone-mcp-docs-stage3-server-mounting-plan.md`
- Backlog design task: `TASK-12091`
- Backlog planning task: `TASK-12092`

## Scope

Included in Stage 4A.1:

- `DocsSettings` source-sync limits and query-persistence policy.
- `docs_sources`, `docs_source_documents`, and `docs_sync_runs`.
- `docs_documents.lifecycle_status` and `docs_documents.preserve_on_source_tombstone`.
- Source population from `docs.import_path` and `docs.ingest_url`.
- Query-bearing URL source creation only when `persist_url_query_strings=true`.
- `docs.list(kind="sources")`.
- `docs.sync_source` for `local_file`, `local_directory`, and `url_page` sources.
- Strict dry-run with no corpus DB mutation.
- Sync-aware document/chunk replacement that merges existing user collections/keywords with source defaults.
- Thin host exposure through `DocsModule`.

Excluded from Stage 4A.1:

- `url_sitemap` registration and sync. The first implementation PR should keep sitemap disabled and unreachable.
- Recursive crawling, broad link discovery, browser automation, cookies, Playwright, embeddings, rerankers, Jobs/Scheduler wrappers, Media DB, ChromaDB, and host RAG bridges.

## File Structure

Create:

- `apps/mcp-unified/src/mcp_unified/docs/source_utils.py` - pure helpers for file URI canonicalization, URL redaction/query detection, source default metadata, and source item URIs.
- `apps/mcp-unified/src/mcp_unified/docs/sync.py` - `DocsSourceSyncService`, sync request/result dataclasses, local and URL sync orchestration.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_sources_store.py` - source schema, store helper, source list, sync run, and metadata-preservation tests.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py` - import and URL ingest source creation/linking tests.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py` - local and URL `docs.sync_source` service/tool tests.

Modify:

- `apps/mcp-unified/src/mcp_unified/docs/models.py` - add source/sync literals and dataclasses.
- `apps/mcp-unified/src/mcp_unified/docs/settings.py` - add source-sync and query-persistence settings.
- `apps/mcp-unified/src/mcp_unified/docs/store/schema.sql` - add Stage 4A tables and columns for fresh databases.
- `apps/mcp-unified/src/mcp_unified/docs/store/sqlite.py` - add migration guards, source helpers, source-aware document upsert, source list, run records, and lifecycle-aware search/list filtering.
- `apps/mcp-unified/src/mcp_unified/docs/importers/local.py` - create/link local file or directory sources while preserving current import behavior.
- `apps/mcp-unified/src/mcp_unified/docs/acquisition/service.py` - create/link URL page sources after successful ingest, enforce query persistence policy for refreshable sources.
- `apps/mcp-unified/src/mcp_unified/docs/mcp_module.py` - advertise/execute `docs.sync_source`, return real sources, and report source-sync capability in `docs.status`.
- `apps/mcp-unified/src/mcp_unified/docs/__init__.py` - export new source/sync dataclasses if tests need public imports.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py` - no new business logic; host validation may need `docs.sync_source` argument checks.
- Existing docs tests under `tldw_Server_API/tests/MCP_unified/docs/` - extend settings, provider, acquisition, importer, host adapter, and boundary tests.

Do not modify:

- `pyproject.toml` for new required dependencies.
- `tldw_Server_API` scraping services.
- Media DB/RAG/ChromaDB services.
- Jobs/Scheduler code.

## Shared Contracts

Use these names consistently across implementation tasks.

```python
SourceType = Literal["local_file", "local_directory", "url_page", "url_sitemap"]
SourceLinkStatus = Literal["active", "tombstoned", "failed"]
SyncMode = Literal["dry_run", "apply"]
StalePolicy = Literal["report", "tombstone"]
SyncItemStatus = Literal["created", "updated", "unchanged", "missing", "tombstoned", "failed", "skipped"]
SyncRunStatus = Literal["completed", "partial", "skipped", "denied", "failed"]


@dataclass(frozen=True)
class SourceRecord:
    id: int
    source_type: SourceType
    canonical_uri: str
    display_name: str
    source_path: str | None
    source_url: str | None
    redacted_source_url: str | None
    sync_enabled: bool
    last_sync_status: str | None
    last_sync_started_at: str | None
    last_sync_completed_at: str | None
    last_error_code: str | None
    document_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncSourceRequest:
    source_id: int | None = None
    source_uri: str | None = None
    mode: SyncMode = "dry_run"
    max_documents: int | None = None
    max_pages: int | None = None
    stale_policy: StalePolicy = "report"
    force: bool = False
```

Stable reason codes to add where missing:

```python
SOURCE_NOT_FOUND = "source_not_found"
SOURCE_SELECTOR_INVALID = "source_selector_invalid"
SOURCE_SCOPE_DENIED = "source_scope_denied"
SOURCE_SYNC_DISABLED = "source_sync_disabled"
SOURCE_SYNC_UNSUPPORTED_TYPE = "source_sync_unsupported_type"
SOURCE_SYNC_LIMIT_EXCEEDED = "source_sync_limit_exceeded"
URL_QUERY_NOT_PERSISTED = "url_query_not_persisted"
STALE_REPORTED = "stale_reported"
TOMBSTONED = "tombstoned"
```

## Test Command Conventions

Use the worktree virtual environment:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q
```

If `.venv` is absent or cannot import project dependencies, stop and report that environment problem. Do not switch to global Python.

---

### Task 1: Settings, Models, And Source-Sync Status

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/settings.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/models.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/mcp_module.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`

- [ ] **Step 1: Write failing settings and status tests**

Add to `tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py`:

```python
def test_from_mapping_uses_safe_source_sync_defaults() -> None:
    settings = DocsSettings.from_mapping({})

    assert settings.enable_source_sync is True  # nosec B101
    assert settings.max_sync_documents == 500  # nosec B101
    assert settings.max_sync_pages == 25  # nosec B101
    assert settings.max_sync_run_items == 500  # nosec B101
    assert settings.default_stale_policy == "report"  # nosec B101
    assert settings.sitemap_sync_enabled is False  # nosec B101
    assert settings.persist_url_query_strings is False  # nosec B101


def test_from_mapping_parses_source_sync_values() -> None:
    settings = DocsSettings.from_mapping(
        {
            "enable_source_sync": "false",
            "max_sync_documents": "9",
            "max_sync_pages": "7",
            "max_sync_run_items": "5",
            "default_stale_policy": "tombstone",
            "sitemap_sync_enabled": "true",
            "persist_url_query_strings": "true",
        }
    )

    assert settings.enable_source_sync is False  # nosec B101
    assert settings.max_sync_documents == 9  # nosec B101
    assert settings.max_sync_pages == 7  # nosec B101
    assert settings.max_sync_run_items == 5  # nosec B101
    assert settings.default_stale_policy == "tombstone"  # nosec B101
    assert settings.sitemap_sync_enabled is True  # nosec B101
    assert settings.persist_url_query_strings is True  # nosec B101


@pytest.mark.parametrize("value", ["", "delete", "hide"])
def test_from_mapping_rejects_unknown_default_stale_policy(value: str) -> None:
    with pytest.raises(ValueError, match="default_stale_policy"):
        DocsSettings.from_mapping({"default_stale_policy": value})
```

Add to `test_provider_status_reports_web_acquisition_disabled` in `test_docs_mcp_provider.py`:

```python
assert status["source_sync"]["enabled"] is True  # nosec B101
assert status["source_sync"]["default_stale_policy"] == "report"  # nosec B101
assert status["source_sync"]["max_sync_documents"] == 500  # nosec B101
assert status["source_sync"]["sitemap_sync_enabled"] is False  # nosec B101
```

- [ ] **Step 2: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py::test_provider_status_reports_web_acquisition_disabled \
  -q
```

Expected: FAIL with missing `DocsSettings` attributes or missing `source_sync` status.

- [ ] **Step 3: Implement settings and status**

In `settings.py`, add:

```python
StalePolicy = Literal["report", "tombstone"]


def _coerce_stale_policy(value: object, field_name: str) -> StalePolicy:
    text = str(value or "report").strip().lower()
    if text not in {"report", "tombstone"}:
        raise ValueError(f"{field_name} must be report or tombstone")
    return cast(StalePolicy, text)
```

Extend `DocsSettings` and `from_mapping()`:

```python
enable_source_sync: bool = True
max_sync_documents: int = 500
max_sync_pages: int = 25
max_sync_run_items: int = 500
default_stale_policy: StalePolicy = "report"
sitemap_sync_enabled: bool = False
persist_url_query_strings: bool = False
```

In `mcp_module.py`, add:

```python
def _source_sync_status(settings: DocsSettings) -> dict[str, Any]:
    return {
        "enabled": settings.enable_source_sync,
        "max_sync_documents": settings.max_sync_documents,
        "max_sync_pages": settings.max_sync_pages,
        "max_sync_run_items": settings.max_sync_run_items,
        "default_stale_policy": settings.default_stale_policy,
        "sitemap_sync_enabled": settings.sitemap_sync_enabled,
        "persist_url_query_strings": settings.persist_url_query_strings,
    }
```

Then set `status["source_sync"] = _source_sync_status(self.settings)` in `docs.status`.

- [ ] **Step 4: Run tests to verify green state**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/settings.py apps/mcp-unified/src/mcp_unified/docs/models.py apps/mcp-unified/src/mcp_unified/docs/mcp_module.py tldw_Server_API/tests/MCP_unified/docs/test_docs_settings.py tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py
git commit -m "feat: add docs source sync settings"
```

### Task 2: Source Registry Schema And Store Helpers

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/store/schema.sql`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/store/sqlite.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/models.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_sources_store.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py`

- [ ] **Step 1: Write failing store tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_sources_store.py`:

```python
from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.store.sqlite import DocsCatalogStore

pytestmark = pytest.mark.unit


def test_migrate_adds_source_tables_and_document_lifecycle_columns(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()

    with closing(store.connect()) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')")
        }
        document_columns = {row["name"] for row in conn.execute("PRAGMA table_info(docs_documents)")}

    assert {"docs_sources", "docs_source_documents", "docs_sync_runs"}.issubset(tables)  # nosec B101
    assert "lifecycle_status" in document_columns  # nosec B101
    assert "preserve_on_source_tombstone" in document_columns  # nosec B101


def test_store_upserts_and_lists_sources_by_scope(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope_a = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    scope_b = AccessScope(owner_scope="owner-b", profile_scope="profile-a")

    source_id = store.upsert_source(
        scope=scope_a,
        source_type="local_file",
        canonical_uri="file:///docs/a.md",
        display_name="a.md",
        source_path="/docs/a.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={"default_keywords": ["local"]},
    )
    second_id = store.upsert_source(
        scope=scope_a,
        source_type="local_file",
        canonical_uri="file:///docs/a.md",
        display_name="a.md updated",
        source_path="/docs/a.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={"default_keywords": ["local"]},
    )

    assert second_id == source_id  # nosec B101
    assert [source["canonical_uri"] for source in store.list_sources(scope=scope_a)] == ["file:///docs/a.md"]  # nosec B101
    assert store.list_sources(scope=scope_b) == []  # nosec B101


def test_upsert_document_for_sync_merges_existing_organization(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    document_id = store.upsert_document(
        scope=scope,
        title="Guide",
        document_type="text",
        canonical_uri="file:///guide.txt",
        source_path="/guide.txt",
        source_url=None,
        text="old sqlite body",
        sections=[],
        chunks=[{"text": "old sqlite body", "citation": "guide.txt"}],
        keywords=("manual",),
        collection_names=("Manual",),
        metadata={"importer": "local"},
    )

    updated_id = store.upsert_document_for_sync(
        scope=scope,
        title="Guide",
        document_type="text",
        canonical_uri="file:///guide.txt",
        source_path="/guide.txt",
        source_url=None,
        text="new sqlite body",
        sections=[],
        chunks=[{"text": "new sqlite body", "citation": "guide.txt"}],
        source_default_keywords=("source-default",),
        source_default_collections=("Source Defaults",),
        metadata={"importer": "local", "sync": True},
    )

    assert updated_id == document_id  # nosec B101
    assert store.search_chunks(scope, "new", limit=10)[0]["title"] == "Guide"  # nosec B101
    assert {item["keyword"] for item in store.list_keywords(scope)} == {"manual", "source-default"}  # nosec B101
    assert {item["name"] for item in store.list_collections(scope)} == {"Manual", "Source Defaults"}  # nosec B101
```

- [ ] **Step 2: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_sources_store.py -q
```

Expected: FAIL with missing `upsert_source`, `list_sources`, or `upsert_document_for_sync`.

- [ ] **Step 3: Implement schema additions**

In `schema.sql`, add `lifecycle_status` and `preserve_on_source_tombstone` to fresh `docs_documents`, plus:

```sql
CREATE TABLE IF NOT EXISTS docs_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_scope TEXT NOT NULL DEFAULT '',
    profile_scope TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL,
    canonical_uri TEXT NOT NULL,
    display_name TEXT NOT NULL,
    source_path TEXT,
    source_url TEXT,
    redacted_source_url TEXT,
    policy_profile TEXT,
    sync_enabled INTEGER NOT NULL DEFAULT 1,
    last_sync_status TEXT,
    last_sync_started_at TEXT,
    last_sync_completed_at TEXT,
    last_error_code TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (owner_scope, profile_scope, canonical_uri)
);

CREATE TABLE IF NOT EXISTS docs_source_documents (
    source_id INTEGER NOT NULL REFERENCES docs_sources(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
    source_item_uri TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_seen_at TEXT,
    last_hash TEXT,
    last_error_code TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source_id, source_item_uri)
);

CREATE TABLE IF NOT EXISTS docs_sync_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_scope TEXT NOT NULL DEFAULT '',
    profile_scope TEXT NOT NULL DEFAULT '',
    source_id INTEGER NOT NULL REFERENCES docs_sources(id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    requested_limits_json TEXT NOT NULL DEFAULT '{}',
    counts_json TEXT NOT NULL DEFAULT '{}',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
```

- [ ] **Step 4: Implement store helpers**

Add these public methods to `DocsCatalogStore` with the listed signatures:

- `upsert_source(self, *, scope: AccessScope, source_type: str, canonical_uri: str, display_name: str, source_path: str | None, source_url: str | None, redacted_source_url: str | None, policy_profile: str | None, sync_enabled: bool, metadata: Mapping[str, Any]) -> int`
- `get_source(self, *, scope: AccessScope, source_id: int | None = None, canonical_uri: str | None = None) -> dict[str, Any] | None`
- `list_sources(self, *, scope: AccessScope, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]`
- `link_source_document(self, *, scope: AccessScope, source_id: int, document_id: int, source_item_uri: str, status: str, last_hash: str | None, metadata: Mapping[str, Any]) -> None`
- `source_document_links(self, *, scope: AccessScope, source_id: int) -> list[dict[str, Any]]`
- `record_sync_run(self, *, scope: AccessScope, source_id: int, mode: str, status: str, requested_limits: Mapping[str, Any], counts: Mapping[str, int], warnings: list[str], error_code: str | None, metadata: Mapping[str, Any]) -> int`
- `upsert_document_for_sync(self, *, scope: AccessScope, title: str, document_type: str, canonical_uri: str, source_path: str | None, source_url: str | None, text: str, sections: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]], source_default_keywords: Iterable[str], source_default_collections: Iterable[str], metadata: Mapping[str, Any]) -> int`

Add migration guards in `migrate()`:

```python
self._ensure_document_lifecycle_columns(conn)
self._ensure_source_tables(conn)
```

Keep `upsert_document()` replacement-oriented; `upsert_document_for_sync()` must read existing collection/keyword membership in the same transaction, union it with source defaults, then call the existing row/chunk replacement helpers.

- [ ] **Step 5: Run tests to verify green state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_sources_store.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/store/schema.sql apps/mcp-unified/src/mcp_unified/docs/store/sqlite.py apps/mcp-unified/src/mcp_unified/docs/models.py tldw_Server_API/tests/MCP_unified/docs/test_docs_sources_store.py tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py
git commit -m "feat: add docs source registry store"
```

### Task 3: Populate Sources From Local Imports

**Files:**

- Create: `apps/mcp-unified/src/mcp_unified/docs/source_utils.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/importers/local.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py`

- [ ] **Step 1: Write failing local source population tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py -q
```

Expected: FAIL because imports do not create sources or source links.

- [ ] **Step 3: Implement local source helpers**

Create `source_utils.py`:

```python
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def file_uri_for_path(path: Path, *, directory: bool = False) -> str:
    uri = path.expanduser().resolve().as_uri()
    return f"{uri}/" if directory and not uri.endswith("/") else uri


def source_defaults_metadata(*, keywords: tuple[str, ...], collection_names: tuple[str, ...]) -> dict[str, list[str]]:
    return {
        "default_keywords": list(keywords),
        "default_collections": list(collection_names),
    }


def redacted_url_for_display(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", "", ""))


def url_has_query(url: str) -> bool:
    return bool(urlsplit(url).query)
```

- [ ] **Step 4: Link imports to sources**

In `DocsImportService.import_path()`:

```python
source_type = "local_file" if target.is_file() else "local_directory"
source_id = self.store.upsert_source(
    scope=scope,
    source_type=source_type,
    canonical_uri=file_uri_for_path(target, directory=target.is_dir()),
    display_name=target.name or str(target),
    source_path=str(target),
    source_url=None,
    redacted_source_url=None,
    policy_profile=self.settings.web_source_profile,
    sync_enabled=True,
    metadata=source_defaults_metadata(keywords=keyword_tuple, collection_names=collection_tuple),
)
```

After each document upsert:

```python
self.store.link_source_document(
    scope=scope,
    source_id=source_id,
    document_id=document_id,
    source_item_uri=file_uri_for_path(file_path),
    status="active",
    last_hash=None,
    metadata={"importer": "local"},
)
```

Return `"source": self.store.get_source(scope=scope, source_id=source_id)` in the import result.

- [ ] **Step 5: Run tests to verify green state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/source_utils.py apps/mcp-unified/src/mcp_unified/docs/importers/local.py tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py
git commit -m "feat: track local docs import sources"
```

### Task 4: Populate URL Sources With Query Safeguards

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/acquisition/service.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/source_utils.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py`

- [ ] **Step 1: Write failing URL source tests**

Add to `test_docs_source_population.py`:

```python
from mcp_unified.docs.acquisition.models import FetchResponse
from mcp_unified.docs.acquisition.service import DocsAcquisitionService
from tldw_Server_API.tests.MCP_unified.docs.helpers import FakeResolver, FakeTransport


def test_ingest_url_creates_url_page_source_for_queryless_url(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    service = DocsAcquisitionService(
        settings=settings,
        store=store,
        resolver=FakeResolver({"example.com": ["93.184.216.34"]}),
        transport=FakeTransport([FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"sync body"])]),
    )

    result = service.ingest_url(scope=AccessScope(), url="https://example.com/page")

    sources = store.list_sources(scope=AccessScope())
    assert result["source"]["source_type"] == "url_page"  # nosec B101
    assert sources[0]["source_url"] == "https://example.com/page"  # nosec B101
    assert sources[0]["redacted_source_url"] == "https://example.com/page"  # nosec B101


def test_ingest_query_url_does_not_create_refreshable_source_by_default(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    service = DocsAcquisitionService(
        settings=settings,
        store=store,
        resolver=FakeResolver({"example.com": ["93.184.216.34"]}),
        transport=FakeTransport([FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"query body"])]),
    )

    result = service.ingest_url(scope=AccessScope(), url="https://example.com/page?token=secret")

    assert result["status"] == "created"  # nosec B101
    assert result["reason_code"] == "ok"  # nosec B101
    assert result["source"] is None  # nosec B101
    assert "url_query_not_persisted" in result["warnings"]  # nosec B101
    assert store.list_sources(scope=AccessScope()) == []  # nosec B101
```

Add a companion opt-in test:

```python
def test_ingest_query_url_creates_source_when_query_persistence_enabled(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
            "persist_url_query_strings": True,
        }
    )
    service = DocsAcquisitionService(
        settings=settings,
        store=store,
        resolver=FakeResolver({"example.com": ["93.184.216.34"]}),
        transport=FakeTransport([FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"query body"])]),
    )

    result = service.ingest_url(scope=AccessScope(), url="https://example.com/page?token=secret")

    sources = store.list_sources(scope=AccessScope())
    assert result["source"]["source_url"] == "https://example.com/page?token=secret"  # nosec B101
    assert sources[0]["redacted_source_url"] == "https://example.com/page"  # nosec B101
```

- [ ] **Step 2: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py \
  -q
```

Expected: FAIL because successful URL ingest does not create a source or warnings.

- [ ] **Step 3: Implement URL source creation**

In `DocsAcquisitionService.ingest_url()`, after `document_id` is known:

```python
warnings = list(parsed.warnings)
source: dict[str, Any] | None = None
can_persist_query_source = self.settings.persist_url_query_strings or not url_has_query(parsed.canonical_uri)
if can_persist_query_source:
    redacted_source_url = redacted_url_for_display(parsed.canonical_uri)
    source_id = self.store.upsert_source(
        scope=scope,
        source_type="url_page",
        canonical_uri=parsed.canonical_uri,
        display_name=parsed.title,
        source_path=None,
        source_url=parsed.canonical_uri,
        redacted_source_url=redacted_source_url,
        policy_profile=self.settings.web_source_profile,
        sync_enabled=True,
        metadata=source_defaults_metadata(
            keywords=tuple(keywords),
            collection_names=tuple(collection_names),
        ),
    )
    self.store.link_source_document(
        scope=scope,
        source_id=source_id,
        document_id=document_id,
        source_item_uri=parsed.canonical_uri,
        status="active",
        last_hash=new_hash,
        metadata={"importer": "url"},
    )
    source = self.store.get_source(scope=scope, source_id=source_id)
else:
    warnings.append("url_query_not_persisted")
```

Return `"source": source` and `"warnings": warnings`.

- [ ] **Step 4: Run tests to verify green state**

Run the command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/acquisition/service.py apps/mcp-unified/src/mcp_unified/docs/source_utils.py tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_service.py tldw_Server_API/tests/MCP_unified/docs/test_docs_source_population.py
git commit -m "feat: track approved url docs sources"
```

### Task 5: Source Listing And `docs.sync_source` Tool Contract

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/mcp_module.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/sync.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`

- [ ] **Step 1: Write failing provider contract tests**

Create the first tests in `test_docs_source_sync.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs.mcp_module import DocsMCPToolProvider
from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore

pytestmark = pytest.mark.unit


def test_provider_advertises_sync_source_when_enabled(tmp_path: Path) -> None:
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(tmp_path,)))

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}

    assert "docs.sync_source" in tools  # nosec B101
    assert tools["docs.sync_source"]["metadata"]["category"] == "ingestion"  # nosec B101
    assert tools["docs.sync_source"]["metadata"]["readOnlyHint"] is False  # nosec B101


def test_provider_list_sources_returns_real_sources(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    store.upsert_source(
        scope=scope,
        source_type="local_file",
        canonical_uri="file:///docs/guide.md",
        display_name="guide.md",
        source_path="/docs/guide.md",
        source_url=None,
        redacted_source_url=None,
        policy_profile="locked_down",
        sync_enabled=True,
        metadata={},
    )
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db"), store=store)

    result = provider.execute("docs.list", {"kind": "sources"}, scope=scope)

    assert result["sources"][0]["canonical_uri"] == "file:///docs/guide.md"  # nosec B101
    assert result["warnings"] == []  # nosec B101
```

- [ ] **Step 2: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py::test_provider_advertises_sync_source_when_enabled \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py::test_provider_list_sources_returns_real_sources \
  -q
```

Expected: FAIL because the tool is absent and source listing returns the Stage 1 warning.

- [ ] **Step 3: Add provider wiring**

In `mcp_module.py`, instantiate:

```python
self.source_sync = DocsSourceSyncService(settings=settings, store=self.store)
```

Add a `docs.sync_source` tool when `settings.enable_source_sync` is true:

```python
_tool(
    "docs.sync_source",
    "Refresh one existing docs source with bounded dry-run or apply semantics.",
    {
        "source_id": {"type": "integer"},
        "source_uri": {"type": "string"},
        "mode": {"type": "string"},
        "max_documents": {"type": "integer"},
        "max_pages": {"type": "integer"},
        "stale_policy": {"type": "string"},
        "force": {"type": "boolean"},
    },
    [],
    "ingestion",
)
```

Change `docs.list(kind="sources")` to:

```python
return {"sources": self.store.list_sources(scope=scope, limit=limit, offset=offset), "warnings": []}
```

For `docs.sync_source`, call `self.source_sync.sync_source(scope=scope, request=request)`. If sync is disabled, return `{"status": "denied", "reason_code": "source_sync_disabled"}` for stale direct calls.

- [ ] **Step 4: Run tests to verify green state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py \
  -q
```

Expected: PASS for existing provider tests plus the new contract tests.

- [ ] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/mcp_module.py apps/mcp-unified/src/mcp_unified/docs/sync.py tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py
git commit -m "feat: expose docs source sync tool"
```

### Task 6: Local File And Directory Source Sync

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/sync.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/store/sqlite.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/importers/local.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py`

- [ ] **Step 1: Add failing local sync tests**

Add to `test_docs_source_sync.py`:

```python
def test_local_file_sync_dry_run_does_not_mutate_document_or_run_rows(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nOld sqlite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    imported = provider.execute("docs.import_path", {"path": str(guide)}, scope=scope)
    source_id = imported["source"]["id"]
    guide.write_text("# Guide\n\nNew sqlite content.\n", encoding="utf-8")

    result = provider.execute("docs.sync_source", {"source_id": source_id, "mode": "dry_run"}, scope=scope)
    search = provider.execute("docs.search", {"query": "New"}, scope=scope)
    status = provider.store.status()

    assert result["mode"] == "dry_run"  # nosec B101
    assert result["counts"]["updated"] == 1  # nosec B101
    assert search["results"] == []  # nosec B101
    assert status["counts"]["sync_runs"] == 0  # nosec B101


def test_local_file_sync_apply_updates_content_and_preserves_user_metadata(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nOld sqlite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    imported = provider.execute("docs.import_path", {"path": str(guide), "keywords": ["source"], "collections": ["Source"]}, scope=scope)
    document_id = imported["documents"][0]["id"]
    source_id = imported["source"]["id"]
    provider.execute("docs.keywords.apply", {"document_id": document_id, "keywords": ["manual"]}, scope=scope)
    provider.execute("docs.collections.set_membership", {"collection": "Manual", "document_id": document_id, "action": "add"}, scope=scope)
    guide.write_text("# Guide\n\nNew sqlite content.\n", encoding="utf-8")

    result = provider.execute("docs.sync_source", {"source_id": source_id, "mode": "apply"}, scope=scope)
    search = provider.execute("docs.search", {"query": "New", "filters": {"keywords": ["manual"], "collection": "Manual"}}, scope=scope)

    assert result["counts"]["updated"] == 1  # nosec B101
    assert search["results"][0]["document_id"] == document_id  # nosec B101
```

Add directory missing/report/tombstone tests:

```python
def test_local_directory_sync_report_missing_does_not_hide_document(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nSQLite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(root)}, scope=scope)
    source_id = imported["source"]["id"]
    guide.unlink()

    result = provider.execute("docs.sync_source", {"source_id": source_id, "mode": "apply", "stale_policy": "report"}, scope=scope)
    search = provider.execute("docs.search", {"query": "SQLite"}, scope=scope)

    assert result["counts"]["missing"] == 1  # nosec B101
    assert result["counts"]["tombstoned"] == 0  # nosec B101
    assert search["results"]  # nosec B101


def test_local_directory_sync_tombstone_hides_document_from_default_search(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    guide = root / "guide.md"
    guide.write_text("# Guide\n\nSQLite content.\n", encoding="utf-8")
    provider = DocsMCPToolProvider(settings=DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root,)))
    scope = AccessScope()
    imported = provider.execute("docs.import_path", {"path": str(root)}, scope=scope)
    source_id = imported["source"]["id"]
    guide.unlink()

    result = provider.execute("docs.sync_source", {"source_id": source_id, "mode": "apply", "stale_policy": "tombstone"}, scope=scope)
    search = provider.execute("docs.search", {"query": "SQLite"}, scope=scope)

    assert result["counts"]["tombstoned"] == 1  # nosec B101
    assert search["results"] == []  # nosec B101
```

- [ ] **Step 2: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py -q
```

Expected: FAIL because local sync logic is not implemented.

- [ ] **Step 3: Implement local sync service**

In `sync.py`, implement:

```python
class DocsSourceSyncService:
    def __init__(self, *, settings: DocsSettings, store: DocsCatalogStore, resolver: object | None = None, transport: object | None = None) -> None:
        self.settings = settings
        self.store = store
        self.importer = DocsImportService(settings=settings, store=store)
        self.acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)

    def sync_source(self, *, scope: AccessScope, request: SyncSourceRequest) -> dict[str, Any]:
        source = self._select_source(scope=scope, request=request)
        if source["source_type"] == "local_file":
            return self._sync_local_file(scope=scope, source=source, request=request)
        if source["source_type"] == "local_directory":
            return self._sync_local_directory(scope=scope, source=source, request=request)
        if source["source_type"] == "url_page":
            return self._sync_url_page(scope=scope, source=source, request=request)
        return self._denied(source=source, request=request, reason_code="source_sync_unsupported_type")
```

Rules to enforce in local methods:

- `dry_run` parses and hashes files but does not call any store mutation method.
- `apply` calls `upsert_document_for_sync()` only for changed or forced content.
- `stale_policy="report"` returns missing items without updating link status.
- `stale_policy="tombstone"` in apply mode calls `store.tombstone_source_item(scope=scope, source_id=source_id, source_item_uri=item_uri)`.
- `max_documents` is enforced before parsing all directory candidates.
- Every path goes through the existing trusted-root and symlink checks.

- [ ] **Step 4: Make default search lifecycle-aware**

In `search_chunks()`, `count_search_chunks()`, and `list_documents()`, add:

```sql
AND d.lifecycle_status = 'active'
```

Tombstoning should set `docs_documents.lifecycle_status='tombstoned'` only when all source links for the document are tombstoned and `preserve_on_source_tombstone=0`.

- [ ] **Step 5: Run tests to verify green state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/sync.py apps/mcp-unified/src/mcp_unified/docs/store/sqlite.py apps/mcp-unified/src/mcp_unified/docs/importers/local.py tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py
git commit -m "feat: sync local docs sources"
```

### Task 7: URL Page Source Sync

**Files:**

- Modify: `apps/mcp-unified/src/mcp_unified/docs/sync.py`
- Modify: `apps/mcp-unified/src/mcp_unified/docs/acquisition/service.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py`

- [ ] **Step 1: Add failing URL sync tests**

Add to `test_docs_source_sync.py`:

```python
def test_url_page_sync_apply_refreshes_existing_source_with_fake_transport(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings.from_mapping(
        {
            "db_path": str(tmp_path / "docs.db"),
            "enable_web_acquisition": True,
            "web_source_profile": "online_capable",
            "allow_arbitrary_public_domains": True,
        }
    )
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport(
        [
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"old sqlite body"]),
            FetchResponse(status_code=200, headers={"content-type": "text/plain"}, body_chunks=[b"new sqlite body"]),
        ]
    )
    acquisition = DocsAcquisitionService(settings=settings, store=store, resolver=resolver, transport=transport)
    scope = AccessScope()
    ingested = acquisition.ingest_url(scope=scope, url="https://example.com/page")
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=ingested["source"]["id"], mode="apply"))
    search = store.search_chunks(scope=scope, query="new", limit=10)

    assert result["counts"]["updated"] == 1  # nosec B101
    assert search[0]["title"]  # nosec B101
```

Add no-fetch-before-approval:

```python
def test_url_page_sync_does_not_fetch_when_policy_requires_approval(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope()
    source_id = store.upsert_source(
        scope=scope,
        source_type="url_page",
        canonical_uri="https://example.com/page",
        display_name="page",
        source_path=None,
        source_url="https://example.com/page",
        redacted_source_url="https://example.com/page",
        policy_profile="local_first",
        sync_enabled=True,
        metadata={},
    )
    settings = DocsSettings.from_mapping({"db_path": str(tmp_path / "docs.db"), "enable_web_acquisition": True, "web_source_profile": "local_first"})
    resolver = FakeResolver({"example.com": ["93.184.216.34"]})
    transport = FakeTransport([FetchResponse(status_code=200, body_chunks=[b"never"])])
    service = DocsSourceSyncService(settings=settings, store=store, resolver=resolver, transport=transport)

    result = service.sync_source(scope=scope, request=SyncSourceRequest(source_id=source_id, mode="apply"))

    assert result["status"] == "denied"  # nosec B101
    assert result["reason_code"] == "approval_required"  # nosec B101
    assert resolver.calls == []  # nosec B101
    assert transport.calls == []  # nosec B101
```

- [ ] **Step 2: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py -q
```

Expected: FAIL because URL sync is not implemented.

- [ ] **Step 3: Implement URL page sync**

In `_sync_url_page()`:

- Require `settings.enable_web_acquisition=true`.
- Fetch exactly `source["source_url"]`.
- Reuse `DocsAcquisitionService` with the same resolver/transport seams.
- Return `approval_required` or `source_policy_denied` without fetch when policy blocks.
- In dry-run, compute item status from old vs new hash and return without store mutation.
- In apply, call `upsert_document_for_sync()` and `link_source_document()`.
- Record a sync run only in apply mode.
- Use `redacted_source_url` in response/source metadata.

- [ ] **Step 4: Run tests to verify green state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/mcp-unified/src/mcp_unified/docs/sync.py apps/mcp-unified/src/mcp_unified/docs/acquisition/service.py tldw_Server_API/tests/MCP_unified/docs/test_docs_source_sync.py tldw_Server_API/tests/MCP_unified/docs/test_docs_acquisition_fetcher.py
git commit -m "feat: sync url page docs sources"
```

### Task 8: Host Exposure, Boundary Tests, And Verification

**Files:**

- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`
- Modify: `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`
- Modify: `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`
- Modify: `tldw_Server_API/tests/MCP_unified/docs/test_docs_package_metadata.py`

- [ ] **Step 1: Add host and boundary tests**

Add to `test_docs_module_shim.py`:

```python
async def test_docs_module_exposes_sync_source_without_media_or_rag(tmp_path: Path) -> None:
    module = DocsModule(ModuleConfig(name="docs", settings={"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)]}))
    await module.on_initialize()

    tools = {tool["name"]: tool for tool in await module.get_tools()}

    assert "docs.sync_source" in tools  # nosec B101
    assert tools["docs.sync_source"]["metadata"]["category"] == "ingestion"  # nosec B101
```

Extend `test_docs_package_does_not_import_optional_web_acquisition_dependencies()` with no new exceptions. Keep `trafilatura` and `bs4` out of import-time dependencies.

- [ ] **Step 2: Run focused tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_package_metadata.py \
  -q
```

Expected: PASS after the host shim recognizes `docs.sync_source` arguments.

- [ ] **Step 3: Run full docs test slice**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs -q
```

Expected: PASS.

- [ ] **Step 4: Run import-boundary and Bandit security checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -q
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r apps/mcp-unified/src/mcp_unified/docs tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py \
  -f json -o /tmp/bandit_mcp_docs_stage4a.json
```

Expected: pytest PASS. Bandit exits 0 or reports only accepted existing findings outside changed code; fix new findings in touched code before continuing.

- [ ] **Step 5: Commit final verification adjustments**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py tldw_Server_API/tests/MCP_unified/docs/test_docs_package_metadata.py
git commit -m "test: verify docs source sync host boundary"
```

## Self-Review Checklist

- [ ] Spec coverage: tasks cover source registry, source population, source listing, local sync, URL page sync, dry-run no mutation, metadata merge, query-persistence safeguards, host exposure, no Media/RAG bridge, and boundary tests.
- [ ] Optional sitemap isolation: no Stage 4A.1 task creates `docs.sources.register` or `url_sitemap` sync.
- [ ] Placeholder scan: run `rg -n "T[B]D|TO[D]O|fill[ ]in|appropria[t]e|implement[ ]later|similar[ ]to" Docs/superpowers/plans/2026-07-02-standalone-mcp-docs-stage4a-source-sync-implementation-plan.md` and fix any match.
- [ ] Type/signature consistency: `SyncSourceRequest`, `DocsSourceSyncService.sync_source`, `upsert_document_for_sync`, and source store helper names match across tasks.
- [ ] Verification commands: focused pytest, docs pytest slice, import-boundary pytest, and Bandit command are included.

## Execution Handoff

Plan complete when this file passes the self-review checklist and `git diff --check`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, with checkpoints after each task.
