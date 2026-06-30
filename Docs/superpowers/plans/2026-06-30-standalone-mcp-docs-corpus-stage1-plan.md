# Standalone MCP Docs Corpus Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage 1 standalone-first MCP document corpus: local file ingestion, SQLite/FTS5 storage, scoped retrieval, bounded context packs, collection/keyword metadata, Context7-compatible read aliases, and the current `tldw_server` MCP module shim.

**Architecture:** Create a new top-level `mcp_unified.docs` package for the runtime-neutral corpus and retrieval code. Keep `tldw_Server_API` integration in one thin module under `tldw_Server_API.app.core.MCP_unified.modules.implementations` so the existing loader can register the tools while the docs core remains portable.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`, stdlib `html.parser`, `pathlib`, dataclasses, existing MCP `BaseModule` only in the shim, pytest, Bandit.

---

## Scope

This plan implements only Stage 1 from `Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md`.

Included:

- Top-level `mcp_unified.docs` package boundary.
- SQLite schema and migrations using FTS5.
- Local Markdown, MDX, text, and static HTML import.
- Path-scope enforcement for configured trusted roots.
- `docs.search`, `docs.get`, `docs.context`, `docs.resolve`, `docs.list`, `docs.status`.
- `docs.import_path`.
- `docs.collections.list`, `docs.collections.create`, `docs.collections.update`, `docs.collections.set_membership`.
- `docs.keywords.list`, `docs.keywords.apply`.
- Context7-compatible read aliases: `resolve-library-id`, `get-library-docs`.
- Store-level `owner_scope` and `profile_scope` enforcement.
- Import-boundary tests proving the docs core does not import `tldw_Server_API`.
- Baseline import tests proving web acquisition dependencies are not required.

Deferred:

- `docs.ingest_url`.
- URL source policy and egress guards.
- Existing `tldw_server` scraping pipeline reuse.
- Playwright/browser extraction.
- Embeddings and reranking.
- Bounded crawl, sitemap sync, and `docs.sync_source`.
- Media DB/RAG host bridges.

## File Structure

Create the runtime-neutral docs package:

- Create: `mcp_unified/__init__.py`
  Public package marker.
- Create: `mcp_unified/docs/__init__.py`
  Export `DocsCatalogStore`, `DocsImportService`, `DocsRetrievalService`, `DocsMCPToolProvider`, and model classes.
- Create: `mcp_unified/docs/errors.py`
  Stable `DocsError` with machine-readable reason codes.
- Create: `mcp_unified/docs/models.py`
  Dataclasses for access scope, records, filters, results, and context packs.
- Create: `mcp_unified/docs/settings.py`
  `DocsSettings` and path root normalization.
- Create: `mcp_unified/docs/store/__init__.py`
- Create: `mcp_unified/docs/store/schema.sql`
  SQLite schema and FTS5 definitions.
- Create: `mcp_unified/docs/store/sqlite.py`
  Store, migration, transaction, scope filtering, FTS query, and mutation helpers.
- Create: `mcp_unified/docs/importers/__init__.py`
- Create: `mcp_unified/docs/importers/base.py`
  Parsed document dataclasses and chunking helpers.
- Create: `mcp_unified/docs/importers/markdown.py`
  Markdown/MDX/text parser.
- Create: `mcp_unified/docs/importers/html.py`
  Static HTML parser using stdlib `html.parser`.
- Create: `mcp_unified/docs/importers/local.py`
  Trusted-root path import orchestration.
- Create: `mcp_unified/docs/retrieval/__init__.py`
- Create: `mcp_unified/docs/retrieval/search.py`
  Search, get, list, collection, keyword service methods.
- Create: `mcp_unified/docs/retrieval/context.py`
  Bounded context pack builder.
- Create: `mcp_unified/docs/retrieval/aliases.py`
  General alias and Context7-compatible resolution.
- Create: `mcp_unified/docs/mcp_module.py`
  Runtime-neutral MCP tool provider with tool definitions and dispatch.

Integrate with the existing `tldw_server` MCP loader:

- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`
  Thin `BaseModule` adapter delegating to `mcp_unified.docs`.
- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml`
  Add a `docs` module entry with web acquisition disabled.
- Modify: `pyproject.toml`
  Include `mcp_unified` and `mcp_unified.*` in package discovery and include `mcp_unified/docs/store/schema.sql` as package data.

Add tests:

- Create: `tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py`
- Create: `tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py`
- Create: `tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py`
- Create: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`
- Create: `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`
- Create: `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py`

## Shared Contracts

Use these names consistently across tasks.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ScopeValue = str | None
DocumentType = Literal["markdown", "mdx", "text", "html", "other"]
RetrievalMode = Literal["metadata", "snippet", "section", "full", "chunk", "chunk_with_neighbors"]


@dataclass(frozen=True)
class AccessScope:
    owner_scope: ScopeValue = None
    profile_scope: ScopeValue = None


@dataclass(frozen=True)
class DocsSettings:
    db_path: Path
    trusted_roots: tuple[Path, ...] = ()
    max_import_file_bytes: int = 2_000_000
    default_scope: AccessScope = AccessScope()
    enable_web_acquisition: bool = False


@dataclass(frozen=True)
class SearchFilters:
    collection: str | None = None
    keywords: tuple[str, ...] = ()
    source_type: str | None = None
    document_type: str | None = None
    uri_prefix: str | None = None
    package: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class SearchRequest:
    query: str
    filters: SearchFilters = SearchFilters()
    limit: int = 10
    offset: int = 0
    snippet_length: int = 300


@dataclass(frozen=True)
class ContextRequest:
    query: str
    filters: SearchFilters = SearchFilters()
    max_chunks: int = 8
    max_documents: int = 4
    max_characters: int = 12_000
    citation_style: str = "inline"
```

Stable reason codes:

```python
class DocsReason:
    PATH_SCOPE_DENIED = "path_scope_denied"
    UNSUPPORTED_IMPORT_FORMAT = "unsupported_import_format"
    DOCUMENT_NOT_FOUND = "document_not_found"
    COLLECTION_NOT_FOUND = "collection_not_found"
    ALIAS_AMBIGUOUS = "alias_ambiguous"
    INDEX_UNAVAILABLE = "index_unavailable"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    STAGED_SOURCE_REQUIRES_ASSIGNMENT = "staged_source_requires_assignment"
    WEB_ACQUISITION_DISABLED = "web_acquisition_disabled"
```

## Task 1: Package Boundary, Settings, Errors, And Packaging

**Files:**

- Create: `mcp_unified/__init__.py`
- Create: `mcp_unified/docs/__init__.py`
- Create: `mcp_unified/docs/errors.py`
- Create: `mcp_unified/docs/models.py`
- Create: `mcp_unified/docs/settings.py`
- Modify: `pyproject.toml`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`

- [ ] **Step 1: Write the failing package boundary tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`:

```python
from __future__ import annotations

import ast
import importlib
from pathlib import Path


DOCS_PACKAGE_ROOT = Path("mcp_unified/docs")
FORBIDDEN_IMPORT_PREFIX = "tldw_Server_API"


def _import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_docs_package_imports_without_host_or_web_dependencies() -> None:
    module = importlib.import_module("mcp_unified.docs")

    assert hasattr(module, "DocsSettings")  # nosec B101
    assert hasattr(module, "AccessScope")  # nosec B101


def test_docs_core_does_not_import_tldw_server_modules() -> None:
    violations: list[tuple[str, str]] = []
    for path in DOCS_PACKAGE_ROOT.rglob("*.py"):
        for name in _import_names(path):
            if name == FORBIDDEN_IMPORT_PREFIX or name.startswith(f"{FORBIDDEN_IMPORT_PREFIX}."):
                violations.append((str(path), name))

    assert violations == []  # nosec B101


def test_docs_package_does_not_import_optional_web_acquisition_dependencies() -> None:
    forbidden = {"playwright", "trafilatura", "requests", "aiohttp"}
    violations: list[tuple[str, str]] = []
    for path in DOCS_PACKAGE_ROOT.rglob("*.py"):
        for name in _import_names(path):
            root = name.split(".", 1)[0]
            if root in forbidden:
                violations.append((str(path), name))

    assert violations == []  # nosec B101
```

- [ ] **Step 2: Run the boundary tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_unified'`.

- [ ] **Step 3: Add the package files**

Create `mcp_unified/__init__.py`:

```python
"""Runtime-neutral MCP utilities."""
```

Create `mcp_unified/docs/errors.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocsError(Exception):
    """Machine-readable docs corpus error."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}
```

Create `mcp_unified/docs/models.py` with the shared contract dataclasses from this plan plus record dataclasses:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ScopeValue = str | None
DocumentType = Literal["markdown", "mdx", "text", "html", "other"]
RetrievalMode = Literal["metadata", "snippet", "section", "full", "chunk", "chunk_with_neighbors"]


@dataclass(frozen=True)
class AccessScope:
    owner_scope: ScopeValue = None
    profile_scope: ScopeValue = None


@dataclass(frozen=True)
class SearchFilters:
    collection: str | None = None
    keywords: tuple[str, ...] = ()
    source_type: str | None = None
    document_type: str | None = None
    uri_prefix: str | None = None
    package: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class SearchRequest:
    query: str
    filters: SearchFilters = SearchFilters()
    limit: int = 10
    offset: int = 0
    snippet_length: int = 300


@dataclass(frozen=True)
class ContextRequest:
    query: str
    filters: SearchFilters = SearchFilters()
    max_chunks: int = 8
    max_documents: int = 4
    max_characters: int = 12_000
    citation_style: str = "inline"


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    title: str
    document_type: str
    canonical_uri: str
    source_path: str | None
    source_url: str | None
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    document_id: int
    chunk_id: int
    title: str
    snippet: str
    score: float
    uri: str
    citation: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

Create `mcp_unified/docs/settings.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import AccessScope


@dataclass(frozen=True)
class DocsSettings:
    db_path: Path
    trusted_roots: tuple[Path, ...] = ()
    max_import_file_bytes: int = 2_000_000
    default_scope: AccessScope = AccessScope()
    enable_web_acquisition: bool = False

    @classmethod
    def from_mapping(cls, values: dict) -> "DocsSettings":
        roots = tuple(Path(item).expanduser().resolve() for item in values.get("trusted_roots", []) or [])
        return cls(
            db_path=Path(values.get("db_path", "Databases/mcp_docs.db")).expanduser(),
            trusted_roots=roots,
            max_import_file_bytes=int(values.get("max_import_file_bytes", 2_000_000)),
            enable_web_acquisition=bool(values.get("enable_web_acquisition", False)),
        )
```

Create `mcp_unified/docs/__init__.py`:

```python
from .errors import DocsError
from .models import AccessScope, ContextRequest, DocumentRecord, SearchFilters, SearchRequest, SearchResult
from .settings import DocsSettings

__all__ = [
    "AccessScope",
    "ContextRequest",
    "DocsError",
    "DocsSettings",
    "DocumentRecord",
    "SearchFilters",
    "SearchRequest",
    "SearchResult",
]
```

Modify `pyproject.toml` package discovery:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["tldw_Server_API", "tldw_Server_API.*", "mcp_unified", "mcp_unified.*"]
```

Add package data:

```toml
[tool.setuptools.package-data]
mcp_unified = [
  "docs/store/schema.sql",
]
```

- [ ] **Step 4: Run the boundary tests and verify they pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add pyproject.toml mcp_unified tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py
git commit -m "feat: add standalone docs package boundary"
```

## Task 2: SQLite Schema, Migrations, And Scoped Store

**Files:**

- Create: `mcp_unified/docs/store/__init__.py`
- Create: `mcp_unified/docs/store/schema.sql`
- Create: `mcp_unified/docs/store/sqlite.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py`

- [ ] **Step 1: Write failing store tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py`:

```python
from __future__ import annotations

from pathlib import Path

from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.store.sqlite import DocsCatalogStore


def test_store_migrates_and_reports_status(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()

    status = store.status()

    assert status["schema_version"] == 1  # nosec B101
    assert status["fts_available"] is True  # nosec B101
    assert status["counts"]["documents"] == 0  # nosec B101


def test_document_without_collection_is_searchable(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    document_id = store.upsert_document(
        scope=scope,
        title="Install Guide",
        document_type="markdown",
        canonical_uri="file:///docs/install.md",
        source_path="/docs/install.md",
        source_url=None,
        text="Install the server with sqlite fts enabled.",
        sections=[{"heading": "Install", "level": 1, "start_char": 0, "end_char": 43}],
        chunks=[{"text": "Install the server with sqlite fts enabled.", "citation": "install.md:1"}],
        keywords=("setup",),
        collection_names=(),
        metadata={"source": "unit"},
    )

    results = store.search_chunks(scope=scope, query="sqlite", limit=10)

    assert document_id > 0  # nosec B101
    assert len(results) == 1  # nosec B101
    assert results[0]["title"] == "Install Guide"  # nosec B101


def test_store_enforces_owner_and_profile_scope(tmp_path: Path) -> None:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope_a = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    scope_b = AccessScope(owner_scope="owner-b", profile_scope="profile-a")

    store.upsert_document(
        scope=scope_a,
        title="Private Doc",
        document_type="text",
        canonical_uri="file:///private.txt",
        source_path="/private.txt",
        source_url=None,
        text="private sqlite material",
        sections=[],
        chunks=[{"text": "private sqlite material", "citation": "private.txt"}],
        keywords=(),
        collection_names=(),
        metadata={},
    )

    assert store.search_chunks(scope=scope_a, query="sqlite", limit=10)  # nosec B101
    assert store.search_chunks(scope=scope_b, query="sqlite", limit=10) == []  # nosec B101
```

- [ ] **Step 2: Run store tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `mcp_unified.docs.store`.

- [ ] **Step 3: Implement schema and store**

Create `mcp_unified/docs/store/schema.sql`:

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS docs_schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS docs_documents (
  id INTEGER PRIMARY KEY,
  owner_scope TEXT,
  profile_scope TEXT,
  title TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  document_type TEXT NOT NULL,
  language TEXT,
  canonical_uri TEXT NOT NULL,
  source_url TEXT,
  source_path TEXT,
  content_hash TEXT NOT NULL,
  raw_content_hash TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  indexed_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_documents_scope_uri
  ON docs_documents(owner_scope, profile_scope, canonical_uri);

CREATE TABLE IF NOT EXISTS docs_collections (
  id INTEGER PRIMARY KEY,
  owner_scope TEXT,
  profile_scope TEXT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  collection_type TEXT NOT NULL DEFAULT 'general',
  visibility TEXT NOT NULL DEFAULT 'private',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_collections_scope_name
  ON docs_collections(owner_scope, profile_scope, name);

CREATE TABLE IF NOT EXISTS docs_collection_members (
  collection_id INTEGER NOT NULL REFERENCES docs_collections(id) ON DELETE CASCADE,
  document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
  member_path TEXT,
  order_index INTEGER NOT NULL DEFAULT 0,
  role TEXT NOT NULL DEFAULT 'member',
  added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(collection_id, document_id)
);

CREATE TABLE IF NOT EXISTS docs_keywords (
  id INTEGER PRIMARY KEY,
  owner_scope TEXT,
  profile_scope TEXT,
  keyword TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_keywords_scope_keyword
  ON docs_keywords(owner_scope, profile_scope, keyword);

CREATE TABLE IF NOT EXISTS docs_document_keywords (
  document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
  keyword_id INTEGER NOT NULL REFERENCES docs_keywords(id) ON DELETE CASCADE,
  PRIMARY KEY(document_id, keyword_id)
);

CREATE TABLE IF NOT EXISTS docs_sections (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
  heading TEXT NOT NULL DEFAULT '',
  heading_path TEXT NOT NULL DEFAULT '',
  level INTEGER NOT NULL DEFAULT 1,
  order_index INTEGER NOT NULL DEFAULT 0,
  start_char INTEGER,
  end_char INTEGER,
  offset_precision TEXT NOT NULL DEFAULT 'unknown'
);

CREATE TABLE IF NOT EXISTS docs_chunks (
  id INTEGER PRIMARY KEY,
  document_id INTEGER NOT NULL REFERENCES docs_documents(id) ON DELETE CASCADE,
  section_id INTEGER REFERENCES docs_sections(id) ON DELETE SET NULL,
  chunk_text TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  token_estimate INTEGER NOT NULL DEFAULT 0,
  citation TEXT NOT NULL,
  order_index INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_chunks_fts USING fts5(
  title,
  chunk_text,
  citation,
  document_id UNINDEXED,
  chunk_id UNINDEXED,
  content='',
  tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS docs_aliases (
  id INTEGER PRIMARY KEY,
  owner_scope TEXT,
  profile_scope TEXT,
  alias TEXT NOT NULL,
  target_type TEXT NOT NULL,
  target_id INTEGER NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_aliases_scope_alias
  ON docs_aliases(owner_scope, profile_scope, alias);

INSERT OR IGNORE INTO docs_schema_migrations(version) VALUES (1);
```

Create `mcp_unified/docs/store/__init__.py`:

```python
from .sqlite import DocsCatalogStore

__all__ = ["DocsCatalogStore"]
```

Create `mcp_unified/docs/store/sqlite.py` with these public methods:

```python
from __future__ import annotations

import hashlib
import json
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from ..models import AccessScope


class DocsCatalogStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def migrate(self) -> None:
        schema = resources.files("mcp_unified.docs.store").joinpath("schema.sql").read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(schema)

    def status(self) -> dict[str, Any]:
        with self.connect() as conn:
            version = conn.execute("SELECT MAX(version) FROM docs_schema_migrations").fetchone()[0] or 0
            counts = {
                "documents": conn.execute("SELECT COUNT(*) FROM docs_documents").fetchone()[0],
                "chunks": conn.execute("SELECT COUNT(*) FROM docs_chunks").fetchone()[0],
                "collections": conn.execute("SELECT COUNT(*) FROM docs_collections").fetchone()[0],
                "keywords": conn.execute("SELECT COUNT(*) FROM docs_keywords").fetchone()[0],
            }
            fts_available = bool(conn.execute("SELECT name FROM sqlite_master WHERE name = 'docs_chunks_fts'").fetchone())
        return {"schema_version": int(version), "fts_available": fts_available, "counts": counts}

    def _scope_clause(self, scope: AccessScope, prefix: str = "") -> tuple[str, list[Any]]:
        col = f"{prefix}." if prefix else ""
        return (
            f"(({col}owner_scope IS ?) OR ({col}owner_scope = ?)) AND (({col}profile_scope IS ?) OR ({col}profile_scope = ?))",
            [scope.owner_scope, scope.owner_scope, scope.profile_scope, scope.profile_scope],
        )

    def upsert_document(
        self,
        *,
        scope: AccessScope,
        title: str,
        document_type: str,
        canonical_uri: str,
        source_path: str | None,
        source_url: str | None,
        text: str,
        sections: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        keywords: Iterable[str],
        collection_names: Iterable[str],
        metadata: dict[str, Any],
    ) -> int:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM docs_documents
                WHERE owner_scope IS ? AND profile_scope IS ? AND canonical_uri = ?
                """,
                (scope.owner_scope, scope.profile_scope, canonical_uri),
            ).fetchone()
            if row:
                document_id = int(row["id"])
                conn.execute(
                    """
                    UPDATE docs_documents
                    SET title = ?, document_type = ?, source_path = ?, source_url = ?,
                        content_hash = ?, raw_content_hash = ?, metadata_json = ?,
                        updated_at = CURRENT_TIMESTAMP, indexed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (title, document_type, source_path, source_url, content_hash, content_hash, json.dumps(metadata), document_id),
                )
                conn.execute("DELETE FROM docs_sections WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM docs_chunks WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM docs_chunks_fts WHERE document_id = ?", (str(document_id),))
                conn.execute("DELETE FROM docs_document_keywords WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM docs_collection_members WHERE document_id = ?", (document_id,))
            else:
                cur = conn.execute(
                    """
                    INSERT INTO docs_documents(
                      owner_scope, profile_scope, title, document_type, canonical_uri,
                      source_url, source_path, content_hash, raw_content_hash, metadata_json, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        scope.owner_scope,
                        scope.profile_scope,
                        title,
                        document_type,
                        canonical_uri,
                        source_url,
                        source_path,
                        content_hash,
                        content_hash,
                        json.dumps(metadata),
                    ),
                )
                document_id = int(cur.lastrowid)

            section_ids: list[int | None] = []
            for idx, section in enumerate(sections):
                cur = conn.execute(
                    """
                    INSERT INTO docs_sections(document_id, heading, heading_path, level, order_index, start_char, end_char, offset_precision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        section.get("heading") or "",
                        section.get("heading_path") or section.get("heading") or "",
                        int(section.get("level") or 1),
                        idx,
                        section.get("start_char"),
                        section.get("end_char"),
                        section.get("offset_precision") or "unknown",
                    ),
                )
                section_ids.append(int(cur.lastrowid))

            if not section_ids:
                section_ids.append(None)

            for idx, chunk in enumerate(chunks):
                chunk_text = str(chunk["text"])
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                section_id = section_ids[min(idx, len(section_ids) - 1)]
                cur = conn.execute(
                    """
                    INSERT INTO docs_chunks(document_id, section_id, chunk_text, content_hash, token_estimate, citation, order_index, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        section_id,
                        chunk_text,
                        chunk_hash,
                        max(1, len(chunk_text) // 4),
                        str(chunk.get("citation") or canonical_uri),
                        idx,
                        json.dumps(chunk.get("metadata") or {}),
                    ),
                )
                chunk_id = int(cur.lastrowid)
                conn.execute(
                    "INSERT INTO docs_chunks_fts(title, chunk_text, citation, document_id, chunk_id) VALUES (?, ?, ?, ?, ?)",
                    (title, chunk_text, str(chunk.get("citation") or canonical_uri), str(document_id), str(chunk_id)),
                )

            self._replace_keywords(conn, scope, document_id, keywords)
            self._replace_collections(conn, scope, document_id, collection_names)
            return document_id
```

Continue `mcp_unified/docs/store/sqlite.py` with concrete scoped helper methods:

```python
    def _keyword_id(self, conn: sqlite3.Connection, scope: AccessScope, keyword: str) -> int:
        normalized = keyword.strip().lower()
        cur = conn.execute(
            """
            INSERT INTO docs_keywords(owner_scope, profile_scope, keyword)
            VALUES (?, ?, ?)
            ON CONFLICT(owner_scope, profile_scope, keyword) DO UPDATE SET keyword = excluded.keyword
            RETURNING id
            """,
            (scope.owner_scope, scope.profile_scope, normalized),
        )
        return int(cur.fetchone()["id"])

    def _collection_id(self, conn: sqlite3.Connection, scope: AccessScope, name: str) -> int:
        normalized = name.strip()
        cur = conn.execute(
            """
            INSERT INTO docs_collections(owner_scope, profile_scope, name)
            VALUES (?, ?, ?)
            ON CONFLICT(owner_scope, profile_scope, name) DO UPDATE SET name = excluded.name
            RETURNING id
            """,
            (scope.owner_scope, scope.profile_scope, normalized),
        )
        return int(cur.fetchone()["id"])

    def _replace_keywords(
        self,
        conn: sqlite3.Connection,
        scope: AccessScope,
        document_id: int,
        keywords: Iterable[str],
    ) -> None:
        for keyword in sorted({item.strip().lower() for item in keywords if str(item).strip()}):
            keyword_id = self._keyword_id(conn, scope, keyword)
            conn.execute(
                "INSERT OR IGNORE INTO docs_document_keywords(document_id, keyword_id) VALUES (?, ?)",
                (document_id, keyword_id),
            )

    def _replace_collections(
        self,
        conn: sqlite3.Connection,
        scope: AccessScope,
        document_id: int,
        collection_names: Iterable[str],
    ) -> None:
        for order_index, name in enumerate([item.strip() for item in collection_names if str(item).strip()]):
            collection_id = self._collection_id(conn, scope, name)
            conn.execute(
                """
                INSERT OR REPLACE INTO docs_collection_members(collection_id, document_id, order_index)
                VALUES (?, ?, ?)
                """,
                (collection_id, document_id, order_index),
            )

    def search_chunks(
        self,
        *,
        scope: AccessScope,
        query: str,
        limit: int,
        offset: int = 0,
        filters: Any | None = None,
        snippet_length: int = 300,
    ) -> list[dict[str, Any]]:
        filters = filters or object()
        scope_sql, scope_params = self._scope_clause(scope, "d")
        joins = [
            "JOIN docs_documents d ON d.id = CAST(f.document_id AS INTEGER)",
            "JOIN docs_chunks c ON c.id = CAST(f.chunk_id AS INTEGER)",
        ]
        predicates = [scope_sql, "docs_chunks_fts MATCH ?"]
        predicate_params: list[Any] = [*scope_params, query]

        collection = getattr(filters, "collection", None)
        if collection:
            joins.append("JOIN docs_collection_members cm ON cm.document_id = d.id")
            joins.append("JOIN docs_collections col ON col.id = cm.collection_id")
            predicates.append("col.name = ?")
            predicate_params.append(str(collection))

        keywords = tuple(getattr(filters, "keywords", ()) or ())
        if keywords:
            joins.append("JOIN docs_document_keywords dk ON dk.document_id = d.id")
            joins.append("JOIN docs_keywords kw ON kw.id = dk.keyword_id")
            keyword_marks = ",".join("?" for _ in keywords)
            predicates.append(f"kw.keyword IN ({keyword_marks})")
            predicate_params.extend(str(item).lower() for item in keywords)

        document_type = getattr(filters, "document_type", None)
        if document_type:
            predicates.append("d.document_type = ?")
            predicate_params.append(str(document_type))

        uri_prefix = getattr(filters, "uri_prefix", None)
        if uri_prefix:
            predicates.append("d.canonical_uri LIKE ?")
            predicate_params.append(f"{uri_prefix}%")

        package = getattr(filters, "package", None)
        if package:
            predicates.append("json_extract(d.metadata_json, '$.package') = ?")
            predicate_params.append(str(package))

        version = getattr(filters, "version", None)
        if version:
            predicates.append("json_extract(d.metadata_json, '$.version') = ?")
            predicate_params.append(str(version))

        sql_params = [
            int(max(snippet_length, 50)),
            *predicate_params,
            int(max(limit, 1)),
            int(max(offset, 0)),
        ]
        sql = f"""
            SELECT
              d.id AS document_id,
              c.id AS chunk_id,
              d.title AS title,
              substr(c.chunk_text, 1, ?) AS snippet,
              bm25(docs_chunks_fts) AS score,
              d.canonical_uri AS uri,
              c.citation AS citation,
              d.metadata_json AS metadata_json
            FROM docs_chunks_fts f
            {' '.join(joins)}
            WHERE {' AND '.join(predicates)}
            ORDER BY score
            LIMIT ? OFFSET ?
        """
        with self.connect() as conn:
            rows = conn.execute(sql, sql_params).fetchall()
        return [
            {
                "document_id": int(row["document_id"]),
                "chunk_id": int(row["chunk_id"]),
                "title": row["title"],
                "snippet": row["snippet"],
                "score": float(row["score"] or 0.0),
                "uri": row["uri"],
                "citation": row["citation"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]

    def get_document(self, *, scope: AccessScope, target: str, mode: str = "snippet") -> dict[str, Any]:
        scope_sql, scope_params = self._scope_clause(scope, "d")
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT d.* FROM docs_documents d
                WHERE {scope_sql} AND (CAST(d.id AS TEXT) = ? OR d.canonical_uri = ?)
                """,
                [*scope_params, str(target), str(target)],
            ).fetchone()
            if row is None:
                from ..errors import DocsError

                raise DocsError("document_not_found", "Document not found in active scope.", {"target": target})
            chunks = conn.execute(
                "SELECT id, chunk_text, citation FROM docs_chunks WHERE document_id = ? ORDER BY order_index",
                (int(row["id"]),),
            ).fetchall()
        payload = {
            "id": int(row["id"]),
            "title": row["title"],
            "document_type": row["document_type"],
            "uri": row["canonical_uri"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
        }
        if mode in {"full", "section", "chunk", "chunk_with_neighbors", "snippet"}:
            payload["chunks"] = [dict(chunk) for chunk in chunks]
        return payload

    def list_documents(self, *, scope: AccessScope, limit: int, offset: int) -> list[dict[str, Any]]:
        scope_sql, scope_params = self._scope_clause(scope, "d")
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, title, document_type, canonical_uri, source_path, source_url, metadata_json
                FROM docs_documents d
                WHERE {scope_sql}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                [*scope_params, int(limit), int(offset)],
            ).fetchall()
        return [{"id": int(row["id"]), "title": row["title"], "uri": row["canonical_uri"]} for row in rows]

    def list_collections(self, *, scope: AccessScope) -> list[dict[str, Any]]:
        scope_sql, scope_params = self._scope_clause(scope)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.name, c.description, COUNT(cm.document_id) AS document_count
                FROM docs_collections c
                LEFT JOIN docs_collection_members cm ON cm.collection_id = c.id
                WHERE {scope_sql}
                GROUP BY c.id
                ORDER BY c.name
                """,
                scope_params,
            ).fetchall()
        return [dict(row) for row in rows]

    def list_keywords(self, *, scope: AccessScope) -> list[dict[str, Any]]:
        scope_sql, scope_params = self._scope_clause(scope, "k")
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT k.keyword, COUNT(dk.document_id) AS document_count
                FROM docs_keywords k
                LEFT JOIN docs_document_keywords dk ON dk.keyword_id = k.id
                WHERE {scope_sql}
                GROUP BY k.id
                ORDER BY k.keyword
                """,
                scope_params,
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve_name(self, *, scope: AccessScope, name: str) -> list[dict[str, Any]]:
        needle = name.strip()
        like = f"%{needle}%"
        scope_sql, scope_params = self._scope_clause(scope, "d")
        matches: list[dict[str, Any]] = []
        with self.connect() as conn:
            document_rows = conn.execute(
                f"SELECT id, title, canonical_uri FROM docs_documents d WHERE {scope_sql} AND title LIKE ? ORDER BY title LIMIT 10",
                [*scope_params, like],
            ).fetchall()
            collection_scope_sql, collection_scope_params = self._scope_clause(scope, "c")
            collection_rows = conn.execute(
                f"SELECT id, name, metadata_json FROM docs_collections c WHERE {collection_scope_sql} AND name LIKE ? ORDER BY name LIMIT 10",
                [*collection_scope_params, like],
            ).fetchall()
            keyword_scope_sql, keyword_scope_params = self._scope_clause(scope, "k")
            keyword_rows = conn.execute(
                f"SELECT id, keyword FROM docs_keywords k WHERE {keyword_scope_sql} AND keyword LIKE ? ORDER BY keyword LIMIT 10",
                [*keyword_scope_params, like],
            ).fetchall()

        matches.extend({"target_type": "document", "id": str(row["id"]), "title": row["title"], "uri": row["canonical_uri"]} for row in document_rows)
        matches.extend({"target_type": "collection", "id": row["name"], "title": row["name"], "metadata": json.loads(row["metadata_json"] or "{}")} for row in collection_rows)
        matches.extend({"target_type": "keyword", "id": row["keyword"], "title": row["keyword"]} for row in keyword_rows)
        return matches
```

- [ ] **Step 4: Run store tests and verify they pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add mcp_unified/docs/store tldw_Server_API/tests/MCP_unified/docs/test_docs_schema_store.py
git commit -m "feat: add docs sqlite corpus store"
```

## Task 3: Local Importers And Path Scope Enforcement

**Files:**

- Create: `mcp_unified/docs/importers/__init__.py`
- Create: `mcp_unified/docs/importers/base.py`
- Create: `mcp_unified/docs/importers/markdown.py`
- Create: `mcp_unified/docs/importers/html.py`
- Create: `mcp_unified/docs/importers/local.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py`

- [ ] **Step 1: Write failing importer tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from mcp_unified.docs.errors import DocsError
from mcp_unified.docs.importers.local import DocsImportService
from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore


def _service(tmp_path: Path, root: Path) -> DocsImportService:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    settings = DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(root.resolve(),))
    return DocsImportService(settings=settings, store=store)


def test_import_markdown_extracts_heading_chunks_and_keywords(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "guide.md"
    path.write_text("# Install\n\nUse sqlite FTS for local docs.\n", encoding="utf-8")
    service = _service(tmp_path, root)

    result = service.import_path(
        scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
        path=path,
        keywords=("setup",),
        collection_names=("Project Docs",),
    )

    assert result["status"] == "created"  # nosec B101
    assert result["documents"][0]["title"] == "Install"  # nosec B101
    assert result["documents"][0]["chunks"] >= 1  # nosec B101


def test_import_static_html_without_web_dependencies(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    path = root / "page.html"
    path.write_text("<html><body><h1>API</h1><p>Search docs with FTS.</p><script>ignored()</script></body></html>", encoding="utf-8")
    service = _service(tmp_path, root)

    result = service.import_path(
        scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
        path=path,
        keywords=(),
        collection_names=(),
    )

    assert result["status"] == "created"  # nosec B101
    assert "API" in result["documents"][0]["title"]  # nosec B101


def test_import_rejects_path_outside_trusted_roots(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside.md"
    root.mkdir()
    outside.write_text("# Outside\n", encoding="utf-8")
    service = _service(tmp_path, root)

    with pytest.raises(DocsError) as excinfo:
        service.import_path(
            scope=AccessScope(owner_scope="owner-a", profile_scope="profile-a"),
            path=outside,
            keywords=(),
            collection_names=(),
        )

    assert excinfo.value.code == "path_scope_denied"  # nosec B101
```

- [ ] **Step 2: Run importer tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `mcp_unified.docs.importers`.

- [ ] **Step 3: Implement importer dataclasses and chunking**

Create `mcp_unified/docs/importers/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedSection:
    heading: str
    level: int
    start_char: int | None
    end_char: int | None


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    document_type: str
    text: str
    sections: list[ParsedSection]
    canonical_uri: str
    source_path: str


def chunks_from_text(text: str, *, max_chars: int = 1_200, overlap: int = 120) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()
```

- [ ] **Step 4: Implement Markdown/MDX/text parsing**

Create `mcp_unified/docs/importers/markdown.py`:

```python
from __future__ import annotations

import re
from pathlib import Path

from .base import ParsedDocument, ParsedSection, file_uri

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_markdown(path: Path, text: str, document_type: str) -> ParsedDocument:
    sections: list[ParsedSection] = []
    title = path.stem
    offset = 0
    for line in text.splitlines(keepends=True):
        match = HEADING_RE.match(line.strip())
        if match:
            heading = match.group(2).strip()
            if title == path.stem:
                title = heading
            sections.append(
                ParsedSection(
                    heading=heading,
                    level=len(match.group(1)),
                    start_char=offset,
                    end_char=None,
                )
            )
        offset += len(line)
    return ParsedDocument(
        title=title,
        document_type=document_type,
        text=text,
        sections=sections,
        canonical_uri=file_uri(path),
        source_path=str(path.resolve()),
    )
```

- [ ] **Step 5: Implement static HTML parsing with stdlib only**

Create `mcp_unified/docs/importers/html.py`:

```python
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from .base import ParsedDocument, ParsedSection, file_uri


class StaticHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._active_heading: int | None = None
        self._heading_text: list[str] = []
        self.parts: list[str] = []
        self.sections: list[ParsedSection] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._active_heading = int(tag[1])
            self._heading_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if self._active_heading is not None and tag == f"h{self._active_heading}":
            heading = " ".join("".join(self._heading_text).split())
            if heading:
                self.sections.append(ParsedSection(heading=heading, level=self._active_heading, start_char=None, end_char=None))
                self.parts.append(f"\n{heading}\n")
            self._active_heading = None
            self._heading_text = []
        if tag in {"p", "li", "section", "article", "br"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._active_heading is not None:
            self._heading_text.append(data)
        else:
            clean = " ".join(data.split())
            if clean:
                self.parts.append(clean)
                self.parts.append(" ")


def parse_html(path: Path, text: str) -> ParsedDocument:
    parser = StaticHTMLTextParser()
    parser.feed(text)
    body = "\n".join(part.strip() for part in parser.parts if part.strip())
    title = parser.sections[0].heading if parser.sections else path.stem
    return ParsedDocument(
        title=title,
        document_type="html",
        text=body,
        sections=parser.sections,
        canonical_uri=file_uri(path),
        source_path=str(path.resolve()),
    )
```

- [ ] **Step 6: Implement local import orchestration**

Create `mcp_unified/docs/importers/local.py`:

```python
from __future__ import annotations

from pathlib import Path

from ..errors import DocsError
from ..models import AccessScope
from ..settings import DocsSettings
from ..store.sqlite import DocsCatalogStore
from .base import chunks_from_text
from .html import parse_html
from .markdown import parse_markdown


class DocsImportService:
    def __init__(self, *, settings: DocsSettings, store: DocsCatalogStore) -> None:
        self.settings = settings
        self.store = store

    def _assert_allowed_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        for root in self.settings.trusted_roots:
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except ValueError:
                continue
        raise DocsError("path_scope_denied", "Path is outside configured trusted roots.", {"path": str(resolved)})

    def import_path(
        self,
        *,
        scope: AccessScope,
        path: str | Path,
        keywords: tuple[str, ...],
        collection_names: tuple[str, ...],
    ) -> dict:
        target = self._assert_allowed_path(Path(path))
        files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
        imported: list[dict] = []
        for file_path in files:
            parsed = self._parse_file(file_path)
            chunk_texts = chunks_from_text(parsed.text)
            chunks = [{"text": chunk, "citation": f"{file_path.name}:{idx + 1}"} for idx, chunk in enumerate(chunk_texts)]
            document_id = self.store.upsert_document(
                scope=scope,
                title=parsed.title,
                document_type=parsed.document_type,
                canonical_uri=parsed.canonical_uri,
                source_path=parsed.source_path,
                source_url=None,
                text=parsed.text,
                sections=[section.__dict__ for section in parsed.sections],
                chunks=chunks,
                keywords=keywords,
                collection_names=collection_names,
                metadata={"importer": "local"},
            )
            imported.append({"id": document_id, "title": parsed.title, "chunks": len(chunks)})
        return {"status": "created" if imported else "unchanged", "documents": imported}

    def _parse_file(self, path: Path):
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return parse_markdown(path, text, "markdown")
        if suffix == ".mdx":
            return parse_markdown(path, text, "mdx")
        if suffix in {".txt", ".text"}:
            return parse_markdown(path, text, "text")
        if suffix in {".html", ".htm"}:
            return parse_html(path, text)
        raise DocsError("unsupported_import_format", "Unsupported local import file type.", {"path": str(path)})
```

Create `mcp_unified/docs/importers/__init__.py`:

```python
from .local import DocsImportService

__all__ = ["DocsImportService"]
```

- [ ] **Step 7: Run importer tests and boundary tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py \
  -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add mcp_unified/docs/importers tldw_Server_API/tests/MCP_unified/docs/test_docs_importers.py
git commit -m "feat: add docs local import service"
```

## Task 4: Retrieval, Context Packs, Collections, And Keywords

**Files:**

- Create: `mcp_unified/docs/retrieval/__init__.py`
- Create: `mcp_unified/docs/retrieval/search.py`
- Create: `mcp_unified/docs/retrieval/context.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py`

- [ ] **Step 1: Write failing retrieval tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py`:

```python
from __future__ import annotations

from pathlib import Path

from mcp_unified.docs.models import AccessScope, ContextRequest, SearchFilters, SearchRequest
from mcp_unified.docs.retrieval.context import DocsContextBuilder
from mcp_unified.docs.retrieval.search import DocsRetrievalService
from mcp_unified.docs.store.sqlite import DocsCatalogStore


def _seed_store(tmp_path: Path) -> tuple[DocsCatalogStore, AccessScope]:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    store.upsert_document(
        scope=scope,
        title="SQLite Guide",
        document_type="markdown",
        canonical_uri="file:///docs/sqlite.md",
        source_path="/docs/sqlite.md",
        source_url=None,
        text="SQLite FTS5 supports local retrieval. Agents need citations.",
        sections=[],
        chunks=[{"text": "SQLite FTS5 supports local retrieval.", "citation": "sqlite.md:1"}],
        keywords=("database", "fts"),
        collection_names=("Reference",),
        metadata={"package": "sqlite", "version": "3"},
    )
    return store, scope


def test_search_filters_by_collection_and_keyword(tmp_path: Path) -> None:
    store, scope = _seed_store(tmp_path)
    service = DocsRetrievalService(store)

    response = service.search(
        scope=scope,
        request=SearchRequest(query="retrieval", filters=SearchFilters(collection="Reference", keywords=("fts",))),
    )

    assert response["results"][0]["title"] == "SQLite Guide"  # nosec B101
    assert response["results"][0]["citation"] == "sqlite.md:1"  # nosec B101


def test_context_pack_respects_character_budget(tmp_path: Path) -> None:
    store, scope = _seed_store(tmp_path)
    builder = DocsContextBuilder(DocsRetrievalService(store))

    pack = builder.build(scope=scope, request=ContextRequest(query="SQLite", max_chunks=2, max_characters=40))

    assert pack["budget"]["max_characters"] == 40  # nosec B101
    assert pack["budget"]["used_characters"] <= 40  # nosec B101
    assert pack["chunks"]  # nosec B101
    assert pack["citations"][0]["uri"] == "file:///docs/sqlite.md"  # nosec B101


def test_list_collections_and_keywords(tmp_path: Path) -> None:
    store, scope = _seed_store(tmp_path)
    service = DocsRetrievalService(store)

    collections = service.list_collections(scope=scope)
    keywords = service.list_keywords(scope=scope)

    assert collections["collections"][0]["name"] == "Reference"  # nosec B101
    assert {item["keyword"] for item in keywords["keywords"]} == {"database", "fts"}  # nosec B101
```

- [ ] **Step 2: Run retrieval tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `mcp_unified.docs.retrieval`.

- [ ] **Step 3: Add retrieval service**

Create `mcp_unified/docs/retrieval/search.py`:

```python
from __future__ import annotations

from typing import Any

from ..models import AccessScope, SearchRequest
from ..store.sqlite import DocsCatalogStore


class DocsRetrievalService:
    def __init__(self, store: DocsCatalogStore) -> None:
        self.store = store

    def search(self, *, scope: AccessScope, request: SearchRequest) -> dict[str, Any]:
        rows = self.store.search_chunks(
            scope=scope,
            query=request.query,
            limit=request.limit,
            offset=request.offset,
            filters=request.filters,
            snippet_length=request.snippet_length,
        )
        return {"results": rows, "count": len(rows), "warnings": []}

    def get(self, *, scope: AccessScope, target: str, mode: str = "snippet") -> dict[str, Any]:
        return self.store.get_document(scope=scope, target=target, mode=mode)

    def list_documents(self, *, scope: AccessScope, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return {"documents": self.store.list_documents(scope=scope, limit=limit, offset=offset)}

    def list_collections(self, *, scope: AccessScope) -> dict[str, Any]:
        return {"collections": self.store.list_collections(scope=scope)}

    def list_keywords(self, *, scope: AccessScope) -> dict[str, Any]:
        return {"keywords": self.store.list_keywords(scope=scope)}
```

Use the `DocsCatalogStore.search_chunks` implementation from Task 2. It already supports collection, keyword, document type, URI prefix, package metadata, and version metadata filters with parameterized predicates.

- [ ] **Step 4: Add context builder**

Create `mcp_unified/docs/retrieval/context.py`:

```python
from __future__ import annotations

from typing import Any

from ..models import AccessScope, ContextRequest, SearchRequest
from .search import DocsRetrievalService


class DocsContextBuilder:
    def __init__(self, retrieval: DocsRetrievalService) -> None:
        self.retrieval = retrieval

    def build(self, *, scope: AccessScope, request: ContextRequest) -> dict[str, Any]:
        search = self.retrieval.search(
            scope=scope,
            request=SearchRequest(
                query=request.query,
                filters=request.filters,
                limit=max(request.max_chunks * 2, request.max_chunks),
                snippet_length=request.max_characters,
            ),
        )
        chunks: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        seen_documents: set[int] = set()
        used = 0
        for result in search["results"]:
            if len(chunks) >= request.max_chunks:
                break
            if result["document_id"] not in seen_documents and len(seen_documents) >= request.max_documents:
                continue
            text = result["snippet"]
            if used + len(text) > request.max_characters:
                remaining = request.max_characters - used
                if remaining <= 0:
                    break
                text = text[:remaining]
            used += len(text)
            seen_documents.add(result["document_id"])
            chunk = dict(result)
            chunk["text"] = text
            chunks.append(chunk)
            citations.append({"uri": result["uri"], "citation": result["citation"], "title": result["title"]})
        return {
            "query": request.query,
            "chunks": chunks,
            "citations": citations,
            "omitted": max(0, len(search["results"]) - len(chunks)),
            "budget": {"max_characters": request.max_characters, "used_characters": used},
            "warnings": search.get("warnings", []),
        }
```

Create `mcp_unified/docs/retrieval/__init__.py`:

```python
from .context import DocsContextBuilder
from .search import DocsRetrievalService

__all__ = ["DocsContextBuilder", "DocsRetrievalService"]
```

- [ ] **Step 5: Run retrieval tests and verify they pass**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add mcp_unified/docs/retrieval mcp_unified/docs/store/sqlite.py tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py
git commit -m "feat: add docs retrieval and context packs"
```

## Task 5: Alias Resolution And Context7-Compatible Read Aliases

**Files:**

- Create: `mcp_unified/docs/retrieval/aliases.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`

- [ ] **Step 1: Write failing alias tests**

Add these tests to `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`:

```python
from __future__ import annotations

from pathlib import Path

from mcp_unified.docs.mcp_module import DocsMCPToolProvider
from mcp_unified.docs.models import AccessScope
from mcp_unified.docs.settings import DocsSettings
from mcp_unified.docs.store.sqlite import DocsCatalogStore


def _provider(tmp_path: Path) -> tuple[DocsMCPToolProvider, AccessScope]:
    store = DocsCatalogStore(tmp_path / "docs.db")
    store.migrate()
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")
    store.upsert_document(
        scope=scope,
        title="SQLite Reference",
        document_type="markdown",
        canonical_uri="file:///docs/sqlite.md",
        source_path="/docs/sqlite.md",
        source_url=None,
        text="SQLite FTS5 reference for agents.",
        sections=[],
        chunks=[{"text": "SQLite FTS5 reference for agents.", "citation": "sqlite.md:1"}],
        keywords=("database",),
        collection_names=("sqlite",),
        metadata={"package": "sqlite", "version": "3"},
    )
    settings = DocsSettings(db_path=tmp_path / "docs.db", trusted_roots=(tmp_path,))
    return DocsMCPToolProvider(settings=settings, store=store), scope


def test_context7_resolve_library_id_prefers_package_like_collection(tmp_path: Path) -> None:
    provider, scope = _provider(tmp_path)

    result = provider.execute("resolve-library-id", {"libraryName": "sqlite"}, scope=scope)

    assert result["matches"][0]["id"] == "sqlite"  # nosec B101
    assert result["matches"][0]["canonical_tool"] == "docs.resolve"  # nosec B101


def test_context7_get_library_docs_routes_to_context(tmp_path: Path) -> None:
    provider, scope = _provider(tmp_path)

    result = provider.execute("get-library-docs", {"context7CompatibleLibraryID": "sqlite", "topic": "FTS5"}, scope=scope)

    assert result["canonical_tool"] == "docs.context"  # nosec B101
    assert result["chunks"]  # nosec B101


def test_general_resolve_does_not_force_library_version_semantics(tmp_path: Path) -> None:
    provider, scope = _provider(tmp_path)

    result = provider.execute("docs.resolve", {"name": "database"}, scope=scope)

    assert result["matches"]  # nosec B101
    assert result["query"] == "database"  # nosec B101
```

- [ ] **Step 2: Run alias tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `mcp_unified.docs.mcp_module`.

- [ ] **Step 3: Implement alias resolver**

Create `mcp_unified/docs/retrieval/aliases.py`:

```python
from __future__ import annotations

from typing import Any

from ..models import AccessScope
from ..store.sqlite import DocsCatalogStore


class DocsAliasResolver:
    def __init__(self, store: DocsCatalogStore) -> None:
        self.store = store

    def resolve(self, *, scope: AccessScope, name: str) -> dict[str, Any]:
        query = name.strip()
        matches = self.store.resolve_name(scope=scope, name=query)
        return {"query": query, "matches": matches, "ambiguous": len(matches) > 1}

    def resolve_library_id(self, *, scope: AccessScope, library_name: str) -> dict[str, Any]:
        result = self.resolve(scope=scope, name=library_name)
        package_like = [match for match in result["matches"] if match.get("target_type") in {"collection", "package"}]
        return {
            "query": library_name,
            "matches": [{**match, "canonical_tool": "docs.resolve"} for match in package_like],
            "canonical_tool": "docs.resolve",
        }
```

Use the `DocsCatalogStore.resolve_name` implementation from Task 2. It returns document, collection, and keyword matches from the active scope. Package-like collections are represented by collection matches whose `metadata` carries package or version fields.

- [ ] **Step 4: Commit Task 5 after provider implementation in Task 6**

Do not commit after this task if `DocsMCPToolProvider` is not implemented yet. Continue directly to Task 6 and commit alias and provider work together.

## Task 6: Runtime-Neutral MCP Tool Provider

**Files:**

- Create: `mcp_unified/docs/mcp_module.py`
- Modify: `mcp_unified/docs/__init__.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`

- [ ] **Step 1: Extend provider tests for tool discovery and write classification**

Add to `test_docs_mcp_provider.py`:

```python
def test_provider_advertises_stage1_tools_without_ingest_url(tmp_path: Path) -> None:
    provider, _scope = _provider(tmp_path)

    names = {tool["name"] for tool in provider.tool_definitions()}

    assert "docs.search" in names  # nosec B101
    assert "docs.context" in names  # nosec B101
    assert "docs.import_path" in names  # nosec B101
    assert "resolve-library-id" in names  # nosec B101
    assert "get-library-docs" in names  # nosec B101
    assert "docs.ingest_url" not in names  # nosec B101


def test_provider_marks_write_tools_with_ingestion_or_management_category(tmp_path: Path) -> None:
    provider, _scope = _provider(tmp_path)
    write_names = {
        "docs.import_path",
        "docs.collections.create",
        "docs.collections.update",
        "docs.collections.set_membership",
        "docs.keywords.apply",
    }

    tools = {tool["name"]: tool for tool in provider.tool_definitions()}

    for name in write_names:
        assert tools[name]["metadata"]["category"] in {"ingestion", "management"}  # nosec B101
```

- [ ] **Step 2: Implement provider dispatch**

Create `mcp_unified/docs/mcp_module.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from .importers.local import DocsImportService
from .models import AccessScope, ContextRequest, SearchFilters, SearchRequest
from .retrieval.aliases import DocsAliasResolver
from .retrieval.context import DocsContextBuilder
from .retrieval.search import DocsRetrievalService
from .settings import DocsSettings
from .store.sqlite import DocsCatalogStore


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str], category: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required},
        "metadata": {"category": category, "readOnlyHint": category not in {"ingestion", "management"}},
    }


class DocsMCPToolProvider:
    def __init__(self, *, settings: DocsSettings, store: DocsCatalogStore | None = None) -> None:
        self.settings = settings
        self.store = store or DocsCatalogStore(settings.db_path)
        self.store.migrate()
        self.retrieval = DocsRetrievalService(self.store)
        self.context = DocsContextBuilder(self.retrieval)
        self.aliases = DocsAliasResolver(self.store)
        self.importer = DocsImportService(settings=settings, store=self.store)

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            _tool("docs.search", "Search the local docs corpus.", {"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"], "search"),
            _tool("docs.get", "Get a document, section, or chunk.", {"id": {"type": "string"}, "mode": {"type": "string"}}, ["id"], "retrieval"),
            _tool("docs.context", "Build a bounded RAG context pack.", {"query": {"type": "string"}, "max_chunks": {"type": "integer"}, "max_characters": {"type": "integer"}}, ["query"], "retrieval"),
            _tool("docs.resolve", "Resolve a document, collection, source, keyword, or package-like docs name.", {"name": {"type": "string"}}, ["name"], "retrieval"),
            _tool("docs.list", "List docs corpus records.", {"kind": {"type": "string"}, "limit": {"type": "integer"}, "offset": {"type": "integer"}}, ["kind"], "retrieval"),
            _tool("docs.status", "Report docs corpus health and capability status.", {}, [], "retrieval"),
            _tool("docs.import_path", "Import local files under configured trusted roots.", {"path": {"type": "string"}, "keywords": {"type": "array"}, "collections": {"type": "array"}}, ["path"], "ingestion"),
            _tool("docs.collections.list", "List collections.", {}, [], "retrieval"),
            _tool("docs.collections.create", "Create a collection.", {"name": {"type": "string"}, "description": {"type": "string"}}, ["name"], "management"),
            _tool("docs.collections.update", "Update a collection.", {"name": {"type": "string"}, "description": {"type": "string"}}, ["name"], "management"),
            _tool("docs.collections.set_membership", "Set collection membership.", {"collection": {"type": "string"}, "document_id": {"type": "integer"}, "action": {"type": "string"}}, ["collection", "document_id", "action"], "management"),
            _tool("docs.keywords.list", "List keywords.", {}, [], "retrieval"),
            _tool("docs.keywords.apply", "Apply keywords to a document.", {"document_id": {"type": "integer"}, "keywords": {"type": "array"}}, ["document_id", "keywords"], "management"),
            _tool("resolve-library-id", "Context7-compatible library id resolver backed by docs collections.", {"libraryName": {"type": "string"}}, ["libraryName"], "retrieval"),
            _tool("get-library-docs", "Context7-compatible docs retrieval backed by docs.context.", {"context7CompatibleLibraryID": {"type": "string"}, "topic": {"type": "string"}}, ["context7CompatibleLibraryID"], "retrieval"),
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any], *, scope: AccessScope) -> Any:
        args = dict(arguments or {})
        if tool_name == "docs.status":
            status = self.store.status()
            status["web_acquisition_enabled"] = self.settings.enable_web_acquisition
            return status
        if tool_name == "docs.search":
            return self.retrieval.search(scope=scope, request=SearchRequest(query=str(args["query"]), limit=int(args.get("limit", 10))))
        if tool_name == "docs.context":
            return self.context.build(
                scope=scope,
                request=ContextRequest(
                    query=str(args["query"]),
                    max_chunks=int(args.get("max_chunks", 8)),
                    max_characters=int(args.get("max_characters", 12_000)),
                ),
            )
        if tool_name == "docs.resolve":
            return self.aliases.resolve(scope=scope, name=str(args["name"]))
        if tool_name == "resolve-library-id":
            return self.aliases.resolve_library_id(scope=scope, library_name=str(args["libraryName"]))
        if tool_name == "get-library-docs":
            collection = str(args["context7CompatibleLibraryID"])
            topic = str(args.get("topic") or collection)
            result = self.context.build(scope=scope, request=ContextRequest(query=topic, filters=SearchFilters(collection=collection)))
            result["canonical_tool"] = "docs.context"
            return result
        if tool_name == "docs.import_path":
            return self.importer.import_path(
                scope=scope,
                path=Path(str(args["path"])),
                keywords=tuple(str(item) for item in args.get("keywords") or ()),
                collection_names=tuple(str(item) for item in args.get("collections") or ()),
            )
        return self._execute_management_or_list(tool_name=tool_name, args=args, scope=scope)
```

Continue `mcp_unified/docs/mcp_module.py` with explicit management and list dispatch:

```python
    def _execute_management_or_list(self, *, tool_name: str, args: dict[str, Any], scope: AccessScope) -> Any:
        if tool_name == "docs.get":
            return self.retrieval.get(scope=scope, target=str(args["id"]), mode=str(args.get("mode") or "snippet"))
        if tool_name == "docs.list":
            kind = str(args["kind"])
            limit = int(args.get("limit", 50))
            offset = int(args.get("offset", 0))
            if kind == "documents":
                return self.retrieval.list_documents(scope=scope, limit=limit, offset=offset)
            if kind == "collections":
                return self.retrieval.list_collections(scope=scope)
            if kind == "keywords":
                return self.retrieval.list_keywords(scope=scope)
            if kind == "sources":
                return {"sources": [], "warnings": [{"code": "sources_not_populated_in_stage1"}]}
            raise ValueError(f"Unsupported docs.list kind: {kind}")
        if tool_name == "docs.collections.list":
            return self.retrieval.list_collections(scope=scope)
        if tool_name == "docs.collections.create":
            name = str(args["name"]).strip()
            description = str(args.get("description") or "")
            collection_id = self.store.create_collection(scope=scope, name=name, description=description)
            return {"status": "created", "id": collection_id, "name": name}
        if tool_name == "docs.collections.update":
            name = str(args["name"]).strip()
            description = str(args.get("description") or "")
            updated = self.store.update_collection(scope=scope, name=name, description=description)
            return {"status": "updated" if updated else "unchanged", "name": name}
        if tool_name == "docs.collections.set_membership":
            collection = str(args["collection"]).strip()
            document_id = int(args["document_id"])
            action = str(args["action"]).strip().lower()
            result = self.store.set_collection_membership(scope=scope, collection=collection, document_id=document_id, action=action)
            return {"status": result, "collection": collection, "document_id": document_id}
        if tool_name == "docs.keywords.list":
            return self.retrieval.list_keywords(scope=scope)
        if tool_name == "docs.keywords.apply":
            document_id = int(args["document_id"])
            keywords = tuple(str(item) for item in args.get("keywords") or ())
            self.store.apply_keywords(scope=scope, document_id=document_id, keywords=keywords)
            return {"status": "updated", "document_id": document_id, "keywords": list(keywords)}
        raise ValueError(f"Unknown docs tool: {tool_name}")
```

Add the backing store methods in `mcp_unified/docs/store/sqlite.py`:

```python
    def create_collection(self, *, scope: AccessScope, name: str, description: str = "") -> int:
        with self.connect() as conn:
            collection_id = self._collection_id(conn, scope, name)
            conn.execute(
                "UPDATE docs_collections SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (description, collection_id),
            )
            return collection_id

    def update_collection(self, *, scope: AccessScope, name: str, description: str) -> bool:
        scope_sql, scope_params = self._scope_clause(scope)
        with self.connect() as conn:
            cur = conn.execute(
                f"UPDATE docs_collections SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE {scope_sql} AND name = ?",
                [description, *scope_params, name],
            )
            return cur.rowcount > 0

    def set_collection_membership(self, *, scope: AccessScope, collection: str, document_id: int, action: str) -> str:
        scope_sql, scope_params = self._scope_clause(scope, "d")
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT d.id FROM docs_documents d WHERE {scope_sql} AND d.id = ?",
                [*scope_params, document_id],
            ).fetchone()
            if row is None:
                from ..errors import DocsError

                raise DocsError("document_not_found", "Document not found in active scope.", {"document_id": document_id})
            collection_id = self._collection_id(conn, scope, collection)
            if action == "add":
                conn.execute(
                    "INSERT OR IGNORE INTO docs_collection_members(collection_id, document_id) VALUES (?, ?)",
                    (collection_id, document_id),
                )
                return "added"
            if action == "remove":
                conn.execute(
                    "DELETE FROM docs_collection_members WHERE collection_id = ? AND document_id = ?",
                    (collection_id, document_id),
                )
                return "removed"
            raise ValueError("action must be add or remove")

    def apply_keywords(self, *, scope: AccessScope, document_id: int, keywords: Iterable[str]) -> None:
        scope_sql, scope_params = self._scope_clause(scope, "d")
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT d.id FROM docs_documents d WHERE {scope_sql} AND d.id = ?",
                [*scope_params, document_id],
            ).fetchone()
            if row is None:
                from ..errors import DocsError

                raise DocsError("document_not_found", "Document not found in active scope.", {"document_id": document_id})
            conn.execute("DELETE FROM docs_document_keywords WHERE document_id = ?", (document_id,))
            self._replace_keywords(conn, scope, document_id, keywords)
```

- [ ] **Step 3: Update exports**

Update `mcp_unified/docs/__init__.py`:

```python
from .errors import DocsError
from .importers import DocsImportService
from .mcp_module import DocsMCPToolProvider
from .models import AccessScope, ContextRequest, DocumentRecord, SearchFilters, SearchRequest, SearchResult
from .settings import DocsSettings
from .store import DocsCatalogStore

__all__ = [
    "AccessScope",
    "ContextRequest",
    "DocsCatalogStore",
    "DocsError",
    "DocsImportService",
    "DocsMCPToolProvider",
    "DocsSettings",
    "DocumentRecord",
    "SearchFilters",
    "SearchRequest",
    "SearchResult",
]
```

- [ ] **Step 4: Run provider, alias, retrieval, and boundary tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_retrieval_context.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit Tasks 5 and 6**

Run:

```bash
git add mcp_unified/docs tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py
git commit -m "feat: add docs mcp tool provider"
```

## Task 7: Existing MCP Module Shim And Config Registration

**Files:**

- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`
- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`

- [ ] **Step 1: Write failing shim tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.docs_module import DocsModule
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


@pytest.mark.asyncio
async def test_docs_module_advertises_provider_tools(tmp_path: Path) -> None:
    module = DocsModule(
        ModuleConfig(
            name="docs",
            settings={"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)], "enable_web_acquisition": False},
        )
    )
    await module.on_initialize()

    names = {tool["name"] for tool in await module.get_tools()}

    assert "docs.search" in names  # nosec B101
    assert "docs.import_path" in names  # nosec B101
    assert "docs.ingest_url" not in names  # nosec B101


@pytest.mark.asyncio
async def test_docs_module_executes_with_context_scope(tmp_path: Path) -> None:
    doc_path = tmp_path / "guide.md"
    doc_path.write_text("# Guide\n\nSQLite local docs.\n", encoding="utf-8")
    module = DocsModule(
        ModuleConfig(
            name="docs",
            settings={"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)], "enable_web_acquisition": False},
        )
    )
    await module.on_initialize()
    ctx = RequestContext(request_id="docs-test", user_id="user-1", client_id="unit", metadata={"profile_scope": "profile-1"})

    await module.execute_tool("docs.import_path", {"path": str(doc_path), "keywords": ["sqlite"]}, context=ctx)
    result = await module.execute_tool("docs.search", {"query": "SQLite"}, context=ctx)

    assert result["results"]  # nosec B101
```

- [ ] **Step 2: Run shim tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `docs_module`.

- [ ] **Step 3: Implement shim**

Create `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`:

```python
from __future__ import annotations

from typing import Any

from loguru import logger

from mcp_unified.docs import AccessScope, DocsMCPToolProvider, DocsSettings
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule


class DocsModule(BaseModule):
    """Thin adapter from the existing MCP module runtime to mcp_unified.docs."""

    async def on_initialize(self) -> None:
        self._settings = DocsSettings.from_mapping(self.config.settings or {})
        self._provider = DocsMCPToolProvider(settings=self._settings)
        logger.info("Initialized Docs MCP module with db_path={}", self._settings.db_path)

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": hasattr(self, "_provider")}

    async def get_tools(self) -> list[dict[str, Any]]:
        return self._provider.tool_definitions()

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments or {})
        self.validate_tool_arguments(tool_name, args)
        return self._provider.execute(tool_name, args, scope=self._scope_from_context(context))

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == "docs.import_path" and not str(arguments.get("path") or "").strip():
            raise ValueError("path is required")
        if tool_name in {"docs.search", "docs.context"} and not str(arguments.get("query") or "").strip():
            raise ValueError("query is required")

    def _scope_from_context(self, context: Any | None) -> AccessScope:
        metadata = getattr(context, "metadata", None) if context is not None else None
        profile_scope = metadata.get("profile_scope") if isinstance(metadata, dict) else None
        user_id = getattr(context, "user_id", None) if context is not None else None
        return AccessScope(owner_scope=str(user_id) if user_id is not None else None, profile_scope=str(profile_scope) if profile_scope else None)
```

- [ ] **Step 4: Register module in config**

Add to `tldw_Server_API/Config_Files/mcp_modules.yaml` near `knowledge`:

```yaml
  - id: docs
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.docs_module:DocsModule
    enabled: true
    name: Docs Corpus
    version: "0.1.0"
    department: knowledge
    max_concurrent: 10
    settings:
      db_path: Databases/mcp_docs.db
      trusted_roots:
        - Docs
      enable_web_acquisition: false
      max_import_file_bytes: 2000000
```

- [ ] **Step 5: Add config and write-classification tests**

Extend `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`:

```python
def test_default_mcp_modules_config_declares_docs_module_without_web_acquisition() -> None:
    config_path = Path("tldw_Server_API/Config_Files/mcp_modules.yaml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modules = {module["id"]: module for module in data["modules"]}

    docs_module = modules["docs"]

    assert docs_module["enabled"] is True  # nosec B101
    assert docs_module["class"] == "tldw_Server_API.app.core.MCP_unified.modules.implementations.docs_module:DocsModule"  # nosec B101
    assert docs_module["settings"]["enable_web_acquisition"] is False  # nosec B101
```

Extend `tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py` by importing `DocsModule` and adding it to the `modules` list:

```python
from tldw_Server_API.app.core.MCP_unified.modules.implementations.docs_module import DocsModule

DocsModule(ModuleConfig(name="docs", settings={"db_path": ":memory:", "trusted_roots": []}))
```

- [ ] **Step 6: Run shim and catalog tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py::test_default_mcp_modules_config_declares_docs_module_without_web_acquisition \
  tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py::test_write_tools_have_ingestion_or_management_category \
  -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 7**

Run:

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py \
  tldw_Server_API/Config_Files/mcp_modules.yaml \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py
git commit -m "feat: register docs corpus mcp module"
```

## Task 8: Integration Pass, Docs Status, And Final Verification

**Files:**

- Modify: `mcp_unified/docs/store/sqlite.py`
- Modify: `mcp_unified/docs/mcp_module.py`
- Modify: `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`

- [ ] **Step 1: Add disabled web acquisition status regression**

Add to `tldw_Server_API/tests/MCP_unified/docs/test_docs_mcp_provider.py`:

```python
def test_status_reports_web_acquisition_disabled(tmp_path: Path) -> None:
    provider, scope = _provider(tmp_path)

    status = provider.execute("docs.status", {}, scope=scope)
    names = {tool["name"] for tool in provider.tool_definitions()}

    assert status["web_acquisition_enabled"] is False  # nosec B101
    assert "docs.ingest_url" not in names  # nosec B101
```

- [ ] **Step 2: Run the full Stage 1 docs test set**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs -v
```

Expected: PASS.

- [ ] **Step 3: Run adjacent MCP regression tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_knowledge_search_defaults.py \
  -v
```

Expected: PASS.

- [ ] **Step 4: Run import-boundary and packaging checks**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -v
python -m pip install -e . --no-deps
python -c "import mcp_unified.docs as docs; print(docs.DocsSettings)"
```

Expected: pytest PASS, editable install succeeds, and the Python command prints the `DocsSettings` class.

- [ ] **Step 5: Run Bandit on touched code**

Run:

```bash
source .venv/bin/activate
python -m bandit -r mcp_unified tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py -f json -o /tmp/bandit_mcp_docs_stage1.json
```

Expected: exit 0 with no new findings in touched code.

- [ ] **Step 6: Run a final scoped diff review**

Run:

```bash
git diff --stat
git diff -- mcp_unified tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py tldw_Server_API/Config_Files/mcp_modules.yaml tldw_Server_API/tests/MCP_unified/docs
```

Expected: changes are limited to the docs corpus package, MCP shim/config, and tests named in this plan.

- [ ] **Step 7: Commit final cleanup**

If Task 8 changed files after Task 7, run:

```bash
git add mcp_unified tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py tldw_Server_API/tests/MCP_unified/docs
git commit -m "test: verify docs corpus mcp integration"
```

If Task 8 only verified existing committed files, do not create an empty commit.

## Stage 1 Completion Checklist

- [ ] `mcp_unified.docs` imports without importing `tldw_Server_API`.
- [ ] `mcp_unified.docs` imports without Playwright, trafilatura, requests, aiohttp, or existing web-scraping service modules.
- [ ] `docs.ingest_url` is not advertised in Stage 1.
- [ ] `docs.status` reports `web_acquisition_enabled: false` for default config.
- [ ] Local Markdown, MDX, text, and static HTML files can be imported from trusted roots.
- [ ] Path escapes outside trusted roots raise `path_scope_denied`.
- [ ] Documents are searchable even without collection membership.
- [ ] Collection and keyword filters work.
- [ ] `owner_scope` and `profile_scope` are enforced in store helpers.
- [ ] `docs.context` returns bounded chunks with citations and budget metadata.
- [ ] Context7-compatible aliases route through canonical docs operations and do not require library/version fields for general documents.
- [ ] Write tools are categorized as `ingestion` or `management`.
- [ ] Existing MCP module loader can register `DocsModule`.
- [ ] Bandit has no new findings in touched code.

## Commands For Final Verification

Run these before calling Stage 1 complete:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/MCP_unified/docs -v
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_write_tools_validators.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_knowledge_search_defaults.py \
  -v
python -m bandit -r mcp_unified tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py -f json -o /tmp/bandit_mcp_docs_stage1.json
```

## Spec Coverage Review

- Document-first data model: Tasks 1, 2, and 4.
- SQLite + FTS5 required backend: Task 2.
- Local import from trusted roots: Task 3.
- Optional web acquisition excluded from baseline: Tasks 1, 6, and 8.
- Canonical `docs.*` tool surface: Tasks 6 and 7.
- Explicit collection and keyword read/write tools: Tasks 4, 6, and 7.
- Context7-compatible read aliases: Tasks 5 and 6.
- Store-level scope enforcement: Task 2.
- Import-boundary cleanliness: Tasks 1 and 8.
- Existing `tldw_server` MCP availability through shim: Task 7.
