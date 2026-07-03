# RAG Query Log Sanitization Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate `AUDIT-2026-06-27-CHAT-002` by preventing RAG endpoint info logs from emitting raw query text.

**Architecture:** Keep the fix local to `rag_unified.py` by reusing the existing safe `query_hash` plus query-length logging pattern already used by `/rag/simple`. Add focused log-capture tests that call the affected endpoint functions with sentinel private query text and assert info logs do not contain it.

**Tech Stack:** FastAPI endpoint functions, Loguru logging, pytest, project virtualenv Python, Bandit.

---

## File Structure

- Modify `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
  - Replace raw query text in `unified_search_endpoint` info log with safe metadata.
  - Replace raw query text in `advanced_search_endpoint` info log with safe metadata.
  - Add a small private helper only if it removes duplication without expanding scope.
- Create or modify `tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py`
  - Add log-capture tests for `unified_search_endpoint` and `advanced_search_endpoint`.
- Update `backlog/tasks/task-12135 - Remediate-RAG-raw-query-logging-audit-finding.md`
  - Record implementation notes, verification, final summary, and touched files.

## Stage 1: Red Tests For Raw Query Info Logs

**Goal:** Prove the current endpoint logs leak raw query text at info level.

**Success Criteria:** A focused pytest file fails because the sentinel private query appears in captured info logs.

**Tests:** `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py -q`

**Status:** Not Started

- [ ] **Step 1: Add a log-capture test file**

Create `tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py` with tests that:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints import rag_unified
from tldw_Server_API.app.api.v1.schemas.rag_schemas import UnifiedRAGRequest


pytestmark = pytest.mark.unit


class _LogCapture:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self._handler_id: int | None = None

    def __enter__(self) -> "_LogCapture":
        self._handler_id = logger.add(
            lambda message: self.messages.append(str(message)),
            level="INFO",
            format="{message}",
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handler_id is not None:
            logger.remove(self._handler_id)


@pytest.mark.asyncio
async def test_unified_search_info_log_omits_raw_query(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_query = "private customer token sk-live-rag-secret"

    monkeypatch.setattr(
        rag_unified,
        "_apply_media_collection_scope",
        lambda request, _collections_db: request,
    )
    monkeypatch.setattr(
        rag_unified,
        "_build_standard_request_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after logging")),
    )

    request_raw = SimpleNamespace(state=SimpleNamespace())
    current_user = SimpleNamespace(username="alice", id=1)

    with _LogCapture() as logs:
        with pytest.raises(Exception, match="Search failed"):
            await rag_unified.unified_search_endpoint(
                request_raw=request_raw,
                request=UnifiedRAGRequest(query=sentinel_query),
                background_tasks=SimpleNamespace(),
                current_user=current_user,
                media_db=None,
                chacha_db=None,
                prompts_db=None,
                collections_db=None,
            )

    joined = "\n".join(logs.messages)
    assert sentinel_query not in joined
    assert "query_hash=" in joined
    assert f"len={len(sentinel_query)}" in joined


@pytest.mark.asyncio
async def test_advanced_search_info_log_omits_raw_query(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_query = "confidential acquisition plan RAG query"

    async def _stop_after_logging(*args, **kwargs):
        raise RuntimeError("stop after logging")

    monkeypatch.setattr(rag_unified, "advanced_search", _stop_after_logging)

    current_user = SimpleNamespace(username="alice", id=1)

    with _LogCapture() as logs:
        with pytest.raises(Exception, match="Search failed"):
            await rag_unified.advanced_search_endpoint(
                request=SimpleNamespace(state=SimpleNamespace()),
                query=sentinel_query,
                with_citations=True,
                with_answer=True,
                current_user=current_user,
                media_db=None,
                chacha_db=None,
            )

    joined = "\n".join(logs.messages)
    assert sentinel_query not in joined
    assert "query_hash=" in joined
    assert f"len={len(sentinel_query)}" in joined
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py -q
```

Expected: failures because existing info logs include `private customer token sk-live-rag-secret` and `confidential acquisition plan RAG query`.

## Stage 2: Sanitize The RAG Info Logs

**Goal:** Replace raw query strings with stable non-sensitive metadata.

**Success Criteria:** The new tests pass and the endpoint logs still provide request visibility with endpoint, query hash, query length, and user where already logged.

**Tests:** Same focused pytest file.

**Status:** Not Started

- [ ] **Step 1: Add a safe query metadata helper**

In `rag_unified.py`, near existing endpoint helpers, add:

```python
def _safe_query_log_metadata(query: str | None) -> tuple[str, int]:
    query_text = query or ""
    query_hash = hashlib.md5(query_text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return query_hash, len(query_text)
```

- [ ] **Step 2: Update `unified_search_endpoint` info logging**

Replace:

```python
logger.info(f"Unified RAG search: query='{request.query}', user={current_user.username if current_user else 'anonymous'}")
```

with:

```python
query_hash, query_length = _safe_query_log_metadata(request.query)
logger.info(
    "Unified RAG search: query_hash={} len={} user={}",
    query_hash,
    query_length,
    current_user.username if current_user else "anonymous",
)
```

- [ ] **Step 3: Update `advanced_search_endpoint` info logging**

Replace:

```python
logger.info(f"Advanced search: query='{query}'")
```

with:

```python
query_hash, query_length = _safe_query_log_metadata(query)
logger.info(
    "Advanced search: query_hash={} len={}",
    query_hash,
    query_length,
)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py -q
```

Expected: `2 passed`.

## Stage 3: Focused Verification And Task Finalization

**Goal:** Verify the touched scope and update the task record.

**Success Criteria:** Focused pytest, Bandit, and diff whitespace checks pass; task record captures verification and residual risk.

**Tests:** Focused pytest, Bandit, `git diff --check`.

**Status:** Not Started

- [ ] **Step 1: Run focused pytest**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py -q
```

- [ ] **Step 2: Run Bandit on touched production code**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit tldw_Server_API/app/api/v1/endpoints/rag_unified.py -f json -o /tmp/bandit_rag_query_logging_12135.json
```

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

- [ ] **Step 4: Update `TASK-12135`**

Record:

```text
Implemented raw-query log sanitization for unified and advanced RAG search info logs. Focused log-capture tests assert sentinel private query strings are not emitted at info level while query_hash and length remain. Verification: focused pytest, Bandit, and git diff --check.
```

- [ ] **Step 5: Commit**

Run:

```bash
git add Docs/superpowers/plans/2026-07-03-rag-query-log-sanitization-remediation.md \
  "backlog/tasks/task-12135 - Remediate-RAG-raw-query-logging-audit-finding.md" \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/tests/RAG/test_rag_query_logging_sanitization.py
git commit -m "Sanitize RAG query info logs"
```

## Self-Review

- Spec coverage: The plan maps directly to `AUDIT-2026-06-27-CHAT-002`, covering both raw-query info logs named in the audit report and preserving non-sensitive diagnostics.
- Placeholder scan: No `TBD`, unscoped TODOs, or “write tests later” placeholders remain.
- Type consistency: The helper returns `(query_hash, query_length)` and both endpoints consume those names consistently.
