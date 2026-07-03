# Testing Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all 10 findings in `audits/2026-07-02-testing-implementation-audit.md` (coverage gates, untested routes, quarantined suites, sleeps, error-path skew, rate-limit tests, skip hygiene, fuzzing, frontend coverage, perf nightly).

**Architecture:** All changes are test/CI-layer only — no production code changes. One branch (`feat/testing-audit-remediation` off `main`), one commit per task, one PR.

**Tech Stack:** pytest (plugin autoload disabled — always pass `-p` flags shown), coverage.py, hypothesis, Vitest v4, GitHub Actions.

## Global Constraints

- Python: always use `.venv/bin/python`, never system python (system is 3.14, too new).
- Every pytest invocation needs: `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1` env and explicit `-p` plugin flags as shown per step.
- Frontend: bun workspace root is `apps/`; run bun commands from the specific package dir.
- Do not modify production code under `tldw_Server_API/app/` — test/CI/config files only.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Measured baselines (2026-07-02, this machine): gated backend coverage 13.39%; excluded suites: Character_Chat_NEW 68F/408P, TTS_NEW 325F/309P, Embeddings 192F/230P; frontend local run has 320 pre-existing failures (environment-dependent — do NOT gate frontend coverage on thresholds in this PR).

---

### Task 1: Branch setup + commit corrected audit report

**Files:**
- Create branch: `feat/testing-audit-remediation` off `main`
- Add: `audits/2026-07-02-testing-implementation-audit.md` (already exists, untracked)

**Interfaces:**
- Produces: the branch all later tasks commit to.

- [ ] **Step 1: Create branch**

```bash
git checkout main && git pull && git checkout -b feat/testing-audit-remediation
```

- [ ] **Step 2: Commit the audit report**

```bash
git add audits/2026-07-02-testing-implementation-audit.md
git commit -m "docs(audits): add testing implementation audit (2026-07-02)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: F1 — Raise backend coverage floor + AuthNZ scoped gate

**Files:**
- Modify: `.github/workflows/coverage-required.yml:76-79`

**Interfaces:**
- Consumes: nothing.
- Produces: CI gate `coverage-required` with floor 12; second step "AuthNZ coverage floor".

- [ ] **Step 1: Verify current measured coverage still exceeds the new floor**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest -q --disable-warnings -p pytest_cov -p pytest_asyncio.plugin -p no:cacheprovider \
  -m "not jobs and not e2e" tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
  --cov=tldw_Server_API --cov-report=term 2>/dev/null | grep TOTAL
```
Expected: `TOTAL ... 13%` (≥ 13). If below 12, stop and report — do not lower the floor to fit.

- [ ] **Step 2: Confirm AuthNZ scoped coverage baseline (measured 2026-07-02: 38.79%, 87 passed, ~3min)**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_SQLite -q -p no:cacheprovider \
  -p pytest_cov -p pytest_asyncio.plugin -p timeout --timeout=120 \
  --cov=tldw_Server_API/app/core/AuthNZ --cov-report=term 2>/dev/null | grep TOTAL
```
Expected: TOTAL ≈ 38-39%. The gate value in Step 3 is **35**. If TOTAL measures below 36, use measured − 3 instead and say so in the commit message.

- [ ] **Step 3: Edit the workflow**

In `.github/workflows/coverage-required.yml`, change line 79's `--cov-fail-under=5` to `--cov-fail-under=12`, and append this step after the "Run global coverage floor" step (same indentation level):

```yaml
      - name: AuthNZ coverage floor
        if: needs.changes.outputs.coverage_required == 'true'
        env:
          PYTHONPATH: .
          PYTEST_DISABLE_PLUGIN_AUTOLOAD: "1"
          TEST_MODE: "true"
          DISABLE_HEAVY_STARTUP: "1"
        run: |
          pytest -q --disable-warnings -p pytest_cov -p pytest_asyncio.plugin -p timeout --timeout=120 \
            tldw_Server_API/tests/AuthNZ_SQLite \
            --cov=tldw_Server_API/app/core/AuthNZ --cov-report=term --cov-fail-under=35
```

- [ ] **Step 4: Lint the workflow**

```bash
.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/coverage-required.yml'))" && echo YAML-OK
```
Expected: `YAML-OK`

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/coverage-required.yml
git commit -m "ci(coverage): raise global floor 5->12, add AuthNZ scoped gate (audit F1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: F3 — Replace norecursedirs exclusion with visible quarantine

**Files:**
- Modify: `pyproject.toml:614-618` (delete `norecursedirs` block)
- Create: `tldw_Server_API/tests/Character_Chat_NEW/conftest.py` — NOTE: check first whether it exists; if so, append the hook instead of creating
- Create/modify: `tldw_Server_API/tests/TTS_NEW/conftest.py` (same note)
- Create/modify: `tldw_Server_API/tests/Embeddings/conftest.py` (same note)
- Create: `audits/2026-07-02-quarantined-suites.md`

**Interfaces:**
- Produces: env var `RUN_QUARANTINED=1` opt-in; default runs show these tests as skipped with reason, not hidden.

- [ ] **Step 1: Check for existing conftest files**

```bash
ls tldw_Server_API/tests/Character_Chat_NEW/conftest.py tldw_Server_API/tests/TTS_NEW/conftest.py tldw_Server_API/tests/Embeddings/conftest.py 2>&1
```

- [ ] **Step 2: Add the quarantine hook to each of the three conftest files** (append if the file exists, create otherwise). Identical block in all three:

```python
import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    """Quarantine: this suite has known failures (see audits/2026-07-02-quarantined-suites.md).

    Skipped by default so it is VISIBLE in every run instead of hidden via
    norecursedirs. Run for real with RUN_QUARANTINED=1.

    Scoped to this conftest's own directory: pytest_collection_modifyitems
    receives the FULL session item list even in a subdirectory conftest, so
    an unscoped loop would skip every collected test in the whole run, not
    just this suite.
    """
    if os.getenv("RUN_QUARANTINED") == "1":
        return
    here = Path(__file__).resolve().parent
    skip = pytest.mark.skip(
        reason="quarantined: known-failing suite, run with RUN_QUARANTINED=1 "
        "(audits/2026-07-02-quarantined-suites.md)"
    )
    for item in items:
        try:
            item_path = Path(str(getattr(item, "path", None) or item.fspath)).resolve()
        except Exception:
            continue
        if here == item_path or here in item_path.parents:
            item.add_marker(skip)
```

- [ ] **Step 3: Delete the `norecursedirs` block from `pyproject.toml`** (lines 614-618, the whole `norecursedirs = [...]` assignment).

- [ ] **Step 4: Write the burn-down doc** `audits/2026-07-02-quarantined-suites.md`:

```markdown
# Quarantined Test Suites — burn-down tracker

These suites were hidden via `norecursedirs` until 2026-07-02; they are now
collected and skipped-with-reason by default (opt in: `RUN_QUARANTINED=1`).
Measured 2026-07-02 (60s per-test timeout):

| Suite | Failed | Passed | Skipped/xfail | Runtime |
|---|---|---|---|---|
| tests/Character_Chat_NEW | 68 | 408 | 4 | 6m45s |
| tests/TTS_NEW | 325 | 309 | 2 xfail | 10m24s |
| tests/Embeddings | 192 | 230 | 17 | 2m54s |

Exit criteria per suite: 0 failures with `RUN_QUARANTINED=1`, then delete the
quarantine hook from its conftest. Reproduce:

    RUN_QUARANTINED=1 PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true \
    DISABLE_HEAVY_STARTUP=1 .venv/bin/python -m pytest tldw_Server_API/tests/<suite> -q \
    -p no:cacheprovider -p timeout --timeout=60
```

- [ ] **Step 5: Verify default collection now skips instead of hiding**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings -q -p no:cacheprovider 2>/dev/null | tail -2
```
Expected: `439 skipped` (or close), 0 failed.

- [ ] **Step 6: Verify the CI-gated scope is unaffected**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest -q -p pytest_asyncio.plugin -p no:cacheprovider \
  -m "not jobs and not e2e" tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests 2>/dev/null | tail -1
```
Expected: `258 passed` (same as baseline).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tldw_Server_API/tests/Character_Chat_NEW/conftest.py \
  tldw_Server_API/tests/TTS_NEW/conftest.py tldw_Server_API/tests/Embeddings/conftest.py \
  audits/2026-07-02-quarantined-suites.md
git commit -m "test: replace norecursedirs hiding with visible quarantine skip (audit F3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: F2 — Tests for untested storage routes

**Files:**
- Create: `tldw_Server_API/tests/Storage/test_storage_user_routes.py`

**Interfaces:**
- Consumes: fixtures `mock_user`, `mock_storage_service`, `mock_files_repo` from `tldw_Server_API/tests/Storage/conftest.py` (mock defaults: `get_file_by_id → None`, `get_user_folders → []`). House pattern: `monkeypatch.setattr(storage_endpoints, "_get_service", AsyncMock(...))` + local FastAPI app with `include_router(storage_endpoints.router, prefix="/api/v1")` and `get_request_user` override (see `test_storage_endpoints.py:27-32,401-405`).
- Produces: coverage for `POST /storage/folders`, `PATCH+DELETE /storage/files/{id}`, `POST /storage/trash/restore/{id}`, `DELETE /storage/trash/{id}`.

- [ ] **Step 1: Write the test file** `tldw_Server_API/tests/Storage/test_storage_user_routes.py`:

```python
"""Route-level tests for user storage folders/files/trash (audit F2)."""
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import storage as storage_endpoints


def _app(mock_user) -> FastAPI:
    app = FastAPI()
    app.include_router(storage_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[storage_endpoints.get_request_user] = lambda: mock_user
    return app


@pytest.fixture
def client(mock_user, mock_storage_service, monkeypatch):
    monkeypatch.setattr(
        storage_endpoints, "_get_service", AsyncMock(return_value=mock_storage_service)
    )
    with TestClient(_app(mock_user)) as c:
        yield c


class TestFolderRoutes:
    @pytest.mark.unit
    def test_create_folder_returns_normalized_tag(self, client):
        resp = client.post("/api/v1/storage/folders", json={"name": "reports"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["folder_tag"] == "reports"

    @pytest.mark.unit
    def test_create_folder_empty_name_is_422(self, client):
        # FolderCreateRequest enforces min_length=1 at schema level
        resp = client.post("/api/v1/storage/folders", json={"name": ""})
        assert resp.status_code == 422

    @pytest.mark.unit
    def test_list_folders_empty(self, client):
        resp = client.get("/api/v1/storage/folders")
        assert resp.status_code == 200
        assert resp.json()["folders"] == []


class TestFileMutationRoutes:
    @pytest.mark.unit
    def test_patch_unknown_file_is_404(self, client):
        resp = client.patch("/api/v1/storage/files/999999", json={"folder_tag": "archive"})
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_delete_unknown_file_is_404(self, client):
        resp = client.delete("/api/v1/storage/files/999999")
        assert resp.status_code == 404


class TestTrashMutationRoutes:
    @pytest.mark.unit
    def test_restore_unknown_item_is_404(self, client):
        resp = client.post("/api/v1/storage/trash/restore/999999")
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_restore_other_users_file_is_403(self, client, mock_files_repo):
        mock_files_repo.get_file_by_id = AsyncMock(
            return_value={"id": 5, "user_id": 999, "is_deleted": True}
        )
        resp = client.post("/api/v1/storage/trash/restore/5")
        assert resp.status_code == 403

    @pytest.mark.unit
    def test_restore_not_deleted_file_is_400(self, client, mock_files_repo):
        mock_files_repo.get_file_by_id = AsyncMock(
            return_value={"id": 5, "user_id": 1, "is_deleted": False}
        )
        resp = client.post("/api/v1/storage/trash/restore/5")
        assert resp.status_code == 400

    @pytest.mark.unit
    def test_permanent_delete_unknown_item_is_404(self, client):
        resp = client.delete("/api/v1/storage/trash/999999")
        assert resp.status_code == 404

    @pytest.mark.unit
    def test_permanent_delete_other_users_file_is_403(self, client, mock_files_repo):
        mock_files_repo.get_file_by_id = AsyncMock(
            return_value={"id": 5, "user_id": 999, "is_deleted": True}
        )
        resp = client.delete("/api/v1/storage/trash/5")
        assert resp.status_code == 403
```

Contract sources: `storage_trash.py:56-105` (404/403/400 mapping), `storage_user_folders.py:46-62` (virtual folder create), `storage_user_files.py:207,238`. Note: `mock_files_repo` is wired into `mock_storage_service` by the conftest, so overriding `mock_files_repo.get_file_by_id` flows through. If PATCH/DELETE file routes map missing files to a different status (e.g. via repo `update_file` returning None → check `storage_user_files.py:207-260` at implementation time), assert the actual documented mapping — do not weaken to `in (400, 404)`.

- [ ] **Step 2: Run the new tests**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/Storage/test_storage_user_routes.py -v \
  -p pytest_asyncio.plugin -p no:cacheprovider
```
Expected: 10 passed. If any fail on status code, read the route implementation, fix the assertion to the real contract, and note it in the commit message.

- [ ] **Step 3: Commit**

```bash
git add tldw_Server_API/tests/Storage/test_storage_user_routes.py
git commit -m "test(storage): cover folder create, file patch/delete, trash mutations (audit F2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: F6 — Error-path sweep for top-10 endpoints

**Files:**
- Create: `tldw_Server_API/tests/integration/test_error_paths_top_endpoints.py`

**Interfaces:**
- Consumes: `client_with_single_user` / `client_user_only` fixtures (root `tldw_Server_API/tests/conftest.py:818-886`) for authenticated cases; a bare `TestClient(app)` (no override, no API key) for 401 cases.
- Produces: ~30 error-path assertions (401 unauthenticated + 422 malformed body) across the highest-traffic endpoints.

- [ ] **Step 1: Write the test file** `tldw_Server_API/tests/integration/test_error_paths_top_endpoints.py`:

```python
"""Error-path sweep: unauthenticated and malformed-body cases for top endpoints (audit F6)."""
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

# (method, path, minimal-but-malformed JSON body or None for GET)
PROTECTED_ROUTES = [
    ("POST", "/api/v1/chat/completions", {"model": 123}),          # model must be str
    ("POST", "/api/v1/embeddings", {"input": None}),               # input required
    ("POST", "/api/v1/rag/search", {"query": None}),               # query required
    ("GET", "/api/v1/media/search", None),
    ("POST", "/api/v1/audio/transcriptions", {}),                  # multipart file required
    ("POST", "/api/v1/audio/speech", {"input": 42}),               # input must be str
    ("GET", "/api/v1/notes/", None),
    ("GET", "/api/v1/prompts/", None),
    ("GET", "/api/v1/characters/", None),
    ("GET", "/api/v1/mcp/status", None),
]


@pytest.fixture(scope="module")
def anon_client():
    """Client with NO auth override and NO API key: every request is anonymous."""
    from tldw_Server_API.app.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES,
                         ids=[f"{m}-{p}" for m, p, _ in PROTECTED_ROUTES])
def test_unauthenticated_request_is_rejected(anon_client, method, path, body):
    resp = anon_client.request(method, path, json=body)
    assert resp.status_code in (401, 403), (
        f"{method} {path} returned {resp.status_code}; expected auth rejection. "
        f"404 means the path is wrong - fix it from /openapi.json, do not delete the case."
    )


MALFORMED_BODY_ROUTES = [(m, p, b) for m, p, b in PROTECTED_ROUTES if b is not None]


@pytest.mark.parametrize("method,path,body", MALFORMED_BODY_ROUTES,
                         ids=[f"{m}-{p}" for m, p, _ in MALFORMED_BODY_ROUTES])
def test_malformed_body_is_422_not_500(client_user_only, method, path, body):
    resp = client_user_only.request(method, path, json=body)
    assert 400 <= resp.status_code < 500, (
        f"{method} {path} returned {resp.status_code} for malformed input; "
        f"validation errors must never be 5xx."
    )
```

- [ ] **Step 2: Run and correct any wrong paths**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/integration/test_error_paths_top_endpoints.py -v \
  -p pytest_asyncio.plugin -p no:cacheprovider 2>&1 | tail -25
```
Expected: all pass. A 404 in the unauthenticated test means a wrong path: dump routes with
```bash
PYTHONPATH=. TEST_MODE=true DISABLE_HEAVY_STARTUP=1 .venv/bin/python -c "
from tldw_Server_API.app.main import app
for r in app.routes:
    if hasattr(r, 'methods'): print(sorted(r.methods), r.path)" | grep -iE "chat|embed|rag|media|audio|notes|prompts|character|mcp" | head -40
```
and fix the entry (keep 10 routes; substitute a neighboring real route if one was removed).

- [ ] **Step 3: Commit**

```bash
git add tldw_Server_API/tests/integration/test_error_paths_top_endpoints.py
git commit -m "test: error-path sweep (401/422) across top-10 endpoints (audit F6)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: F7 — Dedicated rate-limiting suite

**Files:**
- Create: `tldw_Server_API/tests/RateLimiting/__init__.py` (empty)
- Create: `tldw_Server_API/tests/RateLimiting/test_character_rate_limiter_429.py`

**Interfaces:**
- Consumes: `CharacterRateLimiter.check_rate_limit(user_id, operation)` from `tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py:139-182` — raises `HTTPException(429, headers={"Retry-After": ...})` when the ResourceGovernor decision is `{"allowed": False, "retry_after": N}`; module-level seams `_rg_character_enabled`, `_rg_character_enforce_requests`, `_maybe_enforce_with_rg_character`.
- Produces: first tests asserting limits FIRE (429 + `Retry-After`), complementing the suppression fixture in `tests/Chat/conftest.py:21-46`.

- [ ] **Step 1: Write the test file** `tldw_Server_API/tests/RateLimiting/test_character_rate_limiter_429.py`:

```python
"""Assert rate limits actually fire: 429 + Retry-After (audit F7)."""
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Character_Chat import character_rate_limiter as crl


@pytest.fixture
def limiter(monkeypatch):
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setattr(crl, "_rg_character_enabled", lambda: True)
    monkeypatch.setattr(crl, "_rg_character_enforce_requests", lambda: True)
    lim = crl.CharacterRateLimiter()
    lim.enabled = True
    return lim


@pytest.mark.unit
async def test_denied_decision_raises_429_with_retry_after(limiter, monkeypatch):
    monkeypatch.setattr(
        crl, "_maybe_enforce_with_rg_character",
        AsyncMock(return_value={"allowed": False, "retry_after": 7, "policy_id": "p1"}),
    )
    with pytest.raises(HTTPException) as exc:
        await limiter.check_rate_limit(user_id=1, operation="character_op")
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "7"


@pytest.mark.unit
async def test_denied_decision_without_retry_after_defaults_to_60(limiter, monkeypatch):
    monkeypatch.setattr(
        crl, "_maybe_enforce_with_rg_character",
        AsyncMock(return_value={"allowed": False, "policy_id": "p1"}),
    )
    with pytest.raises(HTTPException) as exc:
        await limiter.check_rate_limit(user_id=1)
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "60"


@pytest.mark.unit
async def test_allowed_decision_passes(limiter, monkeypatch):
    monkeypatch.setattr(
        crl, "_maybe_enforce_with_rg_character",
        AsyncMock(return_value={"allowed": True}),
    )
    allowed, _ = await limiter.check_rate_limit(user_id=1)
    assert allowed is True


@pytest.mark.unit
async def test_unavailable_governor_fails_open(limiter, monkeypatch):
    monkeypatch.setattr(
        crl, "_maybe_enforce_with_rg_character", AsyncMock(return_value=None)
    )
    allowed, _ = await limiter.check_rate_limit(user_id=1)
    assert allowed is True  # documented fail-open behavior (crl.py:159-163)


@pytest.mark.unit
async def test_disabled_limiter_short_circuits(monkeypatch):
    monkeypatch.setenv("CHARACTER_RATE_LIMIT_ENABLED", "false")
    lim = crl.CharacterRateLimiter()
    lim.enabled = False
    allowed, _ = await lim.check_rate_limit(user_id=1)
    assert allowed is True
```

Note: `asyncio_mode = "auto"` in pyproject, so bare `async def` tests run without a marker. If `CharacterRateLimiter()`'s constructor signature differs (check `character_rate_limiter.py` top of class), adapt construction only — the seam monkeypatches are the point.

- [ ] **Step 2: Run**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/RateLimiting/ -v -p pytest_asyncio.plugin -p no:cacheprovider
```
Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add tldw_Server_API/tests/RateLimiting/
git commit -m "test(rate-limiting): dedicated suite asserting 429 + Retry-After fire (audit F7)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: F5 — Remove/shrink sleeps in Evaluations tests

**Files:**
- Modify: `tldw_Server_API/tests/Evaluations/test_error_scenarios.py:110` (the `time.sleep(10)`)
- Modify: `tldw_Server_API/tests/Evaluations/unit/test_circuit_breaker.py` (sleeps at lines 96, 381, 426 and any other `time.sleep` in the file)

**Interfaces:**
- Consumes: `CircuitBreaker` recovery check uses `time.time()` (`app/core/Evaluations/circuit_breaker.py:163-167`), so faking `time.time` controls recovery without real waiting.
- Produces: same tests, ~13s faster, no wall-clock dependence.

- [ ] **Step 1: Shrink the network-timeout simulation.** In `test_error_scenarios.py:110`, the mock blocks 10s to trip a 0.1s `asyncio.wait_for`. Change:

```python
            time.sleep(10)
```
to:
```python
            time.sleep(0.5)  # must exceed the 0.1s wait_for timeout below; 10s wasted wall-clock
```

- [ ] **Step 2: Add a fake-clock fixture to `test_circuit_breaker.py`** (top of file, after imports):

```python
class _FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr("time.time", clock.time)
    return clock
```

- [ ] **Step 3: Replace each recovery-wait sleep.** For every test in the file that calls `time.sleep(X)` to wait out a recovery timeout (lines 96, 381, 426 — re-grep, line numbers may drift):
  1. add `fake_clock` to the test's parameters,
  2. replace `time.sleep(X)` with `fake_clock.advance(X)`.

  Sub-second sleeps used for thread interleaving (not recovery timing) stay as-is.

- [ ] **Step 4: Run both files; compare wall-clock**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/unit/test_circuit_breaker.py \
  tldw_Server_API/tests/Evaluations/test_error_scenarios.py -q -p pytest_asyncio.plugin -p no:cacheprovider
```
Expected: all pass, total time noticeably below the prior run. If a breaker test fails with the fake clock, that test's code path reads a different time source (e.g. `time.monotonic`) — patch that name instead for that test; do NOT reinstate the sleep.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/tests/Evaluations/unit/test_circuit_breaker.py tldw_Server_API/tests/Evaluations/test_error_scenarios.py
git commit -m "test(evals): fake clock for circuit-breaker waits; shrink 10s sleep to 0.5s (audit F5)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: F9 — Skip-marker hygiene: meta-test + triage

**Files:**
- Create: `tldw_Server_API/tests/CI/test_skip_markers_have_reasons.py`
- Modify: every test file flagged by the meta-test (bare `pytest.mark.skip` without `reason=`; ~40 sites)

**Interfaces:**
- Produces: enforced rule — every unconditional `pytest.mark.skip` carries a `reason=`.

- [ ] **Step 1: Write the meta-test** `tldw_Server_API/tests/CI/test_skip_markers_have_reasons.py`:

```python
"""Every unconditional skip must say why (audit F9)."""
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_all_unconditional_skips_have_reasons():
    offenders: list[str] = []
    for path in TESTS_ROOT.rglob("test_*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if "pytest.mark.skip" not in line or "skipif" in line:
                continue
            # OK forms: skip(reason=...) on this line, or continuation with reason nearby
            window = "\n".join(text.splitlines()[i - 1 : i + 2])
            if "reason" not in window:
                offenders.append(f"{path.relative_to(TESTS_ROOT)}:{i}: {line.strip()}")
    assert not offenders, (
        "Unconditional pytest.mark.skip without reason= (add one, "
        "e.g. reason='needs X, see #issue'):\n" + "\n".join(offenders)
    )
```

- [ ] **Step 2: Run it to enumerate offenders**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/CI/test_skip_markers_have_reasons.py -v -p no:cacheprovider 2>&1 | tail -50
```
Expected: FAIL, listing every bare-skip site.

- [ ] **Step 3: Fix each offender.** For each listed `file:line`, read the surrounding test, and add a specific `reason=` string stating what's missing or broken (service dependency, refactor pending, platform limitation). If the skip is clearly dead (the skipped feature shipped or was removed), delete the marker and let the test run; if that test then fails, restore the skip with `reason="broken: <symptom>"`.

- [ ] **Step 4: Re-run to green**

Same command as Step 2. Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/tests/CI/test_skip_markers_have_reasons.py tldw_Server_API/tests/
git commit -m "test: require reason= on unconditional skips; triage existing 40 (audit F9)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: F10 — Hypothesis fuzzing for input surfaces

**Files:**
- Create: `tldw_Server_API/tests/unit/test_sanitize_filename_fuzz.py`
- Create: `tldw_Server_API/tests/unit/test_chunker_fuzz.py`

**Interfaces:**
- Consumes: `sanitize_filename(filename, *, max_total_length=None, extension=None)` (`app/core/Utils/Utils.py:680-722`; strips `< > : " / \ | ? *`, returns "untitled" for empty/dot results); `Chunker.chunk_text(text, method=None, max_size=None, overlap=None, ...) -> list[str]` (`app/core/Chunking/chunker.py:1447-1596`; raises `InvalidInputError` on bad input, returns `[]` on empty). House hypothesis style: `@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)` (see `tests/Embeddings/test_message_validator_fuzz.py`).
- Produces: property tests in the CI-gated `tests/unit` scope (they run under the Task 2 coverage gate).

- [ ] **Step 1: Write** `tldw_Server_API/tests/unit/test_sanitize_filename_fuzz.py`:

```python
"""Property-based tests for sanitize_filename (audit F10)."""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from tldw_Server_API.app.core.Utils.Utils import sanitize_filename

FORBIDDEN = set('<>:"/\\|?*')


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
@given(st.text(max_size=500))
def test_never_raises_and_never_empty(raw):
    out = sanitize_filename(raw)
    assert isinstance(out, str)
    assert out  # never empty: falls back to "untitled"


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
@given(st.text(max_size=500))
def test_output_contains_no_forbidden_characters(raw):
    out = sanitize_filename(raw)
    assert not (set(out) & FORBIDDEN), f"forbidden chars survived: {set(out) & FORBIDDEN}"


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
@given(st.text(min_size=1, max_size=300), st.integers(min_value=10, max_value=100))
def test_length_cap_is_respected(raw, cap):
    out = sanitize_filename(raw, max_total_length=cap, extension=".txt")
    assert len(out) + len(".txt") <= cap or out == "untitled"
```

- [ ] **Step 2: Write** `tldw_Server_API/tests/unit/test_chunker_fuzz.py`:

```python
"""Property-based tests for Chunker.chunk_text (audit F10)."""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from tldw_Server_API.app.core.Chunking.chunker import Chunker


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
@given(
    text=st.text(max_size=5_000),
    max_size=st.integers(min_value=16, max_value=1_024),
    overlap=st.integers(min_value=0, max_value=15),
)
def test_chunk_text_returns_list_of_strings(text, max_size, overlap):
    chunks = Chunker().chunk_text(text, method="words", max_size=max_size, overlap=overlap)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
@given(max_size=st.integers(min_value=16, max_value=1_024))
def test_empty_input_yields_no_chunks(max_size):
    assert Chunker().chunk_text("", method="words", max_size=max_size, overlap=0) == []


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50, deadline=None)
@given(text=st.text(min_size=1, max_size=5_000).filter(lambda s: s.strip()))
def test_nonempty_input_content_is_preserved_in_chunks(text):
    chunks = Chunker().chunk_text(text, method="words", max_size=64, overlap=0)
    joined = " ".join(chunks)
    for word in text.split()[:5]:
        assert word in joined
```

- [ ] **Step 3: Run both**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/unit/test_sanitize_filename_fuzz.py \
  tldw_Server_API/tests/unit/test_chunker_fuzz.py -v -p pytest_asyncio.plugin -p no:cacheprovider
```
Expected: 6 passed. Hypothesis may legitimately find a real bug (e.g. a unicode edge in sanitize_filename) — if so, that is a FINDING: capture the failing example in the commit message and mark that one test `xfail(strict=False, reason="real bug found: <example>")` rather than weakening the property; file it in the PR description.

- [ ] **Step 4: If `Chunker` construction requires config** (check by running Step 3; a TypeError on `Chunker()` means it needs args), read `chunker.py`'s `__init__` and construct with its documented defaults. Adjust only construction.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/tests/unit/test_sanitize_filename_fuzz.py tldw_Server_API/tests/unit/test_chunker_fuzz.py
git commit -m "test: hypothesis fuzzing for sanitize_filename and chunk_text (audit F10)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: F4 — Frontend coverage: config + report-only CI

**Files:**
- Modify: `apps/packages/ui/package.json` (add devDependency)
- Modify: `apps/packages/ui/vitest.config.ts` (coverage block)
- Modify: `apps/tldw-frontend/vitest.config.ts` (coverage block)
- Modify: `.github/workflows/frontend-required.yml` (report-only coverage summary step)

**Interfaces:**
- Produces: `bun run test:coverage` works in both packages; CI prints a coverage summary (NOT gated — 320 pre-existing local failures make thresholds unsafe this PR; ratchet later from CI-observed numbers).

- [ ] **Step 1: Add the coverage provider to packages/ui**

```bash
cd apps/packages/ui && bun add -d @vitest/coverage-v8@4.0.18
```
(Version pinned to match `apps/tldw-frontend`'s existing `@vitest/coverage-v8@4.0.18`.)

- [ ] **Step 2: Add a `coverage` block to `apps/packages/ui/vitest.config.ts`** inside the existing `test: {...}` object:

```ts
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "json-summary"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/__tests__/**", "src/**/*.d.ts"],
    },
```

- [ ] **Step 3: Same block in `apps/tldw-frontend/vitest.config.ts`** with its include set:

```ts
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "json-summary"],
      include: ["components/**/*.{ts,tsx}", "services/**/*.{ts,tsx}", "store/**/*.{ts,tsx}", "utils/**/*.{ts,tsx}"],
      exclude: ["**/__tests__/**", "**/*.d.ts"],
    },
```
(Verify those four dirs exist — `ls apps/tldw-frontend` — and substitute the app's actual source dirs if named differently, e.g. `src/**`.)

- [ ] **Step 4: Verify locally that coverage now produces a summary** (UI package; failures are OK, summary must print)

```bash
cd apps/packages/ui && bun run vitest run --coverage src/components/Common 2>&1 | grep -A4 "Coverage"
```
Expected: a `% Coverage report` / text-summary block. (Scoped to one dir to keep it fast.)

- [ ] **Step 5: Add a report-only step to `.github/workflows/frontend-required.yml`** after the unit-test step (adjust indentation to the file's step list):

```yaml
      - name: Frontend coverage summary (report-only)
        continue-on-error: true
        working-directory: apps/tldw-frontend
        run: |
          bun run test:coverage 2>&1 | tail -30
          echo "NOTE: report-only; thresholds land after a baseline is observed in CI (audit F4)."
```

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/package.json apps/packages/ui/bun.lock apps/packages/ui/vitest.config.ts \
  apps/tldw-frontend/vitest.config.ts .github/workflows/frontend-required.yml
git commit -m "ci(frontend): wire vitest v8 coverage + report-only CI summary (audit F4)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
(If `bun add` modified `apps/bun.lock` at the workspace root instead, `git add apps/bun.lock`.)

---

### Task 11: F8 — Nightly performance workflow

**Files:**
- Create: `.github/workflows/perf-nightly.yml`

**Interfaces:**
- Consumes: existing perf tests in `tldw_Server_API/tests/performance/` (3 files) and the `performance` marker; shared action `.github/actions/setup-python-deps` (pattern from `jobs-suite.yml`).
- Produces: scheduled non-gating perf run with uploaded artifact.

- [ ] **Step 1: Write `.github/workflows/perf-nightly.yml`:**

```yaml
name: perf-nightly

on:
  schedule:
    - cron: "0 6 * * *"  # Daily at 06:00 UTC
  workflow_dispatch:

permissions:
  contents: read

jobs:
  perf:
    name: performance-tests
    runs-on: ubuntu-latest
    timeout-minutes: 45
    env:
      PYTHONPATH: .
      PYTEST_DISABLE_PLUGIN_AUTOLOAD: "1"
      TEST_MODE: "true"
      DISABLE_HEAVY_STARTUP: "1"
    steps:
      - name: Checkout
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd

      - name: Setup Python and dependencies
        uses: ./.github/actions/setup-python-deps
        with:
          python-version: "3.12"
          use-uv: "true"
          cache-dependency-path: |
            pyproject.toml
            uv.lock
          extras: dev

      - name: Run performance suite
        run: |
          pytest -q --disable-warnings -p pytest_asyncio.plugin -p timeout --timeout=300 \
            tldw_Server_API/tests/performance \
            -m "not jobs and not e2e" \
            --junitxml=perf-results.xml

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: perf-results
          path: perf-results.xml
          retention-days: 30
```

- [ ] **Step 2: Validate YAML + verify the perf suite passes locally**

```bash
.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/perf-nightly.yml'))" && echo YAML-OK
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest tldw_Server_API/tests/performance -q -p pytest_asyncio.plugin -p timeout --timeout=300 -p no:cacheprovider 2>/dev/null | tail -2
```
Expected: `YAML-OK` and the 3 perf files pass (if one requires unavailable local services, note the skip in the commit message).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/perf-nightly.yml
git commit -m "ci(perf): nightly non-gating performance run with artifact (audit F8)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Final verification + PR

**Files:** none new.

- [ ] **Step 1: Full gated-scope run with new floor**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest -q --disable-warnings -p pytest_cov -p pytest_asyncio.plugin -p no:cacheprovider \
  -m "not jobs and not e2e" tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
  --cov=tldw_Server_API --cov-report=term --cov-fail-under=12 2>/dev/null | tail -3
```
Expected: pass (the Task 9 fuzz tests are in `tests/unit`, so count > 258 now).

- [ ] **Step 2: Run every suite this PR touched**

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest -q -p pytest_asyncio.plugin -p no:cacheprovider \
  tldw_Server_API/tests/Storage/test_storage_user_routes.py \
  tldw_Server_API/tests/integration/test_error_paths_top_endpoints.py \
  tldw_Server_API/tests/RateLimiting/ \
  tldw_Server_API/tests/Evaluations/unit/test_circuit_breaker.py \
  tldw_Server_API/tests/CI/test_skip_markers_have_reasons.py 2>/dev/null | tail -2
```
Expected: all pass, 0 failed.

- [ ] **Step 3: Local CI lane (repo's own gate runner)**

```bash
make ci-local
```
Expected: green (matches what `backend-required` will run).

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/testing-audit-remediation
gh pr create --base main --title "test/ci: remediate 2026-07-02 testing audit (F1-F10)" --body "$(cat <<'EOF'
Remediates all 10 findings from audits/2026-07-02-testing-implementation-audit.md:

- F1: coverage floor 5% -> 12% (measured actual: 13.39%) + AuthNZ scoped gate
- F2: route tests for storage folders/files/trash mutations
- F3: norecursedirs-hidden suites (585 known failures) now visibly quarantined via skip-with-reason; burn-down doc in audits/
- F4: frontend Vitest v8 coverage wired, report-only CI summary (thresholds after CI baseline)
- F5: fake clock replaces circuit-breaker sleeps; 10s sleep -> 0.5s
- F6: 401/422 error-path sweep across top-10 endpoints
- F7: dedicated RateLimiting suite asserting 429 + Retry-After
- F8: nightly non-gating perf workflow with artifact
- F9: meta-test requiring reason= on unconditional skips; existing 40 triaged
- F10: hypothesis fuzzing for sanitize_filename + chunk_text

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
