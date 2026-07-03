# Testing Implementation Audit — tldw_server

**Date:** 2026-07-02
**Scope:** Backend pytest suite (`tldw_Server_API/tests/`), frontend Vitest/Playwright suites (`apps/`), CI gates (`.github/workflows/`)
**Method:** Static analysis of test code and CI configuration, plus one live measurement: the exact `coverage-required` CI command was reproduced locally (all 258 tests passed, 65s). Every finding lists a reproduction command.

---

## Executive Summary

The project has an unusually large and well-organized test estate — the problem is not test *quantity*, it is **enforcement and visibility**. The single biggest issue: only 258 of ~30,000 backend tests run under coverage measurement, and the enforced floor is 5%.

| Metric | Value | Source |
|---|---|---|
| Backend test files | ~3,900 | `find tldw_Server_API/tests -name "test_*.py" \| wc -l` |
| Backend test functions | ~30,600 | `grep -rc "def test_" tldw_Server_API/tests` |
| Backend test directories | 155+ | `ls tldw_Server_API/tests` |
| Registered pytest markers | 42 | `pyproject.toml:557-607` |
| conftest.py files / fixtures | 53 / 394 | grep counts |
| Frontend Vitest test files | ~2,700 (`.test.*` + `.spec.*`) | `find apps -name "*.test.*" -not -path "*/node_modules/*"` |
| Playwright E2E specs | 165 (frontend) + 127 (extension) | `apps/tldw-frontend/e2e/`, `apps/extension/tests/e2e/` |
| Dedicated a11y test files | 40+ (axe-core) | `apps/**/*accessibility*.test.tsx` |
| **Measured coverage (CI-gated scope)** | **13.39%** | see Finding F1 |
| **Enforced coverage floor** | **5%** | `.github/workflows/coverage-required.yml:79` |
| Whole-suite coverage | **Unable to verify** — never measured | see F1 |

**Overall grade: B−.** World-class breadth (security suite, a11y gates, tiered E2E, per-test Postgres isolation) undermined by a near-meaningless coverage gate, five untested endpoint files (two of them OAuth admin surfaces), and three test directories silently excluded from collection.

**Top 3 actions:**
1. Expand coverage measurement beyond `tests/unit` + `tests/sanity_tests` and ratchet the floor (F1).
2. Decide the fate of the three `norecursedirs`-excluded test directories — fix or delete, don't silently skip (F3).
3. Add tests for the untested storage routes (folder create, file PATCH/DELETE, trash mutations) (F2 — revised; see §1.3 correction).

---

## 1. Test Coverage

### 1.1 What is actually measured (F1)

The only required coverage gate is `.github/workflows/coverage-required.yml:76-79`:

```yaml
pytest -q --disable-warnings -p pytest_cov -p pytest_asyncio.plugin \
  -m "not jobs and not e2e" \
  tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
  --cov=tldw_Server_API --cov-report=xml --cov-report=term-missing --cov-fail-under=5
```

Reproducing this command locally on 2026-07-02 (Python 3.11.13, all 258 tests passed in 65s):

- **Total coverage: 13.39%** — 94,511 of 705,928 statements, across 3,883 measured files.
- The floor is 5%, i.e. **8 points of headroom** — coverage could drop by more than half before CI notices.
- The scope is 258 tests. The other ~30,000 tests run in `ci.yml` shards **without `--cov`**, so their coverage contribution is never recorded anywhere (no badge, no committed report, no artifact trend).

The one exception is `ci.yml:250-257`, which gates `tldw_Server_API/cli/wizard` at `--cov-fail-under=70` — proof the ratchet pattern already exists in this repo and just needs replication.

> **Unable to verify:** true whole-suite coverage. It is plausibly far higher than 13.39% given the suite size, but nothing measures it. Proving it requires a full-suite `--cov` run (hours): `make ci-local-full` with `--cov=tldw_Server_API --cov-report=json` appended, or a merged-`coverage combine` across CI shards.

**Reproduce:**
```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
.venv/bin/python -m pytest -q --disable-warnings -p pytest_cov -p pytest_asyncio.plugin \
  -m "not jobs and not e2e" tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
  --cov=tldw_Server_API --cov-report=term | tail -5
```

### 1.2 Integration & E2E presence

Healthy. `tests/` contains dedicated `integration/`, `e2e/`, `frontend_e2e/`, `server_e2e_tests/` trees; markers show ~4,515 `@pytest.mark.unit` vs ~1,034 `@pytest.mark.integration`. Frontend E2E is tiered in `apps/tldw-frontend/playwright.config.ts` (tier-1 critical → tier-5 specialized + `journeys`, retries=2 in CI, trace on first retry, screenshot/video on failure) and gated by `frontend-ux-gates.yml`, `e2e-smoke.yml`, `frontend-e2e-tiers.yml`.

### 1.3 Uncovered critical paths (F2)

> **Correction (2026-07-02, second pass):** the initial version of this finding claimed five endpoint files had zero tests, based on grepping tests for the module *filenames*. That was wrong: `slack_oauth_admin.py` / `discord_oauth_admin.py` are implementation modules whose routes are defined in `slack.py` / `discord.py`, and tests reference URL *paths*, not module names. Route-level verification shows the OAuth surfaces are well covered — `tests/Slack/test_slack_oauth_lifecycle.py` and `tests/Discord/test_discord_oauth_lifecycle.py` exercise `oauth/start`, `oauth/callback` (including forged-`state` → 4xx), and the `RequireRole("admin")` policy/installation routes, with non-admin 403 cases present.

The **genuine** route-level gaps (grep-verified against test URL paths):

| Route | Tests found | Risk |
|---|---|---|
| `POST /api/v1/storage/folders` (`storage_user_folders.py:46`) | 0 | folder-name validation untested |
| `PATCH /api/v1/storage/files/{id}` (`storage_user_files.py:238`) | 0 | metadata update (retention/expiry) untested |
| `DELETE /api/v1/storage/files/{id}` (`storage_user_files.py:207`) | 0 | soft/hard delete flag untested |
| `POST /api/v1/storage/trash/restore/{id}` (`storage_trash.py:56`) | 1 (thin) | restore semantics |
| `DELETE /api/v1/storage/trash/{id}` (`storage_trash.py:83`) | 1 (thin) | permanent delete |

**Reproduce:**
```bash
grep -rn 'post("/api/v1/storage/folders' tldw_Server_API/tests/ | wc -l   # 0
grep -rn 'patch("/api/v1/storage/files' tldw_Server_API/tests/ | wc -l   # 0
grep -rn 'delete("/api/v1/storage/files' tldw_Server_API/tests/ | wc -l  # 0
```

### 1.4 Frontend coverage is dark (F4)

`@vitest/coverage-v8@4.0.18` is installed and `test:coverage` scripts exist, but neither `apps/tldw-frontend/vitest.config.ts` nor `apps/packages/ui/vitest.config.ts` contains a `coverage` block, and `frontend-required.yml` runs `vitest run` (sometimes `--changed`) without `--coverage`. ~2,700 frontend test files produce no coverage signal at all.

---

## 2. Test Quality

### 2.1 Naming & structure — good

Sampled tests follow descriptive naming and clean Arrange-Act-Assert. Representative good example, `tldw_Server_API/tests/unit/test_moderation_user_override_validation.py:12-21`:

```python
@pytest.mark.unit
def test_set_user_override_rejects_invalid_action(tmp_path):
    svc = ModerationService()                                   # Arrange
    svc._user_overrides_path = str(tmp_path / "overrides.json")
    res = svc.set_user_override("user1", {"input_action": "blok"})  # Act
    assert res["ok"] is False                                   # Assert
    assert "invalid input_action" in (res.get("error") or "")
    assert res.get("error_type") == "validation"
```

Frontend tests use `@testing-library/react` with behavior-level queries (`screen.getByTestId`, `waitFor`) rather than implementation internals — e.g. `apps/packages/ui/src/entries/shared/__tests__/app-shell.splash.test.tsx`.

### 2.2 Test independence & fixtures — good architecture

- Root `tldw_Server_API/tests/conftest.py` pins a deterministic env baseline (`AUTH_MODE=single_user`, schedulers disabled, `MPLBACKEND=Agg`) and tracks leaked aiosqlite connections via weak references.
- `tests/_plugins/postgres.py` provisions **per-test scratch Postgres databases** with Docker auto-start and reachability checks; `tests/AuthNZ/conftest.py` builds per-test isolated schemas.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` with an explicit `plugins` list (`pyproject.toml:619-634`) prevents plugin-drift flakiness.
- `--strict-markers` is on (`pyproject.toml:636`).

### 2.3 Mock usage — appropriate, one gap

- `monkeypatch` dominates (~36k uses) with `unittest.mock.patch` (~2.2k) for object-level mocking — consistent and idiomatic.
- `mock_openai_server/` provides a real FastAPI OpenAI-compatible mock (chat/embeddings/streaming) so LLM tests never hit real providers.
- Frontend uses `vi.mock()`/`vi.hoisted()` at module scope; no MSW — acceptable since services are mocked at the import boundary, though MSW would exercise the actual fetch layer.
- **Gap:** no `respx`/`httpx_mock` — HTTP-client behavior (retries, timeouts, connection errors) is mocked by replacing functions rather than the wire layer, so client-level error handling is under-exercised.

### 2.4 Sleep-based timing (F5)

~292 `time.sleep` calls in backend tests. Worst offenders:

- `tests/Evaluations/test_error_scenarios.py` — includes a `time.sleep(10)` (10 wasted seconds every run)
- `tests/Evaluations/unit/test_circuit_breaker.py` — `time.sleep(1.1)`, `time.sleep(0.6)`, `time.sleep(0.15)` to wait out real breaker timeouts
- `tests/Evaluations/conftest.py` — `time.sleep(0.5)` in fixture setup

**Reproduce:** `grep -rn "time.sleep" tldw_Server_API/tests --include="*.py" | wc -l`

---

## 3. Test Patterns

### 3.1 Test pyramid — healthy

~4.5k unit-marked vs ~1k integration-marked vs ~292 Playwright E2E specs. Correct shape (unit ≫ integration > E2E). Frontend mirrors this: 2,700 component/unit tests vs 292 E2E specs.

### 3.2 Anti-patterns found

**(a) Disabled-by-exclusion test directories (F3).** `pyproject.toml:614-618`:

```toml
norecursedirs = [
    "tldw_Server_API/tests/Character_Chat_NEW",
    "tldw_Server_API/tests/TTS_NEW",
    "tldw_Server_API/tests/Embeddings"
]
```

Three entire test trees never run under a default `pytest` invocation. This is "disabling tests instead of fixing them" — explicitly forbidden by this repo's own CLAUDE.md ("NEVER: Disable tests instead of fixing them"). Whether these suites currently pass is **Unable to verify** (they'd need to be run: `pytest tldw_Server_API/tests/Embeddings -x -q --co` to at least check collection).

**(b) Rate limiting suppressed rather than tested (F7).** `tests/Chat/conftest.py:21-46` (autouse) sets `CHARACTER_RATE_LIMIT_OPS=1000000` to prevent incidental 429s. Pragmatic — but there is no dedicated `tests/RateLimiting/` suite asserting that limits *do* fire; rate-limit assertions are scattered (e.g. `tests/Evaluations/integration/test_rate_limits_endpoint.py`).

**(c) Skip/xfail debt (F9).** 247 markers (40 `skip`, 196 `skipif`, 11 `xfail`), unaudited. `skipif` on missing services is legitimate; the 40 unconditional skips were triaged 2026-07-02: all carry accurate same-line `reason=` text already, so the remaining value is enforcement (a meta-test now guards new reason-less skips).

**Reproduce:** `grep -rEn "pytest.mark.(skip|skipif|xfail)" tldw_Server_API/tests --include="*.py" | wc -l`

### 3.3 Brittleness & speed

- Global `timeout = 300` per test (`pyproject.toml:642`) is a generous ceiling that hides slow tests; the sleeps in §2.4 add real wall-clock.
- No `pytest-rerunfailures`; frontend Playwright uses `retries: 2` in CI (reasonable for E2E).
- Mitigation already present: `make ci-local` / `ci-local-full -n auto` lanes (Makefile:43-45) shard the suite locally.
- Stage-based frontend test naming (`ContentViewer.stage4.accessibility.test.tsx`) cleanly separates fast smoke from heavy a11y/perf stages.

---

## 4. Missing Tests

### 4.1 Error scenarios (F6)

Across the backend suite: 6,807 assertions on 2xx status codes vs 3,302 on 4xx/5xx (measured 2026-07-02) — a 2:1 happy-path skew. Systemically under-tested: DB-connection-failure paths, provider timeout/retry behavior, and 429 responses (see F7).

**Reproduce:**
```bash
grep -rE "status_code == 2" tldw_Server_API/tests --include="*.py" | wc -l
grep -rE "status_code == (4|5)" tldw_Server_API/tests --include="*.py" | wc -l
```

### 4.2 Security tests

A dedicated `tldw_Server_API/tests/Security/` suite exists (13 files, ~1,450 lines): path traversal (`test_mediawiki_security.py`), SSRF/egress (`test_egress.py`, `test_websearch_egress_guard.py`), RBAC (`test_text2sql_rbac_and_acl.py`). **Gaps:** no SQL-injection suite beyond text2sql; no XSS/input-sanitization sweep; and the two OAuth admin endpoints (F2) — the most token-sensitive surfaces — have zero tests including auth-bypass cases.

### 4.3 Performance tests (F8)

`tests/performance/` contains exactly 3 files (`test_authnz_multiuser_sqlite_load.py`, `test_http_client_perf.py`, `test_reading_service_perf.py`). `pytest-benchmark` 5.2.3 is installed and `bench`/`benchmark`/`load`/`stress` markers are registered, but no CI job runs them — no regression baseline exists. No locust/k6.

### 4.4 Edge cases / property-based (F10)

`hypothesis` is a dependency and `@pytest.mark.property` appears ~91 times (sanitization, config/schema validation) — but ingestion parsing and upload validation (the highest-entropy input surfaces) have no fuzzing.

---

## Findings Table

| ID | Finding | Importance (1–10) | Evidence | Remediation |
|---|---|---|---|---|
| F1 | Coverage gate floor is 5%; measured gated coverage 13.39%; ~30k tests run uncovered | **9** | `coverage-required.yml:79`; local run 2026-07-02 | R1 below |
| F2 | 3 storage routes untested (folder create, file PATCH/DELETE) + thin trash-mutation coverage — *revised down from "5 files zero tests"; see correction in §1.3* | **5** | route-path grep, §1.3 | R2 below |
| F3 | 3 test dirs excluded via `norecursedirs` — silently never run | **8** | `pyproject.toml:614-618` | R3 below |
| F4 | Frontend coverage installed but unconfigured/ungated | **6** | both `vitest.config.ts` files lack `coverage` key | R4 below |
| F5 | ~292 `time.sleep` calls; worst `sleep(10)` in Evaluations | **6** | §2.4 | R5 below |
| F6 | 2:1 happy-path vs error-path assertion skew | **5** | §4.1 | Require a 4xx case per new endpoint test (review checklist); backfill top-traffic endpoints first |
| F7 | Rate limiting suppressed in fixtures, no dedicated suite | **5** | `Chat/conftest.py:21-46` | Create `tests/RateLimiting/test_limits_fire.py` asserting 429 + `Retry-After` with a low-limit override fixture |
| F8 | Performance testing minimal, no CI baseline | **4** | `tests/performance/` (3 files) | Nightly (not PR-gating) job running `-m benchmark` with `pytest-benchmark --benchmark-compare-fail=mean:10%` |
| F9 | 247 skip/skipif/xfail markers unaudited | **4** | grep, §3.2c | DONE 2026-07-02: triage found 0 reason-less skips; meta-test `tests/CI/test_skip_markers_have_reasons.py` now enforces `reason=` |
| F10 | Hypothesis underused; no fuzzing of ingestion/upload validation | **3** | §4.4 | Add `@given` tests for chunker + upload validators (see R6) |

---

## Remediation Snippets

### R1 — Widen and ratchet the coverage gate (F1)

`.github/workflows/coverage-required.yml` — measured reality is 13.39%, so a 12% floor is immediately safe; add a scoped gate for AuthNZ (mirroring the existing `cli/wizard` 70% gate at `ci.yml:257`). *Measured 2026-07-02: `tests/AuthNZ_SQLite` covers 38.79% of `app/core/AuthNZ` (87 tests, ~3min), so the scoped gate starts at 35 and ratchets — not the 70 originally guessed:*

```diff
           pytest -q --disable-warnings -p pytest_cov -p pytest_asyncio.plugin \
             -m "not jobs and not e2e" \
             tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
-            --cov=tldw_Server_API --cov-report=xml --cov-report=term-missing --cov-fail-under=5
+            --cov=tldw_Server_API --cov-report=xml --cov-report=term-missing --cov-fail-under=12
+
+      - name: AuthNZ coverage floor
+        if: needs.changes.outputs.coverage_required == 'true'
+        env:
+          PYTHONPATH: .
+          PYTEST_DISABLE_PLUGIN_AUTOLOAD: "1"
+          TEST_MODE: "true"
+          DISABLE_HEAVY_STARTUP: "1"
+        run: |
+          pytest -q --disable-warnings -p pytest_cov -p pytest_asyncio.plugin \
+            tldw_Server_API/tests/AuthNZ_SQLite \
+            --cov=tldw_Server_API/app/core/AuthNZ --cov-report=term --cov-fail-under=35
```

Ratchet policy: whenever measured total exceeds floor + 3, raise the floor. Longer term, add `--cov` + `coverage combine` across the `ci.yml` shards so the whole-suite number finally exists.

### R2 — Tests for the untested storage routes (F2, revised)

The OAuth surfaces already have lifecycle/policy tests (`tests/Slack/test_slack_oauth_lifecycle.py`, `tests/Discord/test_discord_oauth_lifecycle.py`) — no work needed there. The remaining work is the storage routes with zero or single-test coverage. New file `tldw_Server_API/tests/Storage/test_storage_user_routes.py`, using the existing `client_user_only` fixture from the root conftest (`tldw_Server_API/tests/conftest.py:883`, overrides `get_request_user`):

```python
import pytest

pytestmark = pytest.mark.integration

def test_create_folder_ok(client_user_only):
    resp = client_user_only.post("/api/v1/storage/folders", json={"name": "reports"})
    assert resp.status_code == 200

def test_create_folder_empty_name_is_422(client_user_only):
    resp = client_user_only.post("/api/v1/storage/folders", json={"name": ""})
    assert resp.status_code == 422

def test_patch_unknown_file_is_404(client_user_only):
    resp = client_user_only.patch("/api/v1/storage/files/999999",
                                  json={"folder_tag": "archive"})
    assert resp.status_code == 404

def test_delete_unknown_file_is_404(client_user_only):
    resp = client_user_only.delete("/api/v1/storage/files/999999")
    assert resp.status_code == 404

def test_restore_unknown_trash_item_is_404(client_user_only):
    resp = client_user_only.post("/api/v1/storage/trash/restore/999999")
    assert resp.status_code == 404

def test_permanent_delete_unknown_trash_item_is_404(client_user_only):
    resp = client_user_only.delete("/api/v1/storage/trash/999999")
    assert resp.status_code == 404
```

Routes and schemas verified against `storage_user_folders.py:25,46`, `storage_user_files.py:207,238`, `storage_trash.py:56,83`; request bodies from `FolderCreateRequest` (`name`, min_length=1) and `GeneratedFileUpdate` (all-optional metadata fields).

### R3 — Un-hide the excluded test dirs (F3)

`pyproject.toml` — replace silent exclusion with an explicit opt-out marker so the tests are collected, visible, and individually skippable with reasons:

```diff
-norecursedirs = [
-    "tldw_Server_API/tests/Character_Chat_NEW",
-    "tldw_Server_API/tests/TTS_NEW",
-    "tldw_Server_API/tests/Embeddings"
-]
```

and in each affected directory's `conftest.py`:

```python
import pytest
collect_ignore_glob: list[str] = []
pytestmark = pytest.mark.quarantined  # register marker; CI runs -m "not quarantined"
```

Then burn the quarantine list down file by file. (First step, zero risk: run `pytest <dir> --co -q` for each to learn whether they even collect.)

### R4 — Frontend coverage thresholds (F4)

`apps/packages/ui/vitest.config.ts` (same block for `apps/tldw-frontend/vitest.config.ts`):

```ts
export default defineConfig({
  test: {
    // ...existing config...
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "json-summary"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: ["src/**/__tests__/**", "src/**/*.d.ts"],
      thresholds: { lines: 50, functions: 50, branches: 40 }, // set to measured baseline − 2, then ratchet
    },
  },
})
```

CI: in `frontend-required.yml`, change the full-run branch from `bun run test:run` to `bun run test:coverage`. Measure once first and set thresholds just under the real number.

### R5 — Replace sleeps with clock control (F5)

Pattern for `tests/Evaluations/unit/test_circuit_breaker.py` — monkeypatch time instead of waiting out real breaker windows:

```python
def test_breaker_recovers_after_timeout(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: now["t"])
    breaker = CircuitBreaker(reset_timeout=1.0)
    breaker.record_failure(); breaker.record_failure()
    assert breaker.is_open
    now["t"] += 1.1          # was: time.sleep(1.1)
    assert not breaker.is_open
```

(Adjust to whichever time function the breaker actually calls — `time.time` vs `time.monotonic`.) For async waits, prefer polling with deadline over fixed sleeps:

```python
async def wait_for(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    raise TimeoutError
```

Start with the single `time.sleep(10)` in `tests/Evaluations/test_error_scenarios.py`.

### R6 — Property-based fuzzing for input surfaces (F10)

```python
from hypothesis import given, strategies as st

@pytest.mark.property
@given(st.text(min_size=0, max_size=50_000))
def test_chunker_never_raises_and_preserves_content(raw):
    chunks = chunk_text(raw, chunk_size=512, overlap=64)
    assert "".join(c.strip_markers() for c in chunks).replace(" ", "") \
        .startswith(raw[:0])  # replace with the module's real invariant
    for c in chunks:
        assert len(c) <= 512 + 64
```

Target `app/core/Ingestion_Media_Processing/` chunkers and upload filename/type validators first.

---

## Prioritized Improvement Plan

**Now (this week)**
1. R1: floor 5 → 12 (safe today; measured 13.39%) + AuthNZ scoped gate.
2. R2 (revised): tests for the untested storage routes (folder create, file PATCH/DELETE, trash mutations) — half a dozen tests total.
3. Run `pytest --co -q` on the three excluded dirs to learn their state (5 minutes).

**30 days**
4. R3: quarantine-marker migration; burn down at least one of the three dirs.
5. R4: frontend coverage measured + thresholds set at baseline.
6. R7 (F7): dedicated `tests/RateLimiting/` suite asserting 429 + `Retry-After`.
7. F9: triage the 40 unconditional skips.

**Quarter**
8. Shard-level `--cov` + `coverage combine` in `ci.yml` → first-ever whole-suite coverage number; ratchet floors from it.
9. R5: eliminate sleeps in Evaluations/Metrics suites (biggest wall-clock wins first).
10. F8: nightly `pytest-benchmark` job with compare-fail baseline.
11. R6: hypothesis fuzzing for ingestion and upload validation.

---

## What Is Already Good (keep doing this)

- **Test pyramid shape is correct** — ~4.5k unit / ~1k integration / ~292 E2E.
- **Isolation architecture** — per-test Postgres scratch DBs with Docker auto-start (`tests/_plugins/postgres.py`), env-baseline pinning in root conftest, leaked-connection tracking.
- **Deliberate plugin hygiene** — `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` + explicit plugin list + `--strict-markers`.
- **Dedicated security suite** — path traversal, SSRF/egress, RBAC under `tests/Security/`.
- **Frontend a11y as a CI gate** — axe-core on high-risk routes (`e2e/smoke/stage4-axe-high-risk-routes.spec.ts`), 40+ a11y test files.
- **Tiered E2E** with trace/screenshot/video-on-failure and CI retries.
- **Existing ratchet precedent** — the `cli/wizard` 70% gate (`ci.yml:257`) is exactly the model to replicate.
