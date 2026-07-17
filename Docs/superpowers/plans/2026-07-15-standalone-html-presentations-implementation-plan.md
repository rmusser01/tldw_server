# Standalone HTML Presentations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class, form-first `standalone_html` presentation mode that can generate one self-contained HTML+JavaScript document from every existing Slides source family, while keeping the document opaque, bounded, attachment-only, and non-executable across every tldw surface.

**Architecture:** Extend Slides with a discriminated persistence model and one shared standalone domain service. Admission snapshots bounded owner-local source into per-user receipt/input ledgers, Jobs carries only a receipt UUID, and a worker invokes one server-selected allowlisted target through an isolated bounded transport before atomically committing a validated document. REST, MCP, render workers, the WebUI, and the extension all enforce content-kind gates; the WebUI edits inert text and rebuilds only a trusted text outline.

**Tech Stack:** FastAPI, Pydantic, SQLite, Jobs WorkerSDK, httpx, html5lib, tinycss2, Loguru, Next.js, React, TypeScript, Monaco, Web Workers, Vitest, Playwright, pytest, Bandit.

---

## Scope And Locked Implementation Decisions

This plan implements the approved design in [2026-07-15-standalone-html-presentations-design.md](../specs/2026-07-15-standalone-html-presentations-design.md) under Backlog task `TASK-12115`. The following details are locked here so implementation does not pause or silently weaken the design:

1. **No execution in V1.** No iframe, `srcdoc`, DOM parser, HTML renderer, thumbnail, PDF/video renderer, popup, worker constructed from model source, navigation, or inline `text/html` response is permitted. The only browser Blob URL is the short-lived, attachment-validated download handoff.
2. **Closed initial adapter catalog.** V1 accepts only source-controlled adapters with dedicated request/response codecs:
   - `openai_official_chat_v1` → `openai` → `https://api.openai.com:443/v1/chat/completions`;
   - `anthropic_official_messages_v1` → `anthropic` → `https://api.anthropic.com:443/v1/messages`;
   - `llamacpp_loopback_chat_v1_ipv4` / `llamacpp_loopback_chat_v1_ipv6` → `llama.cpp` → `http://127.0.0.1:8080/v1/chat/completions` / `http://[::1]:8080/v1/chat/completions`;
   - `ollama_loopback_chat_v1_ipv4` / `ollama_loopback_chat_v1_ipv6` → `ollama` → `http://127.0.0.1:11434/v1/chat/completions` / `http://[::1]:11434/v1/chat/completions`.
   The administrator still configures exactly one default tuple and an exact tuple allowlist. Custom OpenAI adapters, endpoint overrides, aliases that change identity, routers, proxies, redirects, remote HTTP, DNS-resolved loopback, and every other registry adapter remain ineligible in V1.
3. **Secrets and shared source-free metadata.** `SLIDES_STANDALONE_HMAC_KEYS_JSON` supplies a JSON object of 32-byte base64url secrets keyed by 1–32 character ASCII key IDs; `SLIDES_STANDALONE_HMAC_CURRENT_KEY_ID` selects the current key. Secrets never enter a database. Source-free key IDs/state/timestamps, migration diagnostics, and reconciliation coordination live in dedicated tables in the shared Jobs store so every API and external-worker node observes one authority in SQLite or PostgreSQL.
4. **Fenced reconciliation.** The shared Jobs-store coordination row holds the renewable lease, holder UUID, monotonically increasing fencing token, continuation cursor, deployment/config epoch, startup-complete epoch, last-complete epoch, and measured lag. Acquisition/takeover atomically increments the token; every renew/checkpoint/release compares holder plus token. A stale holder cannot publish progress after takeover, and every process keeps generation admission closed until it observes the shared startup-complete epoch for its deployment/config epoch.
5. **MCP WebSocket fail-closed behavior.** Slides tools are registered on HTTP MCP normally. On WebSocket, `MCP_unified/protocol.py` applies a per-request scope-marker filter to both `tools/list` and `tools/call`; it never mutates the global tool registry. The marker comes only from a guarded protocol subclass of Uvicorn 0.35.0 `WebSocketsSansIOProtocol`, run with `ws_per_message_deflate=False`, which streams decoded data frames through the Slides shallow-path prefilter in at most 64 KiB pieces. Pin `uvicorn[standard]==0.35.0` and `websockets==15.0.1` because this is an internal protocol integration. Supported tldw launchers use the guarded class; an unguarded external Uvicorn launch keeps non-Slides MCP tools but omits Slides from discovery and execution and never advertises a standalone-aware guard.
6. **Extension metadata handoff.** Add authenticated, source-free `GET /api/v1/slides/presentations/{presentation_id}/metadata`. The extension can identify kind and show “Open in WebUI” without requesting `html_document`; HTML new/detail editor routes are not registered in the extension build.
7. **Dependencies.** Add direct runtime bounds `html5lib==1.1` and `tinycss2==1.4.0` to `pyproject.toml`. The repository has no canonical Python lock artifact, so the direct exact pins are the V1 reproducibility gate. Reuse the existing Monaco dependency; add no frontend package.
8. **Prompt loading.** Add `slides.standalone_html_system` to `Config_Files/Prompts/slides.prompts.md`. Add a strict prompt-loader entry point that distinguishes no override from a configured-but-unreadable override; standalone generation fails closed instead of falling back.
9. **Frontend dependency gate.** Before the first frontend test, run `bun install --frozen-lockfile` from `apps/`. The planning worktree could not execute a real frontend baseline because its workspace dependencies were absent; no product test ran or failed. Stop and report if the clean install or the pre-change Presentation Studio regression set fails.

### Backlog Execution Prerequisite

Before Task 1, read `backlog://workflow/overview`, view `TASK-12115` through the official Backlog MCP/CLI, verify it remains `In Progress`, and set its `implementationPlan` to this document. After every task, use the official task-edit workflow to append the commit/test evidence and refresh `modifiedFiles`; never edit the task Markdown by hand. Stage the generated Backlog task file with that task's code/test commit, as shown in every commit block below.

## Five-Stage Delivery Map

### Stage 1: Discriminated Persistence And Authoritative Validation

**Goal:** Make `structured_slides` and `standalone_html` explicit, mutually exclusive, safely migrated, and queryable without loading HTML into metadata paths.

**Success Criteria:** Schema v2 migrates atomically; cross-field invariants are enforced on every create/update/restore; HTML is validated and derived metadata is transactional; list/search/version metadata use source-free projections; legacy clients retain structured-only semantics.

**Tests:** Migration rollback/concurrency, domain invariants, validator budgets/watchdogs, summaries/search/version retention, content-kind negotiation, and unchanged structured Slides regressions.

**Status:** Complete

### Stage 2: Bounded Generation, Receipts, Jobs, And Recovery

**Goal:** Admit one immutable bounded source snapshot, call exactly one configured target per normal attempt, and reconcile Slides/Jobs safely without placing source in Jobs.

**Success Criteria:** Closed configuration and keyring fail closed; all five source resolvers are bounded before tokenization/provider work; provider transport enforces endpoint and raw-body rules; receipts replay idempotently; WorkerSDK terminal outcomes, commit fencing, expiry, and dormant-database reconciliation are covered.

**Tests:** Config/allowlist/prompt/key rotation, max+1 source cases, retrieval-only RAG, raw provider streaming and error redaction, receipt races, active/archive UUID recovery, worker retry/cancel/expiry, and fenced reconciler takeover.

**Status:** In Progress

### Stage 3: REST, MCP, Transport, And Secondary-Surface Guards

**Goal:** Expose the approved API contracts while preventing source materialization, rendering, export confusion, and compatibility bypasses on every transport.

**Success Criteria:** Capabilities/generation/status/raw save/draft/export endpoints match the spec; ASGI and MCP prefilters run before ordinary parsing; CORS/security headers work through normal and drain paths; render workers and MCP reject unsupported HTML operations before loading source.

**Tests:** Chunked/malformed request admission, strict JSON, union response redaction, status/replay matrix, strong ETags, attachment headers, CORS, MCP HTTP/guarded-WS behavior, and API/worker render rejection.

**Status:** Not Started

### Stage 4: Form-First WebUI And Inert Source Workspace

**Goal:** Add a resumable form flow and a dedicated HTML workspace that edits inert text, saves explicitly, recovers safely, and displays only a trusted outline.

**Success Criteria:** Client contracts remain discriminated; capability failure is explicit; immutable snapshots and idempotency resume correctly; HTML never enters the structured Zustand/autosave path; Monaco/fallback, outline worker, conflicts, recovery, downloads, principal changes, and extension handoff satisfy the no-execution contract.

**Tests:** Vitest contract/component/hook/worker suites, structured regressions, typecheck/OpenAPI guards, extension route tests, keyboard/mobile/a11y checks, and multi-engine no-execution E2E.

**Status:** Not Started

### Stage 5: Documentation, Rollout, And Release Gates

**Goal:** Make the feature operable, auditable, and safe to deploy without changing its default-off posture.

**Success Criteria:** Config, API, migration backup, worker/reconciler, key rotation, MCP launcher, safe-outline, download, and rollback docs are complete; OpenAPI is regenerated; focused/full verification and Bandit pass; Backlog evidence is current.

**Tests:** Backend/frontend/extension/E2E matrices, OpenAPI verification, Bandit on touched Python, `git diff --check`, and a final independent code/security review.

**Status:** Not Started

## Planned File Structure

### Backend files to create

- `tldw_Server_API/app/core/Slides/slides_migrations.py` — authoritative schema-v2 runner.
- `tldw_Server_API/app/core/Slides/standalone_html_contracts.py` — limits, enums, closed manifests, provenance, and stable domain errors.
- `tldw_Server_API/app/core/Slides/standalone_html_validator.py` — strict scalar/HTML/CSS validation and derived title/text/slide metadata.
- `tldw_Server_API/app/core/Slides/standalone_html_validation_pool.py` — bounded priority queues and killable subprocess watchdog.
- `tldw_Server_API/app/core/Slides/standalone_html_config.py` — immutable capability/config snapshot and closed adapter catalog.
- `tldw_Server_API/app/core/Slides/standalone_html_registry.py` — keyring/domain logic over shared Jobs-store source-free metadata APIs.
- `tldw_Server_API/app/core/Slides/standalone_html_sources.py` — bounded prompt/chat/media/notes/RAG resolvers.
- `tldw_Server_API/app/core/Slides/standalone_html_provider.py` — isolated bounded HTTP codecs and response extraction.
- `tldw_Server_API/app/core/Slides/standalone_html_service.py` — receipt claim/replay, HTML save/restore/export, and worker commit transactions.
- `tldw_Server_API/app/core/Slides/standalone_html_reconciler.py` — active/dormant reconciliation and retention sweeps.
- `tldw_Server_API/app/core/Slides/presentation_service.py` — shared REST/MCP kind guards and metadata/detail projections.
- `tldw_Server_API/app/core/Security/standalone_html_request_guard.py` — ASGI receive limits, lexical JSON preflight, and shallow forbidden-field scanner.
- `tldw_Server_API/app/api/v1/endpoints/slides_standalone_html.py` — capabilities, generation/status, raw save/draft, and metadata routes.
- `tldw_Server_API/app/services/standalone_html_generation_jobs_worker.py` — `presentation.generate` WorkerSDK handler.
- `tldw_Server_API/app/core/MCP_unified/transport/guarded_slides_websocket.py` — compression-disabled streaming protocol marker/prefilter.
- `tldw_Server_API/Config_Files/Prompts/slides.prompts.md` — packaged standalone system prompt.

### Frontend files to create

- `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioIndex.tsx`
- `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioNew.tsx`
- `apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlGenerationForm.tsx`
- `apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlWorkspace.tsx`
- `apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlSourceEditor.tsx`
- `apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlSafeOutline.tsx`
- `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-source.ts`
- `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-recovery.ts`
- `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-download.ts`
- `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-outline.worker.ts`
- `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-outline-client.ts`
- `apps/packages/ui/src/hooks/useSlidesCapabilities.ts`
- `apps/packages/ui/src/hooks/usePresentationPrincipalScope.ts`
- `apps/packages/ui/src/hooks/useStandaloneHtmlGeneration.ts`

## Task 1: Add Atomic Slides Schema V2 And Discriminated Persistence

**Files:**
- Create: `tldw_Server_API/app/core/Slides/slides_migrations.py`
- Modify: `tldw_Server_API/app/core/Slides/slides_db.py`
- Modify: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_db_migration.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_domain.py`
- Test: `tldw_Server_API/tests/Slides/test_slides_db.py`

- [x] **Step 1: Write failing schema-v2 and invariant tests**

Cover new databases, v0/v1 databases, an empty/multirow version table, idempotent reopen, injected statement rollback, concurrent first access, legacy backfill, explicit projections, and these complete-row invariants:

```python
assert structured.content_kind == "structured_slides"
assert structured.slides is not None and structured.html_document is None
assert html.content_kind == "standalone_html"
assert json.loads(html.slides) == [] and html.html_document is not None
```

Also prove `generation_job_uuid` is unique when nonnull and that list/search/version-metadata queries do not select `html_document` or `payload_json`.

- [x] **Step 2: Run tests and confirm the expected red state**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_db_migration.py \
  tldw_Server_API/tests/Slides/test_standalone_html_domain.py \
  tldw_Server_API/tests/Slides/test_slides_db.py
```

Expected: new tests fail because schema v2, discriminators, and projections do not exist; existing structured tests remain green.

- [x] **Step 3: Implement the transactional migration**

Move feature DDL out of `_ensure_schema` into a `BEGIN IMMEDIATE` runner that re-reads actual schema/version after the lock, executes statements individually, normalizes `schema_version` to one row containing `2`, and rolls back on any failure. Add the seven presentation fields, receipt/input ledgers, indexes, and legacy `structured_slides` backfill exactly as specified.

- [x] **Step 4: Add explicit row and projection APIs**

Replace feature-path `SELECT *` usage with typed detail, summary, kind, version-metadata, receipt, and input projections. Add a no-create Slides DB path resolver for dormant reconciliation. Enforce complete-candidate invariants in database transactions rather than relying on Pydantic alone.

- [x] **Step 5: Run the focused tests and structured regression**

Run the Step 2 command, then:

```bash
python -m pytest -q tldw_Server_API/tests/Slides/test_slides_api.py
```

Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Slides/slides_migrations.py tldw_Server_API/app/core/Slides/slides_db.py tldw_Server_API/app/core/DB_Management/db_path_utils.py tldw_Server_API/tests/Slides/test_standalone_html_db_migration.py tldw_Server_API/tests/Slides/test_standalone_html_domain.py tldw_Server_API/tests/Slides/test_slides_db.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): add discriminated schema v2 (TASK-12115)"
```

## Task 2: Build The Authoritative Validator And Supervised Pool

**Files:**
- Create: `tldw_Server_API/app/core/Slides/standalone_html_contracts.py`
- Create: `tldw_Server_API/app/core/Slides/standalone_html_validator.py`
- Create: `tldw_Server_API/app/core/Slides/standalone_html_validation_pool.py`
- Modify: `pyproject.toml`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_validator.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_validation_pool.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_dependency_smoke.py`

- [x] **Step 1: Write failing pure-validator tests**

Cover UTF-8/scalar validity, allowed HTML whitespace controls, forbidden other C0/C1 controls, NUL, exact 1 MiB bytes, one complete document, required document/slide structure, 1–30 slides, 50,000 HTML tokens, 65,536-byte tokens, 10,000 elements, 20,000 attributes, depth 128, strict first parse error, at most 64 styles/524,288 CSS bytes/100,000 CSS tokens/10,000 declarations/65,536-byte CSS tokens/depth 64/100 CSS errors, and 250,000 indexable characters. Test the allowed single classic final-child script; forbidden URLs/events/styles/forms/frames/workers/storage/network/resource CSS; generation-time notes cardinality versus later edit rules; title NFC/control/bidi/scalar/byte bounds; malformed parser/CSS inputs; iterative semantic-text extraction; derived title/count/digest; and fixed redacted error codes. Include max-1/max/max+1 and adversarial nesting/token cases. Add a clean-install smoke test that imports `html5lib` and `tinycss2` and exercises one accepted HTML and CSS path through the authoritative validator.

- [x] **Step 2: Write failing pool tests**

Cover at most four subprocesses, interactive queue 24, reserved generation queue 8, fairness, saturation `503`, a 60-second watchdog using a shortened test clock, terminate/reap/replace, caller cancellation, and no source in logs/exceptions.

- [x] **Step 3: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_validator.py \
  tldw_Server_API/tests/Slides/test_standalone_html_validation_pool.py \
  tldw_Server_API/tests/Slides/test_standalone_html_dependency_smoke.py
```

Expected: fail because the validator modules and direct dependencies do not exist.

- [x] **Step 4: Implement the smallest authoritative pipeline**

Pin `html5lib==1.1` and `tinycss2==1.4.0`; parse without executing; walk bounded parser output; rebuild only derived scalar metadata; and return a frozen result object. Keep subprocess messages closed and source-bearing only on the private pipe. Do not introduce a sanitizer or renderer. Install the modified project into the activated project environment before running green tests:

```bash
python -m pip install -e .
```

If dependency download is blocked by sandbox/network policy, request the normal package-install approval and retry; do not substitute an unpinned package.

- [x] **Step 5: Re-run tests and dependency metadata check**

```bash
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_validator.py \
  tldw_Server_API/tests/Slides/test_standalone_html_validation_pool.py \
  tldw_Server_API/tests/Slides/test_standalone_html_dependency_smoke.py
python -m pip check
```

Expected: pass.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml tldw_Server_API/app/core/Slides/standalone_html_contracts.py tldw_Server_API/app/core/Slides/standalone_html_validator.py tldw_Server_API/app/core/Slides/standalone_html_validation_pool.py tldw_Server_API/tests/Slides/test_standalone_html_validator.py tldw_Server_API/tests/Slides/test_standalone_html_validation_pool.py tldw_Server_API/tests/Slides/test_standalone_html_dependency_smoke.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): validate standalone HTML safely (TASK-12115)"
```

## Task 3: Add Shared Kind Guards, Summaries, Versions, And Search

**Files:**
- Create: `tldw_Server_API/app/core/Slides/presentation_service.py`
- Modify: `tldw_Server_API/app/core/Slides/slides_db.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/slides_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Modify: `tldw_Server_API/app/core/Slides/slides_export.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_domain.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_api.py`
- Test: `tldw_Server_API/tests/Slides/test_slides_export.py`

- [x] **Step 1: Write failing representation and persistence tests**

Require discriminated detail/summary models, source-free list/search/version metadata, HTML FTS from derived semantic text only, compact UTF-8 snapshots, 25-version retention, no-op source saves, same-kind restore, immutable kind/provenance/job UUID, and metadata-only HTML tombstones. Test omitted/structured-only/dual `X-Slides-Accept-Content-Kinds` behavior before pagination and source loading.

- [x] **Step 2: Run the focused tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_domain.py \
  tldw_Server_API/tests/Slides/test_standalone_html_api.py \
  tldw_Server_API/tests/Slides/test_slides_export.py
```

- [x] **Step 3: Implement shared domain projections and guards**

Make `presentation_service.py` the only kind-aware mapping/mutation seam used by REST and later MCP code. Parse the negotiation header once; apply structured-only filtering inside list/search SQL; check a target's kind using the lightweight projection before detail/version/export/render work; and return fixed `400`, `406`, or `409` errors without loading HTML.

- [x] **Step 4: Implement snapshots, summaries, and explicit export mapping**

Serialize snapshots with `ensure_ascii=False` under the fixed ceiling, prune within the successful transaction, and restore only mutable same-kind fields. Add `html` to export format but defer attachment transport to Task 11. Preserve existing structured response fields and weak ETags.

- [x] **Step 5: Run tests and structured regressions**

Run Step 2, then:

```bash
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_slides_db.py \
  tldw_Server_API/tests/Slides/test_slides_api.py \
  tldw_Server_API/tests/Slides/test_slides_export.py
```

Expected: pass.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Slides/presentation_service.py tldw_Server_API/app/core/Slides/slides_db.py tldw_Server_API/app/api/v1/schemas/slides_schemas.py tldw_Server_API/app/api/v1/endpoints/slides.py tldw_Server_API/app/core/Slides/slides_export.py tldw_Server_API/tests/Slides/test_standalone_html_domain.py tldw_Server_API/tests/Slides/test_standalone_html_api.py tldw_Server_API/tests/Slides/test_slides_export.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): add content-kind domain guards (TASK-12115)"
```

## Task 4: Add Closed Configuration, Prompt, Keyring, And Registry

**Files:**
- Create: `tldw_Server_API/app/core/Slides/standalone_html_config.py`
- Create: `tldw_Server_API/app/core/Slides/standalone_html_registry.py`
- Create: `tldw_Server_API/Config_Files/Prompts/slides.prompts.md`
- Modify: `tldw_Server_API/app/core/Utils/prompt_loader.py`
- Modify: `tldw_Server_API/app/core/config_sections/__init__.py`
- Create: `tldw_Server_API/app/core/config_sections/slides.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Modify: `tldw_Server_API/Config_Files/Prompts/README.md`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_config.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_registry.py`
- Test: `tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py`

- [x] **Step 1: Write failing config, endpoint, prompt, and key tests**

Cover default-off behavior, independent egress kill, exact catalog entries and normalized identities, exact default-tuple membership, model case sensitivity, rejected custom/override/router/proxy/fallback targets, verified HTTPS, all six catalog adapters including the four explicit loopback adapters, remote HTTP/`localhost`/LAN/link-local/userinfo/query/fragment rejection, prompt digest/version/size, unreadable override fail-closed behavior, effective limit clamping, and deterministic `generation_config_revision`.

For the HMAC keyring, cover strict JSON/duplicate IDs/base64url/32-byte secret validation, maximum four current/retiring keys, constant-time comparisons, domain separation, current-key selection, 32-day retirement floor, complete-sweep proof, missing-secret global admission/worker failure, and absence of secrets in persisted registry rows.

- [x] **Step 2: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_config.py \
  tldw_Server_API/tests/Slides/test_standalone_html_registry.py \
  tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py
```

- [x] **Step 3: Implement typed configuration and strict prompt loading**

Add a frozen `SlidesStandaloneHtmlConfig` loaded from `[SlidesStandaloneHtml]` plus environment overrides. The section contains nonsecret enable/kill flags, default provider/model/adapter, exact `allowed_targets_json`, timeouts, token budget, and only downward-adjustable limits. Derive endpoint identity from the closed adapter catalog; do not accept a base URL. Add `load_prompt_strict(module, key, max_bytes)` without changing fallback behavior for unrelated callers.

- [x] **Step 4: Implement keyring logic against a shared-store interface**

Define a narrow injected registry interface for source-free key ID/state/timestamps and sweep-proof queries; test Task 4 with an in-memory fake. Do not create a local coordination database. Task 7 implements the SQLite/PostgreSQL Jobs-store tables and manager methods behind this interface.

- [x] **Step 5: Run tests and inspect stored values**

Run Step 2. Assert serialized fake-repository calls contain no configured secret bytes or base64 strings; Task 7 repeats this assertion against raw SQLite/PostgreSQL rows.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Slides/standalone_html_config.py tldw_Server_API/app/core/Slides/standalone_html_registry.py tldw_Server_API/Config_Files/Prompts/slides.prompts.md tldw_Server_API/app/core/Utils/prompt_loader.py tldw_Server_API/app/core/config_sections/__init__.py tldw_Server_API/app/core/config_sections/slides.py tldw_Server_API/Config_Files/config.txt tldw_Server_API/Config_Files/Prompts/README.md tldw_Server_API/tests/Slides/test_standalone_html_config.py tldw_Server_API/tests/Slides/test_standalone_html_registry.py tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): configure isolated HTML generation (TASK-12115)"
```

## Task 5: Add Bounded Source Resolvers And Retrieval-Only RAG

**Files:**
- Create: `tldw_Server_API/app/core/Slides/standalone_html_sources.py`
- Create: `tldw_Server_API/tests/DB_Management/unit/test_postgresql_error_redaction.py`
- Create: `tldw_Server_API/tests/RAG_NEW/unit/test_preinstalled_local_reranker.py`
- Create: `tldw_Server_API/tests/RAG_NEW/unit/test_slides_source_retrieval_hardening.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/base.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/note_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/repositories/media_lookup_repository.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/execution_ops.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/postgres_helpers.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_sources.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_reranker_trust_remote_code.py`

- [x] **Step 1: Write failing source-contract tests**

For prompt, chat, media, notes, and RAG, cover owner isolation, missing/soft-deleted sources, deterministic ordering/separators, identifier/count/query bounds, character max+1 before tokenization, token max+1 before enqueue, exact provenance summaries, no image/blob selection, and no unbounded `SELECT *`. Assert rejected inputs make zero tokenizer, provider, or Jobs calls.

- [x] **Step 2: Write failing RAG-profile tests**

Require an immutable `slides_source_retrieval_v1` profile with answer generation, HyDE, completion/VLM rewriting, decomposition, clarification, generative reranking, verification, adaptive reruns, web/discussion/URL/image/video fallback, and request-time model downloads disabled. Accept only preinstalled local `flashrank`, `cross_encoder`, or `none`; format bounded `rag_result.documents` and never `generated_answer`.

- [x] **Step 3: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_sources.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py
```

- [x] **Step 4: Add bounded repository projections and resolvers**

Add max+1, owner-scoped title/content/transcript/message projections at the repository layer. Resolve dependencies explicitly in `standalone_html_sources.py`, canonicalize one immutable source snapshot/provenance object, and run the advertised tokenizer only after the character ceiling. Do not call the existing structured generator's chunk-summary path.

- [x] **Step 5: Run focused tests plus affected repository suites**

Run Step 3, then the narrow Chat/Media/RAG suites containing the modified repository tests. Record exact paths and counts in Backlog.

Final Task 5 verification: the 14-file affected matrix passed 589 tests with 11 existing warnings. Ruff passed the scoped production and test files; Black left all five entirely new files unchanged after formatting; `git diff --check` passed. Production-only Bandit scanned the complete touched scope and reported 0 findings. A fresh independent specification/security review returned APPROVED with no remaining P0-P2 findings.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/slides.py tldw_Server_API/app/core/Slides/standalone_html_sources.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends tldw_Server_API/app/core/DB_Management/chacha/message_store.py tldw_Server_API/app/core/DB_Management/chacha/note_store.py tldw_Server_API/app/core/DB_Management/media_db tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py tldw_Server_API/app/core/RAG/rag_service/profiles.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/tests/Slides/test_standalone_html_sources.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py tldw_Server_API/tests/DB_Management/unit/test_postgresql_error_redaction.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py tldw_Server_API/tests/RAG_NEW/unit/test_reranker_trust_remote_code.py tldw_Server_API/tests/RAG_NEW/unit/test_preinstalled_local_reranker.py tldw_Server_API/tests/RAG_NEW/unit/test_slides_source_retrieval_hardening.py Docs/superpowers/plans/2026-07-15-standalone-html-presentations-implementation-plan.md "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): snapshot bounded generation sources (TASK-12115)"
```

## Task 6: Implement The Isolated Provider Transport And Generation Call

**Files:**
- Create: `tldw_Server_API/app/core/Slides/standalone_html_provider.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_provider.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_generation.py`

- [x] **Step 1: Write failing endpoint/payload tests**

For every catalog adapter, require exact endpoint identity, provider-specific payload shape, system/user separation, `stream: false`, no tools/cookies/application credentials/extra headers/extra body/router metadata, fixed token ceiling, and model equality. Prove all endpoint/base URL/proxy overrides are ignored or rejected before network I/O.

- [x] **Step 2: Write failing raw-response tests**

Use an in-process mock transport to cover `trust_env=False`, `follow_redirects=False`, `Accept-Encoding: identity`, conflicting/gzip/br/deflate rejection before body reads, declared/missing/chunked/dishonest 8 MiB ceilings, success JSON lexical budgets, duplicate keys/nonfinite/lone surrogates, exactly one complete outer Markdown fence stripped, nested/additional fences or surrounding prose rejected, fixed non-2xx errors with discarded bounded bodies, cancellation, timeout, connection failure, and 1 MiB extracted-document enforcement. Assert provider body/source never appears in logs or exception text.

- [x] **Step 3: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_provider.py \
  tldw_Server_API/tests/Slides/test_standalone_html_generation.py
```

- [x] **Step 4: Implement one isolated async call path**

Construct `httpx.AsyncClient` locally with no shared eager error hook, environment trust, redirects, or decompression ambiguity. Stream raw bytes through the bounded provider JSON preflight, strictly decode, and extract only the catalog codec's text field. Strip exactly one complete outer Markdown fence when present; reject additional fences or any surrounding prose. Return document bytes to the authoritative validator. Recheck the current enable/egress flag and exact stored tuple immediately before opening the request.

- [x] **Step 5: Re-run tests and prove one-call semantics**

Run Step 3. Add an assertion that one normal worker attempt performs exactly one completion call and no fallback.

Final Task 6 verification: the initial assertion-level run was RED with 60 failures; follow-up race tests separately proved the client-entry and lazy-stream scheduling gaps before their fixes. The final focused provider/generation suite passed 123 tests, and the full standalone HTML regression family passed 761 tests. Black, Ruff, `py_compile`, and `git diff --check` passed. Production-only Bandit scanned 518 LOC with 0 findings or errors. A fresh independent full-range specification and quality review returned READY with no Critical, Important, or Minor findings. Task 6 is recorded in commits `6e1220fe3b` and `a37a484718`.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Slides/standalone_html_provider.py tldw_Server_API/tests/Slides/test_standalone_html_provider.py tldw_Server_API/tests/Slides/test_standalone_html_generation.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): isolate HTML provider transport (TASK-12115)"
```

## Task 7: Extend Jobs UUID, Terminal, And Shared Coordination Primitives

**Files:**
- Modify: `tldw_Server_API/app/core/Slides/standalone_html_registry.py`
- Modify: `tldw_Server_API/app/core/Jobs/worker_sdk.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/app/core/Jobs/migrations.py`
- Modify: `tldw_Server_API/app/core/Jobs/pg_migrations.py`
- Test: `tldw_Server_API/tests/Jobs/test_worker_sdk.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_sqlite.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_postgres.py`

- [ ] **Step 1: Write failing Jobs migration and UUID-authority tests**

Cover SQLite/PostgreSQL archive scope indexes, unique nonnull immutable UUID audit, duplicate/ambiguous legacy diagnostics that disable only standalone generation, owner/domain/queue/type-scoped active/archive UUID lookup, UUID-only archive compression/mutation, numeric-ID reuse rejection, and required nonnull UUID for new `presentation.generate` jobs.

- [ ] **Step 2: Write failing shared coordination and terminal-outcome tests**

Require shared Jobs-store tables for source-free key ID/state/timestamps and one reconciliation row containing lease holder/expiry, monotonically increasing fencing token, continuation cursor, deployment/config epoch, startup-complete epoch, last-complete epoch, and lag. Cover two-manager takeover, stale-token renew/checkpoint/release rejection, SQLite/PostgreSQL parity, secret absence from raw rows, and `WorkerTerminalOutcome(failed|cancelled)` skipping normal `complete_job` through one expected-state/lease/UUID/owner/domain/type CAS.

- [ ] **Step 3: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_postgres.py
```

- [ ] **Step 4: Implement migrations and scoped manager APIs**

Audit before creating indexes; never auto-deduplicate. Add the two source-free shared tables and transactional manager methods. SQLite uses `BEGIN IMMEDIATE`; PostgreSQL uses row locking/atomic `UPDATE ... RETURNING`. Every coordination mutation compares holder plus fencing token, and every startup epoch is tied to the immutable deployment/config revision.

- [ ] **Step 5: Implement WorkerSDK terminal outcomes and the registry adapter**

Add the closed outcome branch before `complete_job`, preserving typed retry behavior. Make `standalone_html_registry.py` call the shared JobManager metadata APIs; it stores key IDs/state/timestamps only and never creates a local database.

- [ ] **Step 6: Run SQLite and PostgreSQL tests**

Run Step 3. If the repository PostgreSQL fixture reports unavailable, record that skip; do not invent a database setup.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Slides/standalone_html_registry.py tldw_Server_API/app/core/Jobs/worker_sdk.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/Jobs/migrations.py tldw_Server_API/app/core/Jobs/pg_migrations.py tldw_Server_API/tests/Jobs/test_worker_sdk.py tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_postgres.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(jobs): add Slides UUID and fencing primitives (TASK-12115)"
```

## Task 8: Add Receipt Claims And The Generation Worker Commit Path

**Files:**
- Create: `tldw_Server_API/app/core/Slides/standalone_html_service.py`
- Create: `tldw_Server_API/app/services/standalone_html_generation_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Slides/slides_db.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_generation_jobs.py`

- [ ] **Step 1: Write failing claim/replay and worker tests**

Cover strict idempotency-key syntax, canonical manifests with `ensure_ascii=False`, domain-separated HMACs, constant-time equality, atomic owner/client claim, same-key exact queued/running/completed/failed/cancelled replay, same-key/different-request conflict, cross-owner indistinguishable lookup, stale config only after replay lookup, exact prompt/source/target snapshots, `input_expires_at` fixed at receipt creation plus 24 hours, deterministic 30-day terminal expiry, and source deletion only on terminal CAS.

Also cover receipt-only payload bytes under Jobs normalization/truncation—including `JOBS_JSON_TRUNCATE=true` and a deliberately tiny JSON limit—API-first/worker-first immutable UUID binding, retryable return-to-queued behavior, final lease/state/cancel recheck, exactly one committed presentation, commit-before-Jobs-complete recovery, cancel-before/after-check races, late-result discard, exhausted/nonretryable cleanup, and completed-presentation precedence.

- [ ] **Step 2: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Slides/test_standalone_html_generation_jobs.py
```

- [ ] **Step 3: Implement the atomic receipt/input service**

Keep the Jobs payload exactly `{\"receipt_id\": \"uuid\"}` with domain `slides`, queue `default`, and type `presentation.generate`. Claim receipt plus immutable input in the per-user Slides transaction before enqueue. Bind numeric Jobs ID only together with the immutable Jobs UUID; numeric ID is never correlation authority.

- [ ] **Step 4: Implement the worker handler and fenced commit**

Load the owner-scoped receipt/input, validate correlation HMACs, and reread no mutable source. Acquire Task 2's low-priority generation validation reservation before the provider call so saturation produces zero provider calls; consume that reservation when the returned document enters validation. Recheck key/kill/target, invoke Task 6, validate, perform the final Jobs check, and atomically commit presentation/receipt/input. Return normal bounded metadata only for completed Jobs; use typed retry or `WorkerTerminalOutcome` for every other path.

- [ ] **Step 5: Run tests and structured worker regressions**

Run Step 2 plus `tldw_Server_API/tests/Slides/test_presentation_render_jobs.py`.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Slides/standalone_html_service.py tldw_Server_API/app/services/standalone_html_generation_jobs_worker.py tldw_Server_API/app/core/Slides/slides_db.py tldw_Server_API/tests/Slides/test_standalone_html_generation_jobs.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): commit receipt-backed HTML jobs (TASK-12115)"
```

## Task 9: Add Fenced Dormant-Database Reconciliation And Lifecycle

**Files:**
- Create: `tldw_Server_API/app/core/Slides/standalone_html_reconciler.py`
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_reconciler.py`
- Test: `tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py`
- Test: `tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py`

- [ ] **Step 1: Write failing reconciliation and lifecycle tests**

Cover active-owner priority, canonical one-level user-DB discovery, containment/regular-file/schema/no-symlink checks, one open database at a time, shared startup epoch before handler admission, Jobs-store/leader failure closed, 15-minute full sweep, cursor resume, lag overload, two-process crash takeover, stale leader publication refusal, Jobs-unavailable 24-hour logical expiry, confirmed 15-minute missing-job failure, active/archive UUID repair, completed-presentation precedence, terminal input cleanup, and key-retirement proof after a complete fenced sweep.

- [ ] **Step 2: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_reconciler.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py \
  tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py
```

- [ ] **Step 3: Implement the fenced sweep**

Acquire and checkpoint only through Task 7's shared Jobs-store row. Prioritize active Jobs owners, then stream dormant databases through the fenced cursor. Apply receipt transitions as idempotent owner-scoped CAS operations and publish startup-complete/lag only with the current token. Reconciliation never re-enqueues a missing job.

- [ ] **Step 4: Register startup, steady-state, and shutdown ownership**

Run the startup sweep before generation handler admission, continue at least once per minute, and guarantee a complete pass every 15 minutes or expose `generation_reconciler_overloaded`. Register stop events/tasks beside existing content Jobs workers and test deterministic shutdown.

- [ ] **Step 5: Run focused tests and worker lifecycle regressions**

Run Step 2 and the existing content-worker catalog tests that cover startup failure/cleanup.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Slides/standalone_html_reconciler.py tldw_Server_API/app/services/startup_content_jobs_pollers.py tldw_Server_API/tests/Slides/test_standalone_html_reconciler.py tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): reconcile standalone generation safely (TASK-12115)"
```

## Task 10: Enforce Receive-Time Admission, Redaction, And Shared CORS

**Files:**
- Create: `tldw_Server_API/app/core/Security/standalone_html_request_guard.py`
- Modify: `tldw_Server_API/app/main.py`
- Modify: `tldw_Server_API/app/core/Security/drain_gate_middleware.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/security/request_guards.py`
- Test: `tldw_Server_API/tests/Security/test_standalone_html_request_admission.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py`
- Test: `tldw_Server_API/tests/Config/test_route_and_cors_guards.py`
- Test: `tldw_Server_API/tests/Services/test_drain_gate_middleware.py`

- [ ] **Step 1: Write failing fixed-route ASGI tests**

Drive raw ASGI `http.request` events for generation (4 MiB), HTML save (1 MiB), and draft attachment (1 MiB). Cover absent/single identity encoding, compressed/multiple/conflicting encoding, duplicate/comma/negative/nondecimal content length, declared max+1, missing/chunked/dishonest lengths, early stop/drain, split UTF-8/escapes, depth/token/container/member/string budgets, duplicate keys, nonfinite values, lone surrogates, malformed JSON, exact `json_structure_too_complex`, and bounded redacted errors/logs.

- [ ] **Step 2: Write failing generic forbidden-field scanner tests**

Feed structured REST and HTTP MCP payloads in 64 KiB and adversarial one-byte chunks. Require escaped-key decoding, key/string distinction, shallow `content_kind`/`html_document` path tracking, rejection before the forbidden value is consumed or spooled, constant scanner state, exact byte replay for allowed structured payloads, cleanup on cancellation/error, and unchanged legal body/frame semantics.

- [ ] **Step 3: Write failing CORS/error-boundary tests**

Exercise preflight and actual normal/drain/maintenance/validation/unexpected-error responses. Preserve configured origins, credentials, and `Vary: Origin`; allow `Idempotency-Key`, `If-Match`, and `X-Slides-Accept-Content-Kinds`; expose `Content-Disposition`, `ETag`, `Last-Modified`, `Retry-After`, `Content-Length`, and existing request/trace headers.

- [ ] **Step 4: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Security/test_standalone_html_request_admission.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py \
  tldw_Server_API/tests/Config/test_route_and_cors_guards.py \
  tldw_Server_API/tests/Services/test_drain_gate_middleware.py
```

- [ ] **Step 5: Implement one route-aware receive wrapper and lexical scanner**

Install the guard before FastAPI body parsing. Fixed routes use bounded buffering only after incremental preflight succeeds; generic routes stream through the shallow rejection scanner and replay exact allowed bytes. Add route-scoped request/response validation handlers that emit only allowlisted codes and sanitized locations; source-bearing response validation/serialization returns fixed `standalone_html_response_invalid`. Disable source-body capture in logging/tracing integrations.

- [ ] **Step 6: Centralize CORS response policy and re-run tests**

Use one immutable exposed-header tuple from normal and drain middleware. Run Step 4 and assert source strings do not occur in captured logs, exceptions, metrics labels, or response bodies.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Security/standalone_html_request_guard.py tldw_Server_API/app/main.py tldw_Server_API/app/core/Security/drain_gate_middleware.py tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py tldw_Server_API/app/core/MCP_unified/security/request_guards.py tldw_Server_API/tests/Security/test_standalone_html_request_admission.py tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py tldw_Server_API/tests/Config/test_route_and_cors_guards.py tldw_Server_API/tests/Services/test_drain_gate_middleware.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): bound standalone request admission (TASK-12115)"
```

## Task 11: Expose Capabilities, Generation, Save, Status, And Attachments

**Files:**
- Create: `tldw_Server_API/app/api/v1/endpoints/slides_standalone_html.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/slides.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/slides_schemas.py`
- Modify: `tldw_Server_API/app/core/Slides/presentation_service.py`
- Modify: `tldw_Server_API/app/core/Slides/slides_export.py`
- Test: `tldw_Server_API/tests/Slides/test_standalone_html_api.py`
- Test: `tldw_Server_API/tests/Slides/test_slides_endpoint_sanitization.py`
- Test: `tldw_Server_API/tests/Slides/test_slides_export.py`

- [ ] **Step 1: Write failing exact capability and generation API tests**

Require the exact capabilities shape, independent persistence/generation/validator availability, `private, no-store`, no live provider health check, every approved safe disabled reason including `generation_reconciler_overloaded`, and a nonnull revision only when enabled. For `POST /slides/generations`, cover all five closed source unions, unknown fields/provider overrides, idempotency header syntax, stale config, request snapshot claim before source resolution, Jobs outage, and exact new/replay status codes/bodies. For status, cover malformed/unknown/other-owner UUID equivalence, synchronous reconciliation, bounded progress/errors, and no HTML.

- [ ] **Step 2: Write failing detail/save/version/attachment tests**

Cover the discriminated detail union, source-free metadata route, strong HTML ETags with weak-tag transition parsing, stale `412` without remote HTML, raw octet-stream save/no-op/lost-response reconciliation, title/digest/count derivation, generic PATCH/reorder/render rejection, draft echo without persistence or deck validation, saved HTML and JSON attachments, fixed filenames, exact MIME/disposition/security headers, version ETag/Last-Modified, and source-bearing JSON `nosniff`/no-store/auth `Vary` headers. Assert no response ever uses `text/html`.

- [ ] **Step 3: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_standalone_html_api.py \
  tldw_Server_API/tests/Slides/test_slides_endpoint_sanitization.py \
  tldw_Server_API/tests/Slides/test_slides_export.py
```

- [ ] **Step 4: Implement thin routes over the shared services**

The new endpoint module performs auth/rate-limit/header mapping only. It receives prevalidated envelopes/raw bytes from Task 10, calls Tasks 3/8 services, and maps closed domain outcomes. Include it from the existing Slides router so route-group enablement remains unchanged. Keep the old synchronous structured generation routes untouched.

- [ ] **Step 5: Apply kind guards to every existing target operation**

Before parsing `row.slides` or a version payload, apply the source-free kind guard to get/update/patch/reorder/delete/restore/export/render routes. Preserve structured weak ETags and response shapes. Add `Vary: X-Slides-Accept-Content-Kinds` wherever representation differs.

- [ ] **Step 6: Run focused and structured suites**

Run Step 3, then:

```bash
python -m pytest -q \
  tldw_Server_API/tests/Slides/test_slides_api.py \
  tldw_Server_API/tests/Slides/test_slides_ordering.py \
  tldw_Server_API/tests/Slides/test_slides_export.py
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/slides_standalone_html.py tldw_Server_API/app/api/v1/endpoints/slides.py tldw_Server_API/app/api/v1/schemas/slides_schemas.py tldw_Server_API/app/core/Slides/presentation_service.py tldw_Server_API/app/core/Slides/slides_export.py tldw_Server_API/tests/Slides/test_standalone_html_api.py tldw_Server_API/tests/Slides/test_slides_endpoint_sanitization.py tldw_Server_API/tests/Slides/test_slides_export.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): expose standalone HTML API (TASK-12115)"
```

## Task 12: Guard MCP, Renderers, Workers, And WebSocket Protocol

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/transport/guarded_slides_websocket.py`
- Create: `tldw_Server_API/scripts/run_server_guarded_mcp.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py`
- Modify: `tldw_Server_API/app/core/Slides/presentation_rendering.py`
- Modify: `tldw_Server_API/app/services/presentation_render_jobs_worker.py`
- Modify: `Dockerfiles/Dockerfile.prod`
- Modify: `pyproject.toml`
- Test: `tldw_Server_API/tests/MCP_unified/test_slides_module_standalone_html.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_guarded_slides_websocket.py`
- Test: `tldw_Server_API/tests/Slides/test_presentation_rendering.py`
- Test: `tldw_Server_API/tests/Slides/test_presentation_render_jobs.py`

- [ ] **Step 1: Write failing MCP operation-matrix tests**

Require source-free list/search/get/delete/undelete/version-list metadata for opted-in MCP, no HTML source in list/version list/status/errors, and explicit rejection before payload load for HTML create/update/patch/reorder/generate/version-content/restore/export/render. Route shared invariants through `presentation_service.py`; prove existing structured MCP generation/exports still pass and standalone RAG never calls the module's current `_get_rag_content` fallback.

- [ ] **Step 2: Write failing real WebSocket protocol tests**

Start the guarded Uvicorn 0.35.0 `WebSocketsSansIOProtocol` subclass with `websockets==15.0.1`, inspect the handshake to prove per-message compression is absent, send one large fragmented frame, and assert the shallow scanner receives at most 64 KiB pieces and rejects before a forbidden value. Exercise both `tools/list` and a direct `tools/call` through `protocol.py`: the guarded marker permits existing structured Slides tools, while an unguarded standard protocol omits/rejects Slides but retains unrelated tools. Verify exact allowed-message replay, disconnect cleanup, and no source in logs.

- [ ] **Step 3: Write failing defense-in-depth render tests**

Require both API snapshot loading and `presentation_render_jobs_worker` to reject `standalone_html` using the kind projection before parsing slides or starting Playwright/ffmpeg. Cover a forged/stale Jobs payload so route-only checks cannot be bypassed.

- [ ] **Step 4: Run tests and confirm failure**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/MCP_unified/test_slides_module_standalone_html.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_guarded_slides_websocket.py \
  tldw_Server_API/tests/Slides/test_presentation_rendering.py \
  tldw_Server_API/tests/Slides/test_presentation_render_jobs.py
```

- [ ] **Step 5: Implement transport-aware Slides registration and shared domain calls**

Pin `uvicorn[standard]==0.35.0` and `websockets==15.0.1`. The custom `WebSocketsSansIOProtocol` subclass scans decoded frame fragments and adds the ASGI scope marker only while run with `ws_per_message_deflate=False`. `server.py` passes transport scope into `protocol.py`; `protocol.py` applies a per-request marker filter to both discovery and execution without mutating the global module/tool registry. The guarded launcher calls `uvicorn.run(app, ws=GuardedSlidesWebSocketProtocol, ws_per_message_deflate=False, ...)`, accepts the documented host/port/workers/log/proxy options, and becomes the production Docker entrypoint. Replace duplicated Slides MCP CRUD/restore/export mapping with calls to the shared source-free projections and kind guards; do not add source-bearing standalone tools in V1.

```bash
python -m pip install -e .
python -c "import uvicorn, websockets; assert uvicorn.__version__ == '0.35.0'; assert websockets.__version__ == '15.0.1'"
```

- [ ] **Step 6: Add renderer/worker guards and run regressions**

Run Step 4, then:

```bash
python -m pytest -q \
  tldw_Server_API/tests/MCP_unified/test_slides_module_exports.py \
  tldw_Server_API/tests/Slides/test_presentation_render_jobs.py \
  tldw_Server_API/tests/Slides/test_presentation_rendering.py
```

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/transport/guarded_slides_websocket.py tldw_Server_API/scripts/run_server_guarded_mcp.py tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/core/MCP_unified/modules/implementations/slides_module.py tldw_Server_API/app/core/Slides/presentation_rendering.py tldw_Server_API/app/services/presentation_render_jobs_worker.py Dockerfiles/Dockerfile.prod pyproject.toml tldw_Server_API/tests/MCP_unified/test_slides_module_standalone_html.py tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py tldw_Server_API/app/core/MCP_unified/tests/test_guarded_slides_websocket.py tldw_Server_API/tests/Slides/test_presentation_rendering.py tldw_Server_API/tests/Slides/test_presentation_render_jobs.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(slides): guard MCP and render surfaces (TASK-12115)"
```

## Task 13: Add Discriminated Client Contracts And Slides Transport Support

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/presentations.ts`
- Modify: `apps/packages/ui/src/services/tldw/request-core.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/store/presentation-studio.tsx`
- Modify: `apps/packages/ui/src/hooks/usePresentationStudioAutosave.tsx`
- Create: `apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-standalone.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts`
- Modify: `apps/packages/ui/src/hooks/__tests__/usePresentationStudioAutosave.test.tsx`

- [ ] **Step 1: Install the exact workspace and run the pre-change frontend baseline**

```bash
cd apps
bun install --frozen-lockfile
cd tldw-frontend
bun run test:run -- \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__ \
  ../packages/ui/src/hooks/__tests__/usePresentationStudioAutosave.test.tsx \
  ../packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx \
  ../packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: dependency installation and the existing Presentation Studio tests pass. If either fails before product edits, stop, preserve the output, and update `TASK-12115`; do not classify it as a feature regression.

- [ ] **Step 2: Write failing client contract tests**

Require structured/HTML/unknown discriminated records, source-free summaries, exact capabilities and receipt variants, preservation of unknown kinds, no coercion to `slides: []`, response-provided ETags, and `X-Slides-Accept-Content-Kinds: structured_slides,standalone_html` on every applicable Slides request including one token-refresh retry.

Test `listPresentations`, `getSlidesCapabilities`, `submitPresentationGeneration`, `getPresentationGeneration`, source-free metadata, raw string `saveStandaloneHtmlSource`, draft attachment, and saved HTML download. The raw string path must reject lone surrogates/NUL/over-limit input before `fetch`; attachment methods must verify status, exact MIME, and fixed disposition before returning bytes.

- [ ] **Step 3: Run new tests and confirm failure**

```bash
cd apps/tldw-frontend
bun run test:run -- \
  ../packages/ui/src/services/__tests__/tldw-api-client.presentations-standalone.test.ts \
  ../packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts \
  ../packages/ui/src/hooks/__tests__/usePresentationStudioAutosave.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 4: Implement the discriminated client and narrow structured state**

Make the domain normalizer authoritative and remove its unused duplicate from `TldwApiClient.ts`. Return `{record, etag}` for detail/save. Keep standalone source out of the structured Zustand store and autosave hook by narrowing both to `StructuredPresentationStudioRecord`. Send raw source as a validated JavaScript string so refresh retry preserves UTF-8 semantics without the current binary-body serialization path.

- [ ] **Step 5: Run focused tests, typecheck, and OpenAPI guard**

Run Step 3, then:

```bash
cd apps/packages/ui
bunx tsc --noEmit -p tsconfig.json
bun run verify:openapi
```

Expected: pass after Task 11 OpenAPI generation; if the generated schema has not landed yet, run typecheck now and record the OpenAPI gate for Task 17 rather than weakening the path guard.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/domains/presentations.ts apps/packages/ui/src/services/tldw/request-core.ts apps/packages/ui/src/services/tldw/openapi-guard.ts apps/packages/ui/src/store/presentation-studio.tsx apps/packages/ui/src/hooks/usePresentationStudioAutosave.tsx apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-standalone.test.ts apps/packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts apps/packages/ui/src/hooks/__tests__/usePresentationStudioAutosave.test.tsx "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(webui): add standalone presentation contracts (TASK-12115)"
```

## Task 14: Build The Index, Direct-Material Form, And Resumable Job UX

**Files:**
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioIndex.tsx`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioNew.tsx`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlGenerationForm.tsx`
- Create: `apps/packages/ui/src/hooks/useSlidesCapabilities.ts`
- Create: `apps/packages/ui/src/hooks/useStandaloneHtmlGeneration.ts`
- Modify: `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioPage.tsx`
- Modify: `apps/packages/ui/src/routes/option-presentation-studio.tsx`
- Modify: `apps/packages/ui/src/routes/option-presentation-studio-new.tsx`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioIndex.test.tsx`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlGenerationForm.test.tsx`
- Create: `apps/packages/ui/src/hooks/__tests__/useSlidesCapabilities.test.tsx`
- Create: `apps/packages/ui/src/hooks/__tests__/useStandaloneHtmlGeneration.test.tsx`

- [ ] **Step 1: Write failing index/capability tests**

Cover paginated load-more/deduplication, kind badges and HTML metadata, loading/empty/error/retry/offline states, unknown kinds, prominent New action, exact no-store capabilities fetch, generation-disabled reason, validator-unavailable distinctions, explicit Retry, and no fallback from coarse server capabilities/OpenAPI inference.

- [ ] **Step 2: Write failing form/job tests**

The first UI release exposes direct pasted material only while sending the shared backend `source: {kind: "prompt", prompt}` union. Cover all closed option values, labels/help, byte/scalar/local field limits, no native form restoration, no provider picker, visible configured provider/model/endpoint, immutable canonical snapshot before POST, `crypto.getRandomValues` idempotency keys, revision echo, duplicate-submit lock, ambiguous submission, exact replay, bounded polling/`Retry-After`, Stop waiting, Resume, Forget, confirmed Start different, terminal Try again with a new key, completed handoff, auth loss, missing binding, and 24-hour principal/origin-scoped session recovery.

- [ ] **Step 3: Run tests and confirm failure**

```bash
cd apps/tldw-frontend
bun run test:run -- \
  ../packages/ui/src/hooks/__tests__/useSlidesCapabilities.test.tsx \
  ../packages/ui/src/hooks/__tests__/useStandaloneHtmlGeneration.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioIndex.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlGenerationForm.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 4: Implement the smallest form-first flow**

Split index/new/detail dispatch out of the current monolith. Store only capped resume metadata plus a separate capped form draft in `sessionStorage`; keep the immutable submitted snapshot in component memory and scoped recovery. Subscribe directly to auth/config, pagehide/pageshow, focus, and visibility events; clear source-bearing state synchronously on scope mismatch. Show real states and text only—never invented percentage progress.

- [ ] **Step 5: Run tests and Presentation Studio regressions**

Run Step 3, then:

```bash
bun run test:run -- \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__ \
  ../packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioIndex.tsx apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioNew.tsx apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlGenerationForm.tsx apps/packages/ui/src/hooks/useSlidesCapabilities.ts apps/packages/ui/src/hooks/useStandaloneHtmlGeneration.ts apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioPage.tsx apps/packages/ui/src/routes/option-presentation-studio.tsx apps/packages/ui/src/routes/option-presentation-studio-new.tsx apps/packages/ui/src/components/Option/PresentationStudio/__tests__/PresentationStudioIndex.test.tsx apps/packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlGenerationForm.test.tsx apps/packages/ui/src/hooks/__tests__/useSlidesCapabilities.test.tsx apps/packages/ui/src/hooks/__tests__/useStandaloneHtmlGeneration.test.tsx "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(webui): add resumable HTML generation form (TASK-12115)"
```

## Task 15: Build The Inert Editor, Safe Outline, Save, Recovery, And Download

**Files:**
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlWorkspace.tsx`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlSourceEditor.tsx`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlSafeOutline.tsx`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-source.ts`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-recovery.ts`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-download.ts`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-outline.worker.ts`
- Create: `apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-outline-client.ts`
- Create: `apps/packages/ui/src/hooks/usePresentationPrincipalScope.ts`
- Modify: `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioPage.tsx`
- Test: `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlSourceEditor.test.tsx`
- Test: `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-outline.test.ts`
- Test: `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlWorkspace.test.tsx`
- Test: `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-recovery.test.ts`
- Test: `apps/packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-download.test.ts`

- [ ] **Step 1: Write failing scalar/editor boundary tests**

Require rejection of lone surrogates, NUL, and UTF-8 max+1 before `TextEncoder`, state, worker, or recovery; exact accepted bytes/digest; visible source label; inert plain-text Monaco model with `links: false`; no HTML language service/link/hover providers; rejecting scoped opener; no successful form name; spellcheck/autocorrect/autocapitalize/autocomplete/password-manager opt-outs; textarea fallback parity; and disposal without global Monaco side effects.

- [ ] **Step 2: Write failing outline worker tests**

Cover static application-owned worker URL, lexical preflight budgets, no DOM/HTML/CSS preservation, trusted text-only headings/paragraphs/lists/tables/figures/notes, C0/C1 and bidi-control removal, `dir="auto"` plus bidi isolation, no links/assets/URLs, one active plus one coalesced pending request, digest correlation, latest-wins, stale/failed/current labels, 10-second terminate/restart watchdog with shortened clock, at most 50,000 nodes, depth 128, 30 cards, 4,096 scalars per block, 20,000 per slide, and 100,000 total, cancellation/disposal, and no `DOMParser`, `innerHTML`, `srcdoc`, source-derived URL/import/module/function, or backend repaint call.

- [ ] **Step 3: Write failing workspace/recovery/save/download tests**

Cover component-local source only, desktop grid/mobile Code/Outline tabs, dirty/save aria-live states, no structured autosave/store/render controls, 24-hour principal/origin/project-scoped recovery, never-autoapply choices, quota warning, pagehide flush/dispose, persisted pageshow reauthentication, subject switch/logout cleanup, navigation warning, explicit `If-Match` save, lost-response digest reconciliation, three conflict choices and second-race behavior, exact attachment validation, application-owned anchor only, fixed filename, delayed ≤1-second revoke, unmount/pagehide revoke, and no navigation/resource sink.

- [ ] **Step 4: Run tests and confirm failure**

```bash
cd apps/tldw-frontend
bun run test:run -- \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlSourceEditor.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-outline.test.ts \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlWorkspace.test.tsx \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-recovery.test.ts \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-download.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 5: Implement the inert source pipeline**

Branch on the source-free metadata/kind before requesting HTML detail or initializing structured styles/store. Fetch HTML detail directly with `no-store`, immediately place source in component-local state, and clear the response reference. Reuse lazy Monaco loading only; create no syntax/link providers. After its lexical preflight, the static outline worker lazily loads the existing `cheerio/slim` dependency, walks its non-browser AST with an explicit iterative stack, and never calls recursive `.text()`/`textContent` helpers. It returns a closed scalar DTO rendered with normal React text nodes.

- [ ] **Step 6: Implement explicit save, recovery, and validated attachments**

Never call `usePresentationStudioAutosave`. Save complete raw source against the last server ETag; preserve local source on every error; require confirmations for discard/overwrite; and adopt only server ETags/titles. Recovery helpers validate before returning source and never let persisted scope select an account. Download helpers create the Blob URL only after exact headers pass and never expose the URL outside the temporary anchor closure.

- [ ] **Step 7: Run tests, typecheck, and static no-execution search**

Run Step 4 and:

```bash
cd apps/packages/ui
bunx tsc --noEmit -p tsconfig.json
rg -n "dangerouslySetInnerHTML|DOMParser|srcdoc|innerHTML|insertAdjacentHTML|window\.open|window\.location" src/components/Option/PresentationStudio
```

Expected: typecheck passes; any search hit is in an explicit negative test or reviewed application-owned code, never a source sink.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlWorkspace.tsx apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlSourceEditor.tsx apps/packages/ui/src/components/Option/PresentationStudio/StandaloneHtmlSafeOutline.tsx apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-source.ts apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-recovery.ts apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-download.ts apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-outline.worker.ts apps/packages/ui/src/components/Option/PresentationStudio/standalone-html-outline-client.ts apps/packages/ui/src/hooks/usePresentationPrincipalScope.ts apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioPage.tsx apps/packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlSourceEditor.test.tsx apps/packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-outline.test.ts apps/packages/ui/src/components/Option/PresentationStudio/__tests__/StandaloneHtmlWorkspace.test.tsx apps/packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-recovery.test.ts apps/packages/ui/src/components/Option/PresentationStudio/__tests__/standalone-html-download.test.ts "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(webui): add inert HTML presentation workspace (TASK-12115)"
```

## Task 16: Keep The Extension Source-Free And Add WebUI Handoff

**Files:**
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-metadata.ts`
- Modify: `apps/packages/ui/src/components/Option/PresentationStudio/ExtensionStartPanel.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx`
- Modify: `apps/extension/tests/e2e/presentation-studio-start.spec.ts`

- [ ] **Step 1: Write failing extension route/source tests**

Require the extension build to retain Presentation Studio metadata index and structured quick-start, omit HTML new/detail editor registrations, use only the source-free metadata endpoint for direct project links, and display a fixed-origin “Open in WebUI” action for HTML. Assert no HTML detail/version/download request and no source-bearing extension storage/message occurs.

- [ ] **Step 2: Run tests and confirm failure**

```bash
cd apps/tldw-frontend
bun run test:run -- \
  ../packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx \
  --maxWorkers=1 --no-file-parallelism
cd ../extension
bun run compile
```

- [ ] **Step 3: Implement transport-aware route registration and handoff**

Next pages continue importing WebUI wrappers directly. Extension route metadata omits source-bearing routes and reuses the existing canonical WebUI-origin helper; never construct a target from presentation source or model output.

- [ ] **Step 4: Run route tests, extension compile, and extension E2E**

```bash
cd apps/tldw-frontend
bun run test:run -- ../packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx --maxWorkers=1 --no-file-parallelism
cd ../extension
bun run compile
bunx playwright test tests/e2e/presentation-studio-start.spec.ts --reporter=line
```

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/route-registry.tsx apps/packages/ui/src/routes/route-metadata.ts apps/packages/ui/src/components/Option/PresentationStudio/ExtensionStartPanel.tsx apps/packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx apps/extension/tests/e2e/presentation-studio-start.spec.ts "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "feat(extension): hand off HTML presentations safely (TASK-12115)"
```

## Task 17: Document, Exercise, Audit, And Prepare Rollout

**Files:**
- Create: `Docs/User_Guides/WebUI_Extension/Presentation_Studio.md`
- Create: `Docs/Deployment/Standalone_HTML_Presentations.md`
- Modify: `Docs/API/Slides.md`
- Modify: `Docs/Design/Presentations.md`
- Modify: `Docs/Product/Slides_Infographics_Workproducts_PRD.md`
- Modify: `Docs/MCP/Unified/Modules.md`
- Modify: `tldw_Server_API/app/core/Slides/README.md`
- Modify: `CHANGELOG.md`
- Modify: `Docs/RELEASE_NOTES.md`
- Modify: `apps/tldw-frontend/playwright.config.ts`
- Create: `apps/tldw-frontend/e2e/workflows/presentation-studio-standalone-html.spec.ts`
- Create: `apps/tldw-frontend/e2e/workflows/presentation-studio-standalone-html.security.spec.ts`
- Create: `tldw_Server_API/tests/Slides/test_standalone_html_integration.py`
- Modify: `backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md` through the Backlog CLI/MCP only.

- [ ] **Step 1: Write failing integration and browser acceptance tests**

Backend integration covers direct material plus mocked chat/media/notes/RAG, exactly one allowed provider call per normal attempt, list/search/reopen/version/save/export, default-off generation with saved HTML still readable, legacy filtering, and no `text/html` response.

Browser coverage includes generate, Stop/Resume, edit, trusted outline, explicit save, lost response, conflict choices, reopen, attachment download, keyboard/mobile flow, 44px targets, visible focus, no horizontal overflow, same-principal pagehide/Back restoration, expired/other-principal clearing, bfcache account switch, malformed/hung outline work, URL/Cmd-click/context-menu inertness, CSP unchanged, no sentinel execution, no outbound resource request, and strict Blob URL sinks. Run the security matrix in Chromium, Firefox, and WebKit.

- [ ] **Step 2: Run focused integration tests and confirm any remaining red state**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Slides/test_standalone_html_integration.py
cd apps/tldw-frontend
bunx playwright test e2e/workflows/presentation-studio-standalone-html.spec.ts --project=chromium --reporter=line
bunx playwright test e2e/workflows/presentation-studio-standalone-html.security.spec.ts --project=chromium --project=standalone-html-firefox --project=standalone-html-webkit --reporter=line
```

Expected before docs/config completion: implementation behavior passes; missing Playwright projects or docs assertions fail until Step 3.

- [ ] **Step 3: Update docs, PRD, Playwright projects, and deployment contract**

Document:

- exact API/capability/error/ETag/attachment examples without executable sample payloads;
- default-off enablement, closed adapter IDs, exact tuple allowlist, key source/rotation, egress kill, worker/reconciler health, and fixed limits;
- schema-v2 backup-first forward migration and old-binary incompatibility;
- guarded MCP WebSocket launcher and the default omission of Slides tools on unguarded WebSocket transports;
- safe outline limitations, explicit save/recovery, executable-file warning, and the absolute no-execution/no-preview promise;
- rollback: disable generation/egress first, drain workers, retain readable saved HTML, and never downgrade the database with an old binary;
- the narrow PRD exception: standalone JavaScript may be generated/stored/edited/versioned/downloaded only as opaque text; arbitrary execution remains prohibited everywhere.

- [ ] **Step 4: Run the complete backend verification matrix**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Slides \
  tldw_Server_API/tests/Jobs/test_worker_sdk.py \
  tldw_Server_API/tests/Jobs/test_jobs_finalize_idempotency_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_slides_generation_coordination_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py \
  tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py \
  tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py \
  tldw_Server_API/tests/Services/test_drain_gate_middleware.py \
  tldw_Server_API/tests/MCP_unified/test_slides_module_exports.py \
  tldw_Server_API/tests/MCP_unified/test_slides_module_standalone_html.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_slides_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_guarded_slides_websocket.py \
  tldw_Server_API/tests/Security/test_standalone_html_request_admission.py \
  tldw_Server_API/tests/Config/test_route_and_cors_guards.py
```

Record fixture-reported PostgreSQL skips separately. No manually constructed database substitute is permitted.

- [ ] **Step 5: Run the complete frontend and extension verification matrix**

```bash
cd apps/tldw-frontend
bun run test:run -- \
  ../packages/ui/src/components/Option/PresentationStudio/__tests__ \
  ../packages/ui/src/hooks/__tests__/usePresentationStudioAutosave.test.tsx \
  ../packages/ui/src/hooks/__tests__/useSlidesCapabilities.test.tsx \
  ../packages/ui/src/hooks/__tests__/useStandaloneHtmlGeneration.test.tsx \
  ../packages/ui/src/routes/__tests__/option-presentation-studio-route-guards.test.tsx \
  ../packages/ui/src/services/__tests__/tldw-api-client.presentations-normalization.test.ts \
  ../packages/ui/src/services/__tests__/tldw-api-client.presentations-standalone.test.ts \
  --maxWorkers=1 --no-file-parallelism
cd ../packages/ui
bunx tsc --noEmit -p tsconfig.json
bun run verify:openapi
bunx eslint src/components/Option/PresentationStudio src/hooks/useSlidesCapabilities.ts src/hooks/usePresentationPrincipalScope.ts src/hooks/useStandaloneHtmlGeneration.ts src/services/tldw/domains/presentations.ts src/services/tldw/TldwApiClient.ts src/services/tldw/request-core.ts
cd ../../tldw-frontend
bunx playwright install chromium firefox webkit
bunx playwright test e2e/workflows/presentation-studio-standalone-html.spec.ts --project=chromium --reporter=line
bunx playwright test e2e/workflows/presentation-studio-standalone-html.security.spec.ts --project=chromium --project=standalone-html-firefox --project=standalone-html-webkit --reporter=line
cd ../extension
bun run compile
bunx playwright test tests/e2e/presentation-studio-start.spec.ts --reporter=line
```

- [ ] **Step 6: Run Bandit and repository hygiene gates**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
git diff --name-only --diff-filter=ACMRT 3f0ce9175d90183dc1e1102d07e32ca13a737eb9 -- '*.py' | rg '^tldw_Server_API/(app|scripts)/' | sort -u
git diff --name-only --diff-filter=ACMRT 3f0ce9175d90183dc1e1102d07e32ca13a737eb9 -- '*.py' | rg '^tldw_Server_API/(app|scripts)/' | sort -u | xargs python -m bandit -f json -o /tmp/bandit_standalone_html_presentations.json
test -s /tmp/bandit_standalone_html_presentations.json
git diff --check
git status --short
```

The diff base is the approved design commit. Stage any newly created Task 17 Python source file before this command so it is present in the diff. Review the printed path list and Bandit JSON; fix every new finding in touched source. Do not accept an omitted source path or empty/missing report as success.

- [ ] **Step 7: Request fresh correctness and security reviews**

Dispatch reviewers with the approved spec, this plan, the complete diff, and verification evidence. Resolve every actionable P0–P2 and any P3 that affects no-execution, data isolation, compatibility, accessibility, or operability. Re-run the narrow affected suites after every fix and the complete gates after the final fix.

- [ ] **Step 8: Finalize Backlog and commit documentation/evidence**

Using the Backlog CLI/MCP, mark acceptance items only when supported by evidence; record exact test counts, PostgreSQL/browser skips, Bandit report path/summary, touched files, commit IDs, and known limitations. Do not mark the task complete or a PR merge-ready until the human requester supplies the required human-written `Change summary` explaining what changed and why the implementation choices were made.

```bash
git add Docs/API/Slides.md Docs/Design/Presentations.md Docs/Product/Slides_Infographics_Workproducts_PRD.md Docs/MCP/Unified/Modules.md Docs/User_Guides/WebUI_Extension/Presentation_Studio.md Docs/Deployment/Standalone_HTML_Presentations.md tldw_Server_API/app/core/Slides/README.md CHANGELOG.md Docs/RELEASE_NOTES.md apps/tldw-frontend/playwright.config.ts apps/tldw-frontend/e2e/workflows/presentation-studio-standalone-html.spec.ts apps/tldw-frontend/e2e/workflows/presentation-studio-standalone-html.security.spec.ts tldw_Server_API/tests/Slides/test_standalone_html_integration.py "backlog/tasks/task-12115 - Add-first-class-standalone-HTML-JS-presentation-generation.md"
git commit -m "docs(slides): document standalone HTML rollout (TASK-12115)"
```

## Execution Notes

- Complete tasks in order. A later task may add a failing test earlier only when it clarifies the next contract; do not merge red commits.
- Keep `TASK-12115` current after each task with commit ID, touched files, verification counts, and deviations from this plan.
- Use `apply_patch` for tracked edits. Preserve unrelated worktree changes and stage only the paths listed for the current task.
- If any source-bearing route, error handler, metric, trace, cache, extension message, or renderer cannot meet the no-execution/redaction boundary, fail the standalone capability closed and stop for review.
- The first product UI remains direct-material only. Chat/media/notes/RAG support is backend-tested but gains no selector UI in this release.
- Use the approved source checkout virtual environment path shown in commands for this worktree. In another checkout, activate that checkout's project `.venv` before Python/pytest/Bandit commands.
- The baseline already established during planning is `100 passed` for `test_slides_db.py`, `test_slides_generator.py`, and `test_slides_api.py`; frontend baseline execution awaits the Task 13 clean workspace install gate.
