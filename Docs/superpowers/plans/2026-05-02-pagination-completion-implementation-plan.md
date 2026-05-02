# Pagination Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use superpowers:subagent-driven-development only when the user explicitly authorizes parallel agents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete API v1 pagination normalization by classifying all list/search endpoints and migrating remaining page, cursor, and custom-envelope families without breaking existing payloads.

**Architecture:** Build an inventory matrix first, then ship separate PRs by pagination model: helper cleanup, page/per-page, cursor, custom legacy envelopes, and contract guardrails. Every migrated endpoint preserves legacy fields and adds canonical nested `pagination` metadata unless it is explicitly exempted as provider-compatible, raw-list, or versioning-blocked.

**Tech Stack:** FastAPI, Pydantic v2, pytest, Bandit, OpenAPI schema inspection, optional frontend Vitest where client parsers change.

---

## Execution Constraints

- Start each implementation tranche from the latest `origin/dev`. If the planning or implementation branch is behind, rebase or recreate the tranche branch before editing source files.
- Activate the project virtual environment before every Python, pytest, or Bandit command. Commands below assume an activated shell; in nested `.worktrees/*` checkouts, use the repository-root venv if the worktree does not have its own `.venv`.
- Stage exact touched files only. Do not use wildcard or whole-directory `git add` commands unless the tranche intentionally owns every changed file in that directory.
- Keep behavior-changing route migrations out of the inventory/helper PRs. The first source-changing PR should still be helper-only.

## File Structure

Planning and inventory artifacts:

- Create: `tools/pagination_inventory.py`
- Create: `Docs/Design/Pagination_Completion_Matrix.md`
- Create: `Docs/Design/Pagination_Contract_Exemptions.md`
- Modify: `Docs/superpowers/specs/2026-05-02-pagination-completion-design.md` only if the design changes.

Shared backend helpers:

- Modify: `tldw_Server_API/app/api/v1/schemas/pagination.py`
- Modify: `tldw_Server_API/app/api/v1/utils/pagination.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py`
- Test: `tldw_Server_API/tests/Utils/test_pagination_contract.py`
- Test: `tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py`

Page/per-page candidate files:

- Modify: `tldw_Server_API/app/api/v1/schemas/prompt_studio_base.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_projects.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_test_cases.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/paper_search_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/research_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/paper_search.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/research.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/media_response_models.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/listing.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/versions.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/navigation.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/privileges.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/collections_feeds.py`

Cursor candidate files:

- Modify: `tldw_Server_API/app/api/v1/schemas/audio_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_history.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/workflows.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workflows.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notifications.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sync.py`

Custom-envelope candidate files:

- Modify in separate PRs only after inventory classification:
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/kanban/*.py`
- `tldw_Server_API/app/api/v1/schemas/kanban_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/mcp_hub_management.py`
- `tldw_Server_API/app/api/v1/schemas/mcp_hub_schemas.py`

Frontend files, only when a frontend client consumes a migrated route:

- Modify: `apps/packages/ui/src/services/response-envelope.ts`
- Modify route-specific service files under `apps/packages/ui/src/services/`
- Test: relevant Vitest files under `apps/packages/ui/src/**/__tests__/` or adjacent test directories.

## Task 1: Build the Inventory Matrix

**Files:**
- Create: `tools/pagination_inventory.py`
- Create: `Docs/Design/Pagination_Completion_Matrix.md`
- Create: `Docs/Design/Pagination_Contract_Exemptions.md`

- [x] **Step 1: Write the inventory script**

Create `tools/pagination_inventory.py` to scan:

- `tldw_Server_API/app/api/v1/endpoints/**/*.py`
- `tldw_Server_API/app/api/v1/schemas/**/*.py`

Prefer AST/static analysis plus optional OpenAPI/app-route introspection. If importing the FastAPI app is too side-effect-prone, the script should still emit the static inventory and mark route method/path as `unknown` rather than guessing.

The script should emit a Markdown table with:

- method/path when detectable
- endpoint file/function
- response model
- query pagination params
- response pagination fields
- family: `offset`, `page`, `cursor`, `custom`, `provider`, `raw-list`, `not-paginated`, or `unknown`
- recommended tranche
- test file candidates

- [x] **Step 2: Run the inventory script**

Run:

```bash
source .venv/bin/activate
python tools/pagination_inventory.py > Docs/Design/Pagination_Completion_Matrix.md
```

Expected: the matrix contains known families such as `prompt_studio`, `paper_search`, `audio_history`, `workflows`, `watchlists`, `kanban`, and raw list endpoints like character list/search routes.

- [x] **Step 3: Create the first exemption document**

Create `Docs/Design/Pagination_Contract_Exemptions.md` with these initial categories:

- Provider-compatible routes: do not alter response shape unless explicitly approved.
- Raw `list[...]` routes: defer until API versioning or add a sibling versioned route.
- Streaming/file-export routes: not paginated or not applicable.
- Internal admin/event routes where count is unavailable: allow `total=None`.

- [x] **Step 4: Verify the inventory does not change behavior**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: only docs and the inventory script changed.

- [x] **Step 5: Commit the inventory PR**

```bash
git add tools/pagination_inventory.py Docs/Design/Pagination_Completion_Matrix.md Docs/Design/Pagination_Contract_Exemptions.md
git commit -m "Phase pagination-completion: add pagination inventory matrix"
```

## Task 2: Consolidate Shared Pagination Helpers

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/pagination.py`
- Modify: `tldw_Server_API/app/api/v1/utils/pagination.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py`
- Test: `tldw_Server_API/tests/Utils/test_pagination_contract.py`

- [x] **Step 1: Add failing alias-helper tests**

Extend `test_pagination_contract.py` with test-only response models that prove:

- offset aliases default from canonical metadata
- explicit matching aliases are accepted
- explicit contradictory aliases are rejected when using the strict helper
- page aliases can default from `PagePaginationMeta`
- cursor aliases can default from `CursorPaginationMeta`

Run:

```bash
python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
```

Expected: new tests fail because page/cursor/default strict helpers do not exist yet.

- [x] **Step 2: Implement schema-level helpers**

In `schemas/pagination.py`, add helpers along these lines:

```python
def default_offset_pagination_aliases(response: Any) -> Any: ...
def validate_offset_pagination_aliases(response: Any) -> Any: ...
def default_page_pagination_aliases(response: Any) -> Any: ...
def default_cursor_pagination_aliases(response: Any) -> Any: ...
```

Keep helpers generic and Pydantic-model-friendly. Do not import endpoint modules.

- [x] **Step 3: Keep metadata builders stable**

In `utils/pagination.py`, preserve current builder behavior and add tests before any behavior change. Do not change `build_offset_pagination_meta`, `build_page_pagination_meta`, or `build_cursor_pagination_meta` semantics without a failing test.

- [x] **Step 4: Rerun focused tests**

Run:

```bash
python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
python -m bandit -r tldw_Server_API/app/api/v1/schemas/pagination.py tldw_Server_API/app/api/v1/utils/pagination.py tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py -f json -o /tmp/bandit_pagination_helpers.json
git diff --check
```

Expected: tests pass, Bandit has zero findings.

- [x] **Step 5: Commit helper consolidation**

```bash
git add tldw_Server_API/app/api/v1/schemas/pagination.py tldw_Server_API/app/api/v1/utils/pagination.py tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py tldw_Server_API/tests/Utils/test_pagination_contract.py
git commit -m "Phase pagination-completion: consolidate pagination helpers"
```

## Task 3: Reduce Duplicate Offset Alias Helpers

**Files:**
- Modify schema files that define local `_default_offset_pagination_aliases`
- Test: touched route-family tests plus `tldw_Server_API/tests/Utils/test_pagination_contract.py`

- [x] **Step 1: Pick a small duplicate-helper tranche**

Start with low-risk schema-only replacements:

- `tldw_Server_API/app/api/v1/schemas/chat_grammar_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/outputs_templates_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/storage_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/prompt_schemas.py`

- [x] **Step 2: Replace local helpers with shared imports**

For each file:

```python
from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
)
```

Then change validators to call `default_offset_pagination_aliases(self)`.

- [x] **Step 3: Run focused tests**

Run:

```bash
python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_grammar_endpoints.py -q
python -m pytest tldw_Server_API/tests/Storage/test_storage_endpoints.py -q
python -m pytest tldw_Server_API/tests/prompt_studio/integration/test_api_endpoints.py -q
git diff --check
```

Expected: behavior unchanged.

- [x] **Step 4: Commit duplicate-helper tranche**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/chat_grammar_schemas.py \
  tldw_Server_API/app/api/v1/schemas/outputs_templates_schemas.py \
  tldw_Server_API/app/api/v1/schemas/storage_schemas.py \
  tldw_Server_API/app/api/v1/schemas/prompt_schemas.py
git commit -m "Phase pagination-completion: reuse offset alias helper"
```

## Task 4: Migrate Prompt Studio Page Family

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/prompt_studio_base.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_projects.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_test_cases.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py`
- Test: `tldw_Server_API/tests/prompt_studio/integration/test_api_endpoints.py`

**Tranche boundary after plan review:** Do not rework Prompt Studio routes that already return canonical `PageListResponse` unless a new test exposes a bug. In the first tranche, target only covered gaps:

- `prompt_studio_projects.py`: preserve `response_model=None` and the legacy top-level `projects` alias unless a route-specific schema explicitly includes it.
- `prompt_studio_optimization.py:list_optimizations`: additive `pagination` is safe because the route already returns a list envelope.

Defer `list_optimization_iterations` to custom-envelope handling unless the tranche adds a route-specific response schema that preserves `data={"iterations": ...}` and adds canonical `pagination` without filtering existing fields.

- [x] **Step 1: Add focused failing tests**

Add tests for each selected Prompt Studio list endpoint proving:

- legacy `metadata` remains present
- canonical `pagination.mode == "page"`
- `page`, `per_page`, `total`, `total_pages`, and `has_more` are correct
- missing/partial backend pagination does not raise `KeyError`
- legacy route-specific aliases, such as top-level `projects`, are not removed

Run:

```bash
python -m pytest tldw_Server_API/tests/prompt_studio/integration/test_api_endpoints.py -k "pagination or list" -q
```

Expected: newly added canonical assertions fail for endpoints not yet migrated.

- [x] **Step 2: Use shared page builder**

In each endpoint, build `PagePaginationMeta` with:

```python
pagination=build_page_pagination_meta(
    page=resolved_page,
    per_page=resolved_per_page,
    total=resolved_total,
    total_pages=resolved_total_pages,
)
```

Resolve missing backend metadata defensively before constructing responses.

- [x] **Step 3: Rerun Prompt Studio tests and Bandit**

Run:

```bash
python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py tldw_Server_API/tests/prompt_studio/integration/test_api_endpoints.py -k "pagination_contract or list_projects_safely_defaults_missing_pagination or list_optimizations_safely_defaults_missing_pagination or list_prompts_safely_defaults_missing_pagination" -q
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/prompt_studio tldw_Server_API/app/api/v1/schemas/prompt_studio_base.py -f json -o /tmp/bandit_pagination_prompt_studio.json
git diff --check
```

If the full Prompt Studio integration file times out in the app/TestClient lifespan harness, record that as baseline harness debt and keep the tranche scoped to direct covered endpoint tests. Do not broaden this pagination tranche into startup/shutdown lifecycle fixes.

- [ ] **Step 4: Commit Prompt Studio page tranche**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/prompt_studio_base.py \
  tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py \
  tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_projects.py \
  tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_test_cases.py \
  tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py \
  tldw_Server_API/tests/prompt_studio/integration/test_api_endpoints.py
git commit -m "Phase pagination-completion: migrate prompt studio page pagination"
```

## Task 5: Migrate Paper and Research Page Family

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/paper_search_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/research_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/paper_search.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/research.py`
- Test: `tldw_Server_API/tests/Research/test_paper_search_endpoints.py`

- [ ] **Step 1: Inventory provider-compatible exceptions**

In `Docs/Design/Pagination_Contract_Exemptions.md`, mark provider-shaped responses where the public payload must remain provider-compatible.

- [ ] **Step 2: Add red tests for safe page metadata**

For first-party page responses, assert:

- nested `pagination.mode == "page"`
- legacy top-level page fields remain
- `has_more` is correct when only `total` and `per_page` are known

Run:

```bash
python -m pytest tldw_Server_API/tests/Research/test_paper_search_endpoints.py -q
```

- [ ] **Step 3: Migrate only covered routes**

Use `build_page_pagination_meta` only for routes covered by tests. Leave uncovered/provider-specific branches unchanged unless tests are added first.

- [ ] **Step 4: Verify**

Run:

```bash
python -m pytest tldw_Server_API/tests/Research/test_paper_search_endpoints.py -q
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/paper_search.py tldw_Server_API/app/api/v1/endpoints/research.py tldw_Server_API/app/api/v1/schemas/paper_search_schemas.py tldw_Server_API/app/api/v1/schemas/research_schemas.py -f json -o /tmp/bandit_pagination_research_page.json
git diff --check
```

- [ ] **Step 5: Commit research page tranche**

```bash
git add tldw_Server_API/app/api/v1/endpoints/paper_search.py tldw_Server_API/app/api/v1/endpoints/research.py tldw_Server_API/app/api/v1/schemas/paper_search_schemas.py tldw_Server_API/app/api/v1/schemas/research_schemas.py tldw_Server_API/tests/Research/test_paper_search_endpoints.py Docs/Design/Pagination_Contract_Exemptions.md
git commit -m "Phase pagination-completion: migrate research page pagination"
```

## Task 6: Migrate Media Page Family

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/media_response_models.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/document_outline.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/reading_progress.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/listing.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/versions.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/navigation.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/document_outline.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py`
- Test: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_versions.py`
- Test: media listing/navigation tests identified by inventory

**Tranche note after inventory review:** Media list, trash, and POST search already emit canonical page fields in their `pagination` payloads and have tests asserting `mode`, `per_page`, `total`, and `has_more`. The first remaining small media gap is `/media/metadata-search`, which preserves legacy `page`, `per_page`, `total`, and `total_pages` but needs canonical `mode` and `has_more`.

- [ ] **Step 1: Confirm media test coverage from inventory**

Use the matrix to identify exact tests for each media route before editing. If a route lacks coverage, add focused route tests first or defer it.

- [ ] **Step 2: Add canonical page assertions**

For covered media page responses, assert nested `PagePaginationMeta` and legacy fields.

- [ ] **Step 3: Migrate one media subfamily at a time**

Recommended order:

1. versions
2. listing
3. navigation/outline
4. reading progress

Do not combine all media files into one PR unless the diff remains small.

- [ ] **Step 4: Verify**

Run focused media tests found in Step 1, then:

```bash
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/media tldw_Server_API/app/api/v1/schemas/media_response_models.py -f json -o /tmp/bandit_pagination_media_page.json
git diff --check
```

- [ ] **Step 5: Commit media page tranche**

```bash
git add \
  <exact touched media endpoint files> \
  <exact touched media schema files> \
  <exact touched media test files>
git commit -m "Phase pagination-completion: migrate media page pagination"
```

## Task 7: Migrate Audio Cursor Family

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/audio_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_history.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py`
- Test: `tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py`
- Test: audio jobs tests identified by inventory

**Status:** Complete for the current tranche. `audio_history.py` and `audio_schemas.py`
were already canonical, so this tranche only changed `audio_jobs.py` after adding
direct async coverage for the admin list cursor contract.

- [x] **Step 1: Add cursor metadata tests**

Assert:

- `pagination.mode == "cursor"`
- `next_cursor` stays opaque
- `has_more == bool(next_cursor)` unless explicitly overridden
- invalid cursor errors keep existing HTTP detail
- legacy top-level cursor aliases remain, if present

Run:

```bash
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py -q
```

- [x] **Step 2: Migrate audio history using `build_cursor_pagination_meta`**

Keep the existing cursor encoder/decoder and safe fallback behavior from #1159. Only change response metadata construction.

No change was needed in this branch: audio history already returns
`CursorPaginationMeta` via `build_cursor_pagination_meta`.

- [x] **Step 3: Add audio jobs only if covered**

If audio jobs lacks direct tests, add route tests before changing payloads.

Added direct async tests for admin list overfetch/trim, invalid cursors, and
accepting a returned cursor before changing `audio_jobs.py`.

- [x] **Step 4: Verify**

Run:

```bash
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py -q
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/audio tldw_Server_API/app/api/v1/schemas/audio_schemas.py -f json -o /tmp/bandit_pagination_audio_cursor.json
git diff --check
```

For this small tranche, Bandit was intentionally run on touched source scope:

```bash
python -m pytest tldw_Server_API/tests/AudioJobs/test_audio_jobs_admin_sanitization.py -q
python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py -q
python -m py_compile tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py -f json -o /tmp/bandit_pagination_audio_jobs.json
git diff --check
```

- [x] **Step 5: Commit audio cursor tranche**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/audio_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_history.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py \
  <exact touched audio jobs test files>
git commit -m "Phase pagination-completion: migrate audio cursor pagination"
```

## Task 8: Migrate Workflows and Jobs Cursor Family

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workflows.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workflows.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py`
- Test: workflow tests identified by inventory
- Test: jobs admin tests identified by inventory

**Status:** Classified for the current branch. `GET /api/v1/workflows/runs`
already returns canonical `OffsetPaginationMeta` or `CursorPaginationMeta`.
`GET /jobs/list` and `GET /api/v1/workflows/runs/{run_id}/events` are raw-list
payloads, and `GET /jobs/events/stream` is an SSE stream, so they are documented
as exemptions rather than migrated in-place.

- [x] **Step 1: Identify cursor semantics**

Record for each route:

- cursor field name (`cursor`, `after`, `after_id`)
- sort key
- invalid cursor behavior
- whether existing `Link` headers are part of the contract

Findings:

- `/api/v1/workflows/runs`: supports offset and cursor; sort key is
  token-carried `order_by` plus `run_id`; invalid cursors are ignored by
  existing behavior; `Link` header is part of the compatibility surface.
- `/api/v1/workflows/runs/{run_id}/events`: raw list body with `Next-Cursor` and
  `Link` headers; body migration is deferred to a versioned/object-envelope
  route.
- `/jobs/list`: raw list body; body migration is deferred to a versioned/object
  envelope route.
- `/jobs/events/stream`: streaming route with `after_id`; not a canonical body
  pagination target.

- [x] **Step 2: Add tests before route changes**

Add tests for stable ordering and no duplicate pages. Prefer overfetch assertions to count queries.

Existing tests already cover `/api/v1/workflows/runs` cursor flow and workflow
events header cursor flow. No route source was changed in this classification
tranche.

- [x] **Step 3: Migrate covered routes**

Use `CursorPaginationMeta` for cursor routes and `OffsetPaginationMeta` for offset routes in mixed response models like `WorkflowRunListResponse`.

No route source migration was needed: the only covered object-envelope route in
this family is already canonical.

- [x] **Step 4: Verify**

Run focused workflow/jobs tests and:

```bash
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/workflows.py tldw_Server_API/app/api/v1/endpoints/jobs_admin.py tldw_Server_API/app/api/v1/schemas/workflows.py -f json -o /tmp/bandit_pagination_workflows_jobs_cursor.json
git diff --check
```

This tranche changed only docs/classification, so source Bandit was not
applicable. Verified:

```bash
python -m pytest tldw_Server_API/tests/Jobs/test_jobs_list_sorting_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_rbac_list_stale_sqlite.py -q
python -m pytest tldw_Server_API/tests/Workflows/test_runs_cursor_pagination.py tldw_Server_API/tests/Workflows/test_events_cursor_pagination.py -q
git diff --check
```

The jobs tests passed. In the combined workflows command,
`test_runs_cursor_pagination.py` passed, then
`test_events_cursor_pagination.py` timed out in TestClient lifecycle cleanup
while joining the anyio portal after app startup/background-worker shutdown.
No workflow route source was changed in this tranche.

- [x] **Step 5: Commit workflows/jobs cursor tranche**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/workflows.py \
  tldw_Server_API/app/api/v1/endpoints/jobs_admin.py \
  tldw_Server_API/app/api/v1/schemas/workflows.py \
  <exact touched workflow test files> \
  <exact touched jobs-admin test files>
git commit -m "Phase pagination-completion: migrate workflow cursor pagination"
```

## Task 9: Classify and Migrate Custom Legacy Envelopes

**Files:**
- Modify only one route family per PR from the custom-envelope list.
- Update: `Docs/Design/Pagination_Completion_Matrix.md`
- Update: `Docs/Design/Pagination_Contract_Exemptions.md`

**Status:** In progress. The watchlists tranche migrated bounded preview/test
responses and classified operation-result, aggregate-count, file-export, and
small catalog routes as non-pagination targets. Sandbox and chatbooks custom
families have also been classified: their true list endpoints are already
canonical, while artifact/snapshot subresources and job detail responses are
explicitly exempt. The first kanban workflow tranche migrated workflow event
and stale-claim recovery lists with overfetch-derived offset metadata and
classified checklist/detail/catalog/status routes that have no pagination
inputs. MCP Hub custom-envelope rows have been classified without source
changes: governance audit findings are generated filtered snapshots with no
page inputs, and the event stream is SSE where `limit` terminates the stream
rather than describing a response-body page.

- [x] **Step 1: Pick the first custom family**

Recommended first custom-family candidates:

1. `watchlists` because it has many list responses but clear domain boundaries.
2. `sandbox` because some admin list responses already use offset metadata.
3. `chatbooks` export/import jobs because job lists are bounded and testable.

Avoid starting with `mcp_hub_management.py` unless the inventory shows a very small covered route.

- [x] **Step 2: Decide migration vs exemption**

For each route in the family:

- If response model is a list/search/history/preview object envelope with
  bounded `items` and existing pagination or limit inputs, add canonical
  pagination without removing legacy fields.
- If response model is an operation-result object envelope, detail response,
  aggregate/count response, or bounded preview with no continuation semantics,
  classify it explicitly rather than assuming pagination from field names like
  `total`.
- If response model is raw `list[...]`, record as raw-list exempt unless adding a versioned/sibling route.
- If provider-compatible, record provider exemption.

Watchlists findings:

- `/watchlists/jobs/{job_id}/preview`, `/watchlists/sources/{source_id}/test`,
  and `/watchlists/sources/test` are bounded preview list envelopes. They now
  preserve `items`, `total`, `ingestable`, and `filtered` while adding canonical
  `OffsetPaginationMeta` with `offset=0` and `has_more=false`.
- `/watchlists/sources/bulk`, `/watchlists/sources/check-now`, and
  `/watchlists/sources/import` are operation-result envelopes, not pagination
  targets.
- `/watchlists/runs/export.csv` is a CSV export route.
- `/watchlists/items/smart-counts`, `/watchlists/templates`,
  `/watchlists/templates/{template_name}/versions`, and
  `/watchlists/{watchlist_id}/clusters` have no pagination inputs and are
  classified as aggregate/small-catalog non-pagination targets.
- Chatbook export/import job list routes already expose canonical offset
  metadata; single job routes are detail responses where `total_items` tracks
  job progress.
- Sandbox admin runs, usage, and idempotency routes already expose canonical
  offset metadata; artifacts, snapshots, and fallback guard routes have no
  pagination inputs and stay exempt.
- Kanban workflow events and stale-claim recovery routes now overfetch
  `limit + 1`, trim to the requested `limit`, and return canonical
  `OffsetPaginationMeta` with `total=None`.
- Kanban checklist/list detail responses, mutation results, status responses,
  and small nested catalogs without pagination inputs are classified as
  non-pagination targets.
- MCP Hub `/audit/findings` is a generated audit snapshot with filters and
  aggregate counts but no pagination inputs. MCP Hub `/events/stream` is an SSE
  stream; its optional `limit` is a stream cutoff rather than a page contract.

- [x] **Step 3: Add route tests**

Each migrated custom route needs tests for:

- legacy fields
- nested `pagination`
- page disjointness or cursor non-duplication
- unknown/expensive total behavior if `total=None`

- [x] **Step 4: Verify**

Run the family’s focused tests, Bandit on touched source, and `git diff --check`.

Verified watchlists tranche:

```bash
python -m pytest tldw_Server_API/tests/Watchlists/test_preview_endpoint_more.py -k canonical_pagination -q
python -m pytest tldw_Server_API/tests/Watchlists/test_preview_endpoint.py tldw_Server_API/tests/Watchlists/test_preview_endpoint_more.py -q
python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py -f json -o /tmp/bandit_pagination_watchlists_preview.json
```

Bandit produced zero findings for the touched watchlists source scope.

Verified kanban workflow tranche:

```bash
python -m pytest tldw_Server_API/tests/kanban/test_workflow_endpoints.py -k canonical_pagination -q
python -m pytest tldw_Server_API/tests/kanban/test_workflow_endpoints.py -q
python -m pytest tldw_Server_API/tests/kanban/test_workflow_transition_contract.py -q
python -m py_compile tldw_Server_API/app/api/v1/schemas/kanban_schemas.py tldw_Server_API/app/api/v1/endpoints/kanban/kanban_workflow.py tldw_Server_API/app/core/DB_Management/Kanban_DB.py
```

Verified audiobook project-list tranche:

```bash
python -m pytest tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py -k canonical_offset_pagination -q
python -m pytest tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py -q
```

Verified Prompt Studio evaluation list tranche:

```bash
python -m pytest tldw_Server_API/tests/prompt_studio/integration/test_api_endpoints.py -k "test_list_evaluations" -q
```

Verified Prompt Studio optimization iterations tranche:

```bash
python -m pytest tldw_Server_API/tests/prompt_studio/integration/test_api_endpoints.py -k "test_list_optimization_iterations_safely_defaults_missing_pagination" -q
```

Classified remaining Prompt Studio page-family rows:

- Project, prompt, optimization, and test-case list routes already expose
  canonical page pagination.
- Optimization and prompt history routes are bounded per-resource snapshots with
  no pagination inputs.

- [ ] **Step 5: Commit one family**

Use a family-specific message, for example:

```bash
git commit -m "Phase pagination-completion: classify watchlist pagination"
```

Repeat Task 9 for each custom family.

## Task 10: Add OpenAPI Pagination Guardrails

**Files:**
- Create: `tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py`
- Update: `Docs/Design/Pagination_Completion_Matrix.md`
- Update: `Docs/Design/Pagination_Contract_Exemptions.md`

- [ ] **Step 1: Write the failing OpenAPI guard test**

Test behavior:

- Load the FastAPI app OpenAPI schema.
- Find candidate list/search routes from the inventory matrix and cross-check against OpenAPI route metadata. Use route-name heuristics such as `List`, `Search`, `History`, `Runs`, `Jobs`, `Events`, or `Collections` only as a fallback and as a drift detector.
- For each route, assert one of:
  - response schema includes canonical `pagination`
  - route is in the explicit exemption document/list
  - route is not actually paginated and is documented as such

Run:

```bash
python -m pytest tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py -q
```

Expected: fail until exemptions are encoded.

- [ ] **Step 2: Encode exemptions**

Prefer a simple Python fixture or YAML/Markdown-derived list with fields:

- route
- method
- reason
- owner/follow-up

Do not hide broad wildcard exemptions in code.

- [ ] **Step 3: Rerun guardrails**

Run:

```bash
python -m pytest tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py -q
python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
git diff --check
```

- [ ] **Step 4: Commit OpenAPI guardrails**

```bash
git add tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py Docs/Design/Pagination_Completion_Matrix.md Docs/Design/Pagination_Contract_Exemptions.md
git commit -m "Phase pagination-completion: add pagination contract guardrails"
```

## Task 11: Frontend Pagination Typing Closeout

**Files:**
- Modify: `apps/packages/ui/src/services/response-envelope.ts`
- Modify route-specific frontend services only when they consume migrated routes.
- Test: route-specific Vitest files.

- [ ] **Step 1: Add shared pagination types**

Add or extend frontend types for:

```ts
type OffsetPaginationMeta = {
  mode: "offset";
  limit: number;
  offset: number;
  total: number | null;
  has_more: boolean;
  next_offset: number | null;
};

type PagePaginationMeta = {
  mode: "page";
  page: number;
  per_page: number;
  total: number | null;
  total_pages: number | null;
  has_more: boolean;
};

type CursorPaginationMeta = {
  mode: "cursor";
  limit: number;
  cursor: string | null;
  next_cursor: string | null;
  has_more: boolean;
};
```

- [ ] **Step 2: Add unwrap/parser tests**

Use fixtures for legacy-only, canonical-only, and combined legacy+canonical responses.

Run:

```bash
cd apps/packages/ui
bunx vitest run <pagination helper test file>
```

- [ ] **Step 3: Update route-specific clients**

Only update services whose backend routes were migrated in the same or prior PR.

- [ ] **Step 4: Verify frontend scope**

Run:

```bash
cd apps/packages/ui
bunx vitest run <touched test files>
```

If OpenAPI/type generation is part of the touched client path, also run the repo’s documented OpenAPI verification command.

- [ ] **Step 5: Commit frontend typing closeout**

```bash
git add <exact touched frontend service files> <exact touched frontend test files>
git commit -m "Phase pagination-completion: add frontend pagination metadata typing"
```

## Task 12: Final Documentation and Tracker Update

**Files:**
- Create or update: `Docs/API/Pagination.md`
- Update: `Docs/Design/Pagination_Completion_Matrix.md`
- Update: GitHub issue #1116 or successor tracker

- [ ] **Step 1: Write API pagination docs**

Document:

- offset metadata
- page metadata
- cursor metadata
- legacy field compatibility
- `total=None`
- provider/raw-list exemptions

- [ ] **Step 2: Verify docs links and markdown**

Run:

```bash
git diff --check
rg -n "PaginationMeta|OffsetPaginationMeta|PagePaginationMeta|CursorPaginationMeta" Docs/API Docs/Design
```

- [ ] **Step 3: Update tracker**

Use `gh issue edit` or issue comment to record:

- completed PRs
- remaining explicit exemptions
- any versioning-blocked raw-list routes

- [ ] **Step 4: Commit docs closeout**

```bash
git add Docs/API/Pagination.md Docs/Design/Pagination_Completion_Matrix.md Docs/Design/Pagination_Contract_Exemptions.md
git commit -m "Phase pagination-completion: document pagination contract"
```

## Final Verification Before Each PR

Run at minimum:

```bash
python -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -q
python -m pytest <focused touched tests> -q
python -m bandit -r <touched_python_source_paths> -f json -o /tmp/bandit_pagination_<tranche>.json
git diff --check
git status --short --branch
```

For frontend-touching PRs:

```bash
cd apps/packages/ui
bunx vitest run <touched test files>
```

For guardrail PRs:

```bash
python -m pytest tldw_Server_API/tests/Utils/test_pagination_openapi_contract.py -q
```

## PR Sequence

1. Inventory matrix and exemptions.
2. Shared helper consolidation.
3. Duplicate offset alias helper cleanup.
4. Prompt Studio page-family migration.
5. Research/paper-search page-family migration.
6. Media page-family migration.
7. Audio cursor-family migration.
8. Workflows/jobs cursor-family migration.
9. Custom-envelope route families, one PR per family.
10. OpenAPI pagination guardrails.
11. Frontend pagination typing closeout.
12. API docs and tracker closeout.

Do not merge a later route-family PR if it depends on unmerged helper changes. Rebase each tranche on latest `dev` after the previous tranche lands.
