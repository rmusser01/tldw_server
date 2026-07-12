# Research Discovery Phase 2A PDF Media Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` when the user explicitly requests subagents; otherwise use `superpowers:executing-plans`. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users submit selected stable PDF candidates from a persisted Research Discovery snapshot through the existing `/api/v1/media/add` endpoint.

**Architecture:** Research adds one internal selection-resolution function that reads the authenticated user's snapshot and returns immutable PDF descriptors. Media owns request validation, duplicate lookup, egress and download enforcement, PDF processing, persistence, and response outcomes. The existing Media PDF pipeline remains the only ingestion implementation.

**Tech Stack:** FastAPI multipart forms, Pydantic v2, ResearchSessionsDB, Media DB APIs, existing Media PDF/download utilities, pytest, Bandit.

**Source of truth:** `Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md`

---

## Boundaries

- Do not add a Research-owned ingestion endpoint, service, queue, job, worker, downloader, parser, duplicate checker, or persistence path.
- Keep `/api/v1/media/add` as the public synchronous Phase 2A handoff surface.
- Do not add another eligibility or recommendation field. Correct `ingest_eligible` and `recommended_candidate_id`.
- Accept only `pdf` candidates with `safe_url`, `url_redacted=false`, and `requires_reresolution=false`.
- Require `media_type=pdf`; do not make the existing required field conditionally optional.
- Reject normal URLs, uploaded files, and cookies in discovery mode. Preserve existing Media-owned PDF parsing, OCR, chunking, analysis, collection, and embedding controls.
- Preserve the existing `{"results": [...]}` response envelope.
- Do not implement HTML handling. Phase 2B requires real source-specific `html_full_text` candidates and a separately reviewed bounded Media extraction-to-persistence path.

## Stage 1: Latest-Dev Contract Gate

**Goal:** Start from a clean current base and revalidate the contracts this plan depends on.

**Success Criteria:** A fresh implementation Backlog task and isolated `codex/` worktree exist on current `origin/dev`; relevant baseline tests pass.

**Tests:** Contract inspection plus focused baseline tests.

**Status:** Complete

### Task 1: Create the implementation workspace

**Files:**
- Read: `tldw_Server_API/app/api/v1/endpoints/media/add.py`
- Read: `tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py`
- Read: `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- Read: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Read: `tldw_Server_API/app/core/Ingestion_Media_Processing/download_utils.py`

- [x] Use `superpowers:using-git-worktrees` to create a clean worktree without altering the dirty existing worktree. If native worktree setup is unavailable, first verify the fallback directory is ignored, then use:

```bash
git fetch origin dev
git check-ignore -q .worktrees
git worktree add .worktrees/codex-research-discovery-phase2a-pdf \
  -b codex/research-discovery-phase2a-pdf origin/dev
```

- [x] Create a separate Backlog implementation task linked to this plan and the spec before code edits.

- [x] Confirm `/media/add` still owns `MEDIA_CREATE`, rate limiting, quota, billing, Media DB, authenticated-user, and usage-log dependencies.

- [x] Confirm `AddMediaForm.media_type` is required, normal ingestion requires URLs/files, `download_url_async` owns redirect-aware egress and streamed byte enforcement, and Media DB exposes URL and safe-metadata identifier lookup.

- [x] Run the baseline suite.

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_identity.py \
  tldw_Server_API/tests/Research/test_research_discovery_service.py \
  tldw_Server_API/tests/Media/test_json_url_download.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_add_deps_error_mapping.py
```

Expected: PASS. Stop and revise the plan if any route or pipeline assumption is false; do not add another endpoint as a workaround.

## Stage 2: Eligibility and Selection Resolution

**Goal:** Make existing discovery eligibility truthful and add the minimal Research-owned snapshot lookup boundary.

**Success Criteria:** Only stable PDFs are recommended/eligible, and owner-scoped selections resolve in request order without Media side effects.

**Tests:** Eligibility, ownership, expiry, malformed snapshots, unsupported candidates, bounds, and order.

**Status:** Complete

### Task 2: Correct existing eligibility semantics

**Files:**
- Modify: `tldw_Server_API/app/core/Research/discovery/models.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/identity.py`
- Modify: `tldw_Server_API/app/core/Research/discovery/service.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_identity.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_service.py`
- Test: `tldw_Server_API/tests/Research/test_research_discovery_endpoint.py`

- [x] Write failing tests proving that only stable PDFs are eligible and the first eligible candidate by `(rank, candidate_id)` is recommended.
- [x] Include ineligible coverage for HTML, landing pages, repository files, metadata-only, redacted, re-resolution-required, and URL-less candidates.
- [x] Run the focused tests and confirm they fail under the current `any(candidate.safe_url)` behavior.
- [x] Add one helper after `DiscoveryOACandidate` and reuse it in identity construction and OA enrichment.

```python
PHASE2A_MEDIA_HANDOFF_TYPES = frozenset({"pdf"})


def is_phase2a_media_handoff_candidate(candidate: DiscoveryOACandidate) -> bool:
    return (
        candidate.candidate_type in PHASE2A_MEDIA_HANDOFF_TYPES
        and bool(candidate.safe_url)
        and not candidate.url_redacted
        and not candidate.requires_reresolution
    )
```

- [x] Set `recommended_candidate_id` from the first eligible candidate and `ingest_eligible` from whether it exists; add no response fields.
- [x] Run the focused Research tests and commit.

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_identity.py \
  tldw_Server_API/tests/Research/test_research_discovery_service.py \
  tldw_Server_API/tests/Research/test_research_discovery_endpoint.py
git add tldw_Server_API/app/core/Research/discovery/models.py \
  tldw_Server_API/app/core/Research/discovery/identity.py \
  tldw_Server_API/app/core/Research/discovery/service.py \
  tldw_Server_API/tests/Research/test_research_discovery_identity.py \
  tldw_Server_API/tests/Research/test_research_discovery_service.py \
  tldw_Server_API/tests/Research/test_research_discovery_endpoint.py
git commit -m "fix(research): align discovery eligibility with PDF handoff"
```

### Task 3: Resolve server-owned snapshot selections

**Files:**
- Create: `tldw_Server_API/app/core/Research/discovery/selection.py`
- Create: `tldw_Server_API/tests/Research/test_research_discovery_selection.py`

- [x] Write failing tests for valid ordered resolution and rejection of missing/expired/foreign snapshots, malformed payloads, missing or duplicate pairs, pair mismatches, unsupported candidates, and more than five selections.
- [x] Use the same `research_discovery_snapshot_unavailable` error for missing, expired, and foreign snapshots.
- [x] Implement one frozen descriptor and one function; do not add a class hierarchy, protocol, factory, or enum.

```python
@dataclass(frozen=True)
class ResolvedDiscoverySelection:
    result_id: str
    candidate_id: str
    fingerprint: str
    candidate_type: str
    url: str
    canonical_url: str | None
    title: str
    authors: tuple[str, ...]
    identifiers: dict[str, str]
    source_id: str
    provider: str
    access_status: str | None
    license_hint: str | None
    safe_metadata: dict[str, Any]


def resolve_discovery_selections(
    *,
    owner_user_id: str,
    discovery_id: str,
    selections: Sequence[tuple[str, str]],
    snapshot_db: ResearchSessionsDB | None = None,
) -> tuple[ResolvedDiscoverySelection, ...]:
    ...
```

- [x] Derive the production DB path with `DatabasePaths.get_research_sessions_db_path(owner_user_id)` when no test DB is supplied.
- [x] Read only server-owned `snapshot.response_json["results"]`; client values are selectors, never metadata authority.
- [x] Build identifiers from DOI, PMID, PMCID, arXiv id, and safe provider ids.
- [x] Run the new tests and commit.

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_selection.py
git add tldw_Server_API/app/core/Research/discovery/selection.py \
  tldw_Server_API/tests/Research/test_research_discovery_selection.py
git commit -m "feat(research): resolve discovery PDF selections for Media"
```

## Stage 3: Existing Media Form and Route

**Goal:** Accept discovery references on `/media/add` without weakening or duplicating its existing contract.

**Success Criteria:** Discovery mode parses deterministically, rejects competing input sources/credentials, and branches before normal URL/file validation.

**Tests:** Form parsing, route integration, dependency preservation, and absence of a Research ingest route.

**Status:** Complete

### Task 4: Add discovery form fields and Media validation

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/media_request_models.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py`
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/research_discovery_handoff.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_research_discovery_handoff.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_add_deps_error_mapping.py`

- [x] Write failing tests for paired discovery fields, JSON shape, non-empty ids, duplicate pairs, max five, `media_type=pdf`, URLs/files exclusion, and cookie rejection.
- [x] Prove existing PDF parser/OCR/chunking/analysis/collection/embedding fields remain accepted.
- [x] Add two optional form fields and matching `Form(...)` dependency parameters.

```python
research_discovery_id: str | None = Field(None, max_length=128)
research_discovery_selections: str | None = Field(None, max_length=8192)
```

- [x] In `research_discovery_handoff.py`, implement only JSON parsing, mode detection, and request validation helpers. Use `json.loads`; return stable 422 errors for malformed selections and 400 errors for conflicting source/credential inputs.
- [x] Run the unit tests.

### Task 5: Branch inside `/media/add`

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/add.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/research_discovery_handoff.py`
- Create: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_research_discovery_media_add.py`

- [x] Write failing tests proving discovery mode branches before `add_media_persist`, normal URL/file requests are unchanged, dependencies remain intact, and no `/api/v1/research/discovery/ingest` route exists.
- [x] Add one branch in the existing route and no router or endpoint.

```python
if is_research_discovery_handoff(form_data):
    return await add_research_discovery_pdfs(
        background_tasks=background_tasks,
        form_data=form_data,
        files=files,
        db=db,
        current_user=current_user,
        usage_log=usage_log,
        request=request,
    )
```

- [x] Leave the existing `add_media_persist(...)` call as the only non-discovery branch, run tests, and commit.

## Stage 4: Bounded PDF Pipeline Reuse

**Goal:** Resolve, preflight, ingest, and report selected PDFs through existing Media code.

**Success Criteria:** Media performs pre-download duplicate lookup, enforces strict byte/MIME policy, preserves trusted identifiers, and returns ordered outcomes.

**Tests:** Download utility, handoff unit, and route integration tests.

**Status:** Complete

### Task 6: Compose existing download limits safely

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/download_utils.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- Test: `tldw_Server_API/tests/Media/test_json_url_download.py`

- [x] Write failing tests proving an explicit 50 MiB cap cannot raise a smaller configured cap, oversized `Content-Length` fails before streaming, chunked overflow fails during streaming, and HTML/octet-stream responses are rejected despite a `.pdf` suffix.
- [x] Change `_resolve_max_bytes` to return the minimum when explicit and configured/inferred limits both exist.
- [x] Add optional `allowed_content_types: set[str] | None = None` to `download_url_async`; keep `None` as the unchanged default.
- [x] Thread these internal-only defaulted arguments through `add_media_persist`, `add_media_orchestrate`, and `process_document_like_item`:

```python
max_download_bytes: int | None = None
allowed_download_content_types: set[str] | None = None
trusted_source_metadata_by_url: Mapping[str, dict[str, Any]] | None = None
```

- [x] Pass limits into `download_url_async`. Merge trusted metadata immediately before the existing safe-metadata allowlist/persistence step. Never parse these values from client form fields.
- [x] Run existing download and add-media tests.

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Media/test_json_url_download.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_add_media_endpoint.py
```

### Task 7: Implement Media duplicate handling and outcomes

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/research_discovery_handoff.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_research_discovery_handoff.py`
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_research_discovery_media_add.py`

- [x] Write failing tests for owner-scoped resolution, duplicate normalized URLs within one request, URL duplicate lookup, identifier lookup through `search_by_safe_metadata(..., match_all=False)`, duplicate short-circuiting, ordered mixed outcomes, trusted metadata, no cookies, and safe logs/errors.
- [x] Resolve selections, create ordered result slots, and fill duplicate slots before download.
- [x] Reject duplicate normalized resolved URLs before building `trusted_source_metadata_by_url`; do not silently overwrite metadata for one candidate with another candidate's metadata.
- [x] Call `add_media_persist` once for remaining PDFs using `form_data.model_copy(update={"urls": resolved_urls, "use_cookies": False, "cookies": None})`, 50 MiB, `{application/pdf}`, and trusted metadata keyed by URL.
- [x] Decode the returned `JSONResponse`, map results back in active-selection order, and preserve all original result slots.
- [x] Add stable outcomes: `created`, `duplicate_existing`, `policy_blocked`, `timeout`, `unsupported`, and `failed`.
- [x] Include `result_id`, `candidate_id`, `outcome`, `db_id`, `media_uuid`, and a bounded safe error where applicable.
- [x] Return HTTP 200 when every item is created/duplicate; otherwise return 207. Shape/snapshot failures remain 4xx.
- [x] Run tests and commit.

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_research_discovery_handoff.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_research_discovery_media_add.py
git add tldw_Server_API/app/core/Ingestion_Media_Processing/download_utils.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/research_discovery_handoff.py \
  tldw_Server_API/tests/Media/test_json_url_download.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_research_discovery_handoff.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_research_discovery_media_add.py
git commit -m "feat(media): ingest bounded discovery PDF selections"
```

## Stage 5: Verification and Handoff

**Goal:** Verify the slice and leave HTML explicitly deferred.

**Success Criteria:** Focused tests pass, Bandit adds no findings, Backlog is current, and no Research ingestion or speculative HTML code exists.

**Tests:** Focused suite, compile check, route scan, diff check, and Bandit.

**Status:** In Progress

### Task 8: Final verification

- [ ] Run the complete focused suite.

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Research/test_research_discovery_identity.py \
  tldw_Server_API/tests/Research/test_research_discovery_service.py \
  tldw_Server_API/tests/Research/test_research_discovery_endpoint.py \
  tldw_Server_API/tests/Research/test_research_discovery_selection.py \
  tldw_Server_API/tests/Media/test_json_url_download.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_add_deps_error_mapping.py \
  tldw_Server_API/tests/MediaIngestion_NEW/unit/test_research_discovery_handoff.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_research_discovery_media_add.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_add_media_endpoint.py
```

- [ ] Compile, scan boundaries, and check the diff.

```bash
source .venv/bin/activate && python -m compileall -q \
  tldw_Server_API/app/core/Research/discovery \
  tldw_Server_API/app/api/v1/endpoints/media \
  tldw_Server_API/app/core/Ingestion_Media_Processing
rg -n "research/discovery/ingest|html_full_text" \
  tldw_Server_API/app/api/v1 \
  tldw_Server_API/app/core/Ingestion_Media_Processing
git diff --check
```

- [ ] Run Bandit on touched application scope.

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Research/discovery \
  tldw_Server_API/app/api/v1/endpoints/media/add.py \
  tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py \
  tldw_Server_API/app/api/v1/schemas/media_request_models.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/research_discovery_handoff.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/download_utils.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py \
  -f json -o /tmp/bandit_research_discovery_phase2a_pdf.json
```

- [ ] Confirm Research only resolves selections; Media owns duplicate lookup, egress, limits, parsing, persistence, and outcomes; `/media/add` is the only public Phase 2A handoff; and no new idempotency table, HTML path, queue, worker, or plugin abstraction exists.
- [ ] Update the implementation Backlog task with files, commits, test and Bandit results, limitations, and the Phase 2B deferral.
- [ ] Request code review, address validated findings, rerun verification, and commit final documentation/task updates.

```bash
git add Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md \
  Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md \
  backlog/tasks
git commit -m "docs: finalize Research Discovery PDF handoff"
```
