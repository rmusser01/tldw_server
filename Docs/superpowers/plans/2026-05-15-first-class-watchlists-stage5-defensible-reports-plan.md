# First-Class Watchlists Stage 5 Defensible Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Watchlists outputs into defensible research artifacts with immutable evidence snapshots, report readiness states, and report-builder UX while preserving the existing Markdown, HTML, Chatbook, and audio output paths.

**Architecture:** Reuse the current `POST /api/v1/watchlists/outputs` pipeline and Collections output artifacts as the report artifact system. Add a Watchlists-owned report evidence/readiness contract around each generated output instead of creating a parallel reports table. Store immutable report evidence snapshots as sidecar JSON files referenced from output metadata, and expose them through output-scoped APIs and the Reports tab.

**Tech Stack:** FastAPI, Pydantic, WatchlistsDatabase, Collections output artifacts, existing Watchlists output storage helpers, React, Ant Design, Zustand Watchlists store, Vitest, pytest, Bandit, real FastAPI + real Next WebUI with Playwright/CDP for browser verification.

---

## Source Spec

- Design spec: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Stage 4 plan: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md`
- Stage 5 planning task: `TASK-349.2`
- API docs: `Docs/API-related/Watchlists_API.md`
- Published API docs mirror: `Docs/Published/API-related/Watchlists_API.md`

## Current Evidence

- `POST /api/v1/watchlists/outputs` already creates output artifacts from a `run_id` and optional `item_ids`, renders Markdown/HTML templates, writes output files, optionally creates TTS/audio variants, optionally ingests to Media DB, and delivers to email/Chatbook.
- New Watchlists outputs already include metadata fields: `origin`, `watchlist_id`, `job_id`, `run_id`, `item_ids`, `version`, `template_name`, `template_version`, `template_source`, `format`, `type`, delivery state, retention, and audio state where applicable.
- `GET /api/v1/watchlists/outputs?watchlist_id=<id>` scopes outputs by the selected Watchlist's jobs and still includes legacy Watchlists outputs that only have a matching stored `job_id`.
- `GET /api/v1/watchlists/outputs/{output_id}/download` returns the rendered artifact content, but there is no separate evidence/provenance API for a report snapshot.
- `ItemsTab` can queue updates for reporting with `queued_for_briefing=true` and can create a report from queued items for a selected run by calling `createWatchlistOutput({ run_id, item_ids })`.
- `OutputsTab` currently lists generated reports, filters by monitor/run/delivery status, previews rendered content, downloads artifacts, regenerates an output, and shows lightweight provenance: monitor ID, run ID, artifact type, template, delivery status, Chatbook path, and storage path.
- Stage 3 content alerts already persist per-item alert evidence with severity, status, rule ID, matched text, snippet, and evidence JSON.
- Stage 4 item triage already exposes alert-aware filters, compact alert summaries, saved views, server-side review queue filtering/sorting, batch triage, and report queue state.
- The current product does not preserve an immutable included/excluded item trail, source diversity/readiness state, weak-evidence warnings, or a report-builder review step before artifact generation.

## Product Decisions For Stage 5

- User-facing copy should use "Reports" for generated artifacts and "Updates" for input items. "Briefing" remains a report preset/output style, not the whole object model.
- A report is still a Watchlists output artifact. Do not create a separate first-class `reports` persistence table unless output artifacts cannot meet the contract.
- Immutable evidence must be captured at output creation time. Legacy outputs can show "Live provenance only" when no snapshot exists, but new Stage 5 outputs must not silently reconstruct evidence from mutable current item rows.
- The evidence snapshot should include enough data for a future reader to understand what was included, why it mattered, where it came from, and what was excluded or unavailable.
- Report readiness is advisory, not a hard blocker. CTI users may need to publish with weak evidence; news users may need a quick briefing. The UI must explain warnings and let users proceed deliberately.
- CTI/OSINT and news should be supported through presets and labels, not separate code paths. The domain can be inferred from the selected Watchlist where useful, but users should be able to override the preset.
- Preserve existing Markdown/HTML/Chatbook/audio paths. Stage 5 should enrich the context/templates and metadata used by those paths.
- Constrained viewports must support full report management, including readiness review, evidence inspection, preset selection, generation, preview, regeneration, and download.

## Implementation Boundaries

- Do not rename `/api/v1/watchlists/outputs` or break existing output clients.
- Do not remove `item_ids`, `queued_for_briefing`, template, Chatbook, TTS, audio, or Media DB ingestion behavior.
- Do not use a mocked server for browser QA. Real-server smoke must run against real FastAPI and real WebUI.
- Do not use Computer Use for browser QA; use CDP/Playwright.
- Do not add LLM-based claim extraction, source credibility scoring, or novelty scoring in Stage 5 unless already backed by existing persisted data.
- Do not make report readiness block scheduled monitor delivery until the product explicitly decides that scheduled reports should be held for analyst review.
- Keep Stage 6 constrained-viewport redesign out of scope except for verifying that Stage 5 controls remain usable in an extension-sized viewport.
- Keep Stage 7 trust/calibration explanations out of scope except where required to explain Stage 5 readiness and evidence warnings.

## Proposed File Responsibilities

Backend:

- `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Add report preset, readiness, evidence snapshot, evidence item, source provenance, alert evidence, excluded item, and output evidence response models.
  - Extend `WatchlistOutputCreateRequest` with report-builder options that are backwards-compatible defaults.
  - Extend `WatchlistOutput` metadata typing only through optional fields, not by making legacy metadata invalid.
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Build report evidence/readiness from the selected run/item set before rendering.
  - Persist the immutable evidence snapshot sidecar during output creation and reference it from output metadata.
  - Add output-scoped endpoints for readiness/evidence retrieval.
  - Add report preset handling to output context so existing templates can include evidence tables and warnings.
- `tldw_Server_API/app/core/Watchlists/report_evidence.py`
  - New focused helper module for evidence snapshot construction, readiness scoring, weak-evidence warnings, and source diversity calculations.
  - Keep the helper deterministic and free of network/LLM calls.
- `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
  - Prefer existing item/source/run/alert methods. Add only narrow helpers if fetching alert/source evidence efficiently cannot be done with current methods.
- `tldw_Server_API/app/services/outputs_service.py`
  - Reuse existing output file/path helpers. Add no report-specific persistence unless the endpoint cannot safely write/read snapshot sidecars itself.
- `tldw_Server_API/app/core/Watchlists/templates/`
  - Update or add Stage 5 report presets that render evidence summaries and source provenance in Markdown/HTML.

Frontend:

- `apps/packages/ui/src/types/watchlists.ts`
  - Add report preset, readiness, evidence snapshot, evidence item, warning, and output evidence API types.
  - Extend `WatchlistOutputCreate` with report-builder options.
- `apps/packages/ui/src/services/watchlists.ts`
  - Add `getWatchlistOutputEvidence`, `getWatchlistOutputReadiness`, and output creation payload fields.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
  - Add helpers for report preset labels, readiness labels/colors, weak-evidence warnings, source diversity, included/excluded counts, and snapshot availability.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx`
  - Add a "Create report" entry point that starts from selected Watchlist/run/queued Updates.
  - Show readiness and evidence availability in the Reports table.
  - Keep existing list, preview, download, regenerate, and delivery workflows intact.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx`
  - New focused report-builder drawer for preset selection, title, format/template, included queued Updates, excluded/unavailable counts, readiness warnings, and generation.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportEvidencePanel.tsx`
  - New reusable panel for evidence table, source provenance, alerts, included/excluded trail, and snapshot metadata.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
  - Add a report evidence tab/section that loads immutable snapshot evidence when present and labels legacy/live-only provenance clearly.
- `apps/packages/ui/src/store/watchlists.tsx`
  - Add small state for report builder drawer and selected output evidence view only if component-local state becomes awkward.
- `apps/packages/ui/src/assets/locale/en/watchlists.json`
  - Add report-builder, evidence, readiness, weak-evidence, CTI preset, news preset, and legacy provenance copy.
- `apps/packages/ui/src/public/_locales/en/watchlists.json`
  - Mirror locale updates if still maintained manually.

Docs:

- `Docs/API-related/Watchlists_API.md`
  - Document report presets, evidence snapshot metadata, readiness states, new output evidence/readiness endpoints, and legacy-output behavior.
- `Docs/Published/API-related/Watchlists_API.md`
  - Mirror API docs.

Tests:

- `tldw_Server_API/tests/Watchlists/test_watchlist_reports_evidence.py`
  - New backend tests for evidence snapshot construction and readiness warnings.
- `tldw_Server_API/tests/Watchlists/test_watchlist_reports_api.py`
  - New API tests for output creation metadata, sidecar evidence retrieval, legacy fallback, scoping, and download compatibility.
- `apps/packages/ui/src/services/__tests__/watchlists-reports.test.ts`
  - Service tests for report-builder payloads and evidence/readiness endpoints.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx`
  - Builder tests for presets, warnings, queued item counts, generation payloads, and error states.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.test.tsx`
  - Evidence table, provenance, included/excluded trail, and empty/legacy snapshot states.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx`
  - Reports table readiness, preview evidence access, and constrained-layout smoke in component tests.

## Backlog Task Map For Implementation

Create implementation tasks before code changes:

- Stage 5A: Backend report evidence/readiness helper and schemas.
- Stage 5B: Output creation snapshot persistence and evidence APIs.
- Stage 5C: Frontend report evidence client contract and metadata helpers.
- Stage 5D: Reports tab builder, evidence review, and preview integration.
- Stage 5E: Report presets, docs, real-server CDP smoke, and closeout.

Keep commits aligned to these task groups.

## Task 0: Baseline And Task Setup

**Files:**
- Reference: `Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md`
- Reference: `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md`
- Reference: `Docs/API-related/Watchlists_API.md`
- Reference: `backlog/tasks/task-349.2 - Plan-Stage-5-Watchlist-defensible-reports.md`

- [ ] **Step 1: Create Stage 5 implementation Backlog tasks**

Use Backlog from the active worktree to create Stage 5A-5E tasks listed above. Each task must reference this plan, the design spec, Stage 4 plan, and API docs.

- [ ] **Step 2: Capture backend baseline**

Run:

```bash
rg -n "create_output|_build_output_context|_row_to_output|list_outputs|download_output|queued_for_briefing|list_content_alerts|include_alert_summary" \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/tests/Watchlists
```

Expected: current output creation, item queue handoff, output metadata, content alert evidence, and output listing paths are identified before edits.

- [ ] **Step 3: Capture frontend baseline**

Run:

```bash
rg -n "OutputsTab|OutputPreviewDrawer|createWatchlistOutput|fetchWatchlistOutputs|downloadWatchlistOutput|queued_for_briefing|queue-generate-report|Provenance|Regenerate" \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab \
  apps/packages/ui/src/components/Option/Watchlists/ItemsTab/ItemsTab.tsx \
  apps/packages/ui/src/services/watchlists.ts \
  apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/store/watchlists.tsx
```

Expected: current report generation from queued items, Reports table, preview drawer, download, regenerate, and delivery behavior are identified before edits.

- [ ] **Step 4: Run focused baseline tests**

Run backend tests that currently cover Watchlists outputs and triage:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py::test_watchlist_scopes_outputs_by_job_and_records_output_provenance \
  tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_watchlists_run_flow_rss \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py \
  -q
```

Run frontend tests that currently cover reports/outputs:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.smoke.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.focus-management.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.regenerate-modal.test.tsx \
  src/components/Option/Watchlists/ItemsTab/__tests__/ItemsTab.batch-controls.test.tsx
```

Expected: baseline is green before Stage 5 edits, or failures are recorded on the active Backlog task before continuing.

## Task 1: Stage 5A Backend Evidence And Readiness Contract

**Files:**
- Create: `tldw_Server_API/app/core/Watchlists/report_evidence.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlist_reports_evidence.py`

- [ ] **Step 1: Write failing helper tests**

Create tests for deterministic evidence and readiness behavior:

```python
def test_report_evidence_snapshot_marks_ready_with_diverse_sources_and_alerts():
    snapshot = build_report_evidence_snapshot(
        watchlist_id=1,
        job=job_row(1, "Hospital ransomware monitor"),
        run=run_row(10, "finished"),
        included_items=[
            item_row(101, source_id=11, title="Advisory", url="https://a.example/cve"),
            item_row(102, source_id=12, title="Local report", url="https://b.example/news"),
        ],
        excluded_items=[item_row(103, source_id=11, title="Ignored", queued_for_briefing=False)],
        sources={11: source_row(11, "Vendor advisory"), 12: source_row(12, "Local news")},
        alerts={101: [alert_row(201, "critical", "new", rule_name="CVE exploit")]},
        preset="cti_osint",
    )

    assert snapshot["readiness"]["state"] == "ready"
    assert snapshot["source_summary"]["unique_source_count"] == 2
    assert snapshot["included_items"][0]["alerts"][0]["severity"] == "critical"
    assert snapshot["excluded_items"][0]["reason"] == "not_queued_for_report"
```

Also test warning states:

- no included items -> `blocked`.
- one source only -> `warning` with `single_source`.
- included items without URL/source -> `warning` with `missing_source_provenance`.
- CTI preset with alert-bearing report expected but no alert matches -> `warning` with `no_alert_evidence`.
- news preset with stale `published_at` values -> `warning` with `stale_updates`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_reports_evidence.py -q
```

Expected: FAIL because `report_evidence.py` and schemas do not exist yet.

- [ ] **Step 3: Add Pydantic schemas**

Add schema models in `watchlists_schemas.py`:

```python
WatchlistReportPreset = Literal["auto", "cti_osint", "news_briefing", "general_research"]
WatchlistReportReadinessState = Literal["ready", "warning", "blocked", "legacy_live_only"]

class WatchlistReportReadinessWarning(BaseModel):
    code: str
    severity: Literal["info", "warning", "blocking"] = "warning"
    message: str
    affected_item_ids: list[int] = Field(default_factory=list)

class WatchlistReportReadiness(BaseModel):
    state: WatchlistReportReadinessState
    score: int = Field(ge=0, le=100)
    warnings: list[WatchlistReportReadinessWarning] = Field(default_factory=list)

class WatchlistReportEvidenceAlert(BaseModel):
    id: int
    rule_id: int
    rule_name: str | None = None
    severity: str
    status: str
    title: str | None = None
    snippet: str | None = None
    matched_text: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None

class WatchlistReportEvidenceItem(BaseModel):
    id: int
    title: str | None = None
    url: str | None = None
    source_id: int | None = None
    source_name: str | None = None
    published_at: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    reviewed: bool = False
    queued_for_briefing: bool = False
    alerts: list[WatchlistReportEvidenceAlert] = Field(default_factory=list)

class WatchlistReportExcludedItem(BaseModel):
    id: int
    title: str | None = None
    url: str | None = None
    reason: str

class WatchlistReportEvidenceSnapshot(BaseModel):
    schema_version: int = 1
    snapshot_id: str
    generated_at: str
    preset: WatchlistReportPreset
    watchlist_id: int | None = None
    job_id: int
    run_id: int
    output_id: int | None = None
    included_items: list[WatchlistReportEvidenceItem] = Field(default_factory=list)
    excluded_items: list[WatchlistReportExcludedItem] = Field(default_factory=list)
    source_summary: dict[str, Any] = Field(default_factory=dict)
    readiness: WatchlistReportReadiness
```

Add response wrappers:

```python
class WatchlistOutputEvidenceResponse(BaseModel):
    output_id: int
    immutable_snapshot: bool
    snapshot: WatchlistReportEvidenceSnapshot | None = None
    readiness: WatchlistReportReadiness
```

- [ ] **Step 4: Implement deterministic helper**

Create `report_evidence.py` with pure functions:

```python
def build_report_evidence_snapshot(
    *,
    watchlist_id: int | None,
    job: Any,
    run: Any,
    included_items: Sequence[Any],
    excluded_items: Sequence[Any],
    sources: Mapping[int, Any],
    alerts: Mapping[int, Sequence[Any]],
    preset: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    ...
```

Helper responsibilities:

- Normalize row objects and Pydantic models through `getattr`/mapping-safe access.
- Preserve `id`, `title`, `url`, `source_id`, `source_name`, `published_at`, `summary`, `tags`, `reviewed`, `queued_for_briefing`, and compact alert evidence.
- Compute `source_summary.unique_source_count`, `source_summary.missing_source_count`, `source_summary.hosts`, and `source_summary.top_sources`.
- Compute `included_count`, `excluded_count`, `alert_count`, `critical_alert_count`.
- Produce stable `snapshot_id` from watchlist/run/item IDs and generation timestamp.
- Keep outputs JSON-serializable and deterministic for tests.

Implement `evaluate_report_readiness(snapshot_like)` in the same module:

- `blocked` when included count is zero.
- `warning` for single-source reports, missing source URLs, missing alert evidence for CTI preset, stale published dates for news preset, and unreviewed queued items.
- `ready` when there are included items and no warnings above warning threshold.
- Score starts at 100 and subtracts deterministic penalties; clamp to 0-100.

- [ ] **Step 5: Run helper tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_reports_evidence.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Stage 5A**

Commit:

```bash
git add \
  tldw_Server_API/app/core/Watchlists/report_evidence.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_reports_evidence.py \
  backlog/tasks/<stage-5a-task-file>
git commit -m "feat: add watchlist report evidence contract"
```

## Task 2: Stage 5B Output Snapshot Persistence And Evidence APIs

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify as needed: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlist_reports_api.py`

- [ ] **Step 1: Write failing API tests**

Cover:

- Creating an output with `report_preset="cti_osint"` stores `metadata.report_preset`, `metadata.report_snapshot_path`, `metadata.report_readiness`, `metadata.included_item_count`, `metadata.excluded_item_count`, `metadata.source_count`, and `metadata.alert_count`.
- `GET /api/v1/watchlists/outputs/{output_id}/evidence` returns the immutable snapshot and `immutable_snapshot=true`.
- `GET /api/v1/watchlists/outputs/{output_id}/readiness` returns the readiness subset without downloading rendered content.
- Legacy Watchlists output with no snapshot returns `immutable_snapshot=false` and a `legacy_live_only` readiness state rather than 500.
- Output evidence endpoints reject non-Watchlists artifacts and cross-user artifacts.
- Markdown/HTML download and Chatbook/audio metadata still work.

Use real test DB fixtures and existing output creation routes. Do not mock the Watchlists API layer.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_reports_api.py -q
```

Expected: FAIL because endpoints and snapshot persistence are not implemented.

- [ ] **Step 3: Extend create request schema**

Add backwards-compatible optional fields to `WatchlistOutputCreateRequest`:

```python
report_preset: WatchlistReportPreset = Field(default="auto")
include_evidence_table: bool = Field(default=True)
include_excluded_items: bool = Field(default=True)
require_reviewed_items: bool = Field(default=False)
allow_weak_evidence: bool = Field(default=True)
```

Compatibility rules:

- Existing clients that omit these fields behave as today, but new outputs still receive a default evidence snapshot.
- `report_preset="auto"` resolves from the Watchlist domain when available; otherwise use `general_research`.
- `require_reviewed_items=true` does not block artifact generation unless `allow_weak_evidence=false`; otherwise it emits readiness warnings.

- [ ] **Step 4: Add snapshot storage helpers**

In `watchlists.py`, add small local helpers near output helpers:

```python
def _build_report_snapshot_filename(output_title: str, ts: str) -> str:
    return _build_output_filename(output_title, "evidence", ts, "json")

def _write_report_snapshot_for_user(user_id: int, filename: str, snapshot: dict[str, Any]) -> None:
    path = _resolve_output_path_for_user(user_id, filename)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

def _load_report_snapshot_for_user(user_id: int, storage_name: str) -> dict[str, Any] | None:
    path = _resolve_output_path_for_user(user_id, storage_name)
    return json.loads(path.read_text(encoding="utf-8"))
```

Use existing `_resolve_output_path_for_user` path safety; do not accept arbitrary absolute paths.

- [ ] **Step 5: Build snapshot before rendering**

Inside `create_output`, after items/job/run/watchlist are resolved and before template rendering:

- Resolve the effective report preset.
- Fetch source rows for included item `source_id` values.
- Fetch content alerts for included item IDs using existing DB methods or add one narrow DB helper if necessary.
- Identify excluded items from the same run:
  - `not_queued_for_report` for same-run items not in explicit `item_ids` when `item_ids` were provided from the report queue.
  - `filtered_or_error` for non-ingested same-run items where available.
  - Keep excluded item snapshot compact; cap to a documented max such as 200 with `excluded_items_truncated=true`.
- Build the evidence snapshot and readiness.
- Add snapshot/readiness to template context:

```python
context["report"] = {
    "preset": effective_report_preset,
    "readiness": snapshot["readiness"],
    "source_summary": snapshot["source_summary"],
    "included_items": snapshot["included_items"],
    "excluded_items": snapshot["excluded_items"],
}
```

- [ ] **Step 6: Persist snapshot after primary output row exists**

Because the snapshot should include `output_id`, persist it immediately after the primary artifact row is created:

- Add `output_id` to the snapshot.
- Write sidecar JSON in the user's outputs directory.
- Update output artifact metadata with:
  - `report_preset`
  - `report_schema_version`
  - `report_snapshot_path`
  - `report_readiness`
  - `included_item_count`
  - `excluded_item_count`
  - `source_count`
  - `alert_count`
  - `weak_evidence_warning_count`
- If later delivery/audio metadata updates occur, merge instead of replacing these report metadata fields.
- If any later variant creation fails and `_cleanup_outputs()` runs, remove the snapshot sidecar too.

- [ ] **Step 7: Add output evidence/readiness endpoints**

Add routes after output detail/download routes:

```python
@router.get("/outputs/{output_id}/evidence", response_model=WatchlistOutputEvidenceResponse)
async def get_output_evidence(...):
    ...

@router.get("/outputs/{output_id}/readiness", response_model=WatchlistReportReadiness)
async def get_output_readiness(...):
    ...
```

Route behavior:

- Validate output exists and `metadata.origin == "watchlists"`.
- For outputs with `report_snapshot_path`, read immutable sidecar and return it.
- For legacy outputs, return `immutable_snapshot=false`, `snapshot=None`, and readiness state `legacy_live_only` with an explanatory info warning.
- For missing sidecar path, return 404 `report_snapshot_missing` unless metadata marks the output as legacy.

- [ ] **Step 8: Run API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_reports_api.py -q
```

Expected: PASS.

- [ ] **Step 9: Run existing output regressions**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py::test_watchlist_scopes_outputs_by_job_and_records_output_provenance \
  tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_watchlists_run_flow_rss \
  -q
```

Expected: PASS.

- [ ] **Step 10: Commit Stage 5B**

Commit:

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_reports_api.py \
  backlog/tasks/<stage-5b-task-file>
git commit -m "feat: persist watchlist report evidence snapshots"
```

## Task 3: Stage 5C Frontend Evidence Client Contract

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/services/watchlists.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
- Test: `apps/packages/ui/src/services/__tests__/watchlists-reports.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts`

- [ ] **Step 1: Write failing frontend contract tests**

Cover:

- `createWatchlistOutput` accepts report preset/readiness options.
- `getWatchlistOutputEvidence(outputId)` calls `/api/v1/watchlists/outputs/{id}/evidence`.
- `getWatchlistOutputReadiness(outputId)` calls `/api/v1/watchlists/outputs/{id}/readiness`.
- Metadata helpers return labels/colors/counts for `ready`, `warning`, `blocked`, and `legacy_live_only`.
- Snapshot helpers handle absent legacy metadata without throwing.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/watchlists-reports.test.ts \
  src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
```

Expected: FAIL because frontend contract does not exist yet.

- [ ] **Step 3: Add TypeScript types**

Add types matching backend schema:

```ts
export type WatchlistReportPreset = "auto" | "cti_osint" | "news_briefing" | "general_research"
export type WatchlistReportReadinessState = "ready" | "warning" | "blocked" | "legacy_live_only"

export interface WatchlistReportReadinessWarning {
  code: string
  severity: "info" | "warning" | "blocking"
  message: string
  affected_item_ids: number[]
}

export interface WatchlistReportReadiness {
  state: WatchlistReportReadinessState
  score: number
  warnings: WatchlistReportReadinessWarning[]
}

export interface WatchlistReportEvidenceSnapshot {
  schema_version: number
  snapshot_id: string
  generated_at: string
  preset: WatchlistReportPreset
  watchlist_id?: number | null
  job_id: number
  run_id: number
  output_id?: number | null
  included_items: WatchlistReportEvidenceItem[]
  excluded_items: WatchlistReportExcludedItem[]
  source_summary: Record<string, unknown>
  readiness: WatchlistReportReadiness
}
```

Extend `WatchlistOutputCreate` with optional Stage 5 fields.

- [ ] **Step 4: Add service functions**

Add to `watchlists.ts`:

```ts
export const getWatchlistOutputEvidence = async (
  outputId: number
): Promise<WatchlistOutputEvidenceResponse> => bgRequest({ ... })

export const getWatchlistOutputReadiness = async (
  outputId: number
): Promise<WatchlistReportReadiness> => bgRequest({ ... })
```

- [ ] **Step 5: Add metadata helpers**

In `outputMetadata.ts`, add:

- `getOutputReportReadiness(metadata)`
- `getOutputReportPreset(metadata)`
- `getOutputReportSnapshotAvailable(metadata)`
- `getReadinessTagColor(state)`
- `getReadinessLabel(state)`
- `getWeakEvidenceWarningCount(metadata)`
- `getIncludedItemCount(metadata)`
- `getExcludedItemCount(metadata)`
- `getSourceCount(metadata)`
- `getAlertCount(metadata)`

Helpers must use defensive parsing and return safe defaults for legacy outputs.

- [ ] **Step 6: Run frontend contract tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/watchlists-reports.test.ts \
  src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit Stage 5C**

Commit:

```bash
git add \
  apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/services/watchlists.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts \
  apps/packages/ui/src/services/__tests__/watchlists-reports.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts \
  backlog/tasks/<stage-5c-task-file>
git commit -m "feat: add watchlist report evidence client contract"
```

## Task 4: Stage 5D Reports Builder And Evidence Review UI

**Files:**
- Create: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx`
- Create: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportEvidencePanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
- Modify as needed: `apps/packages/ui/src/store/watchlists.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/watchlists.json`
- Modify: `apps/packages/ui/src/public/_locales/en/watchlists.json`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.focus-management.test.tsx`

- [ ] **Step 1: Write failing UI tests**

Builder tests:

- Opens from Reports toolbar with "Create report".
- Defaults preset from selected Watchlist domain when available.
- Requires a run before generation.
- Shows queued update count and empty queue guidance.
- Shows readiness warnings from API response.
- Calls `createWatchlistOutput` with `report_preset`, `include_evidence_table`, `include_excluded_items`, title, format/template, and queued `item_ids`.
- Keeps full management usable at 420px width by stacking controls.

Evidence panel tests:

- Renders included evidence rows with title, source, date, URL, alert severity, and review/queue state.
- Renders excluded item trail with reason labels.
- Renders source diversity and snapshot timestamp.
- Shows legacy live-only state when `immutable_snapshot=false`.
- Shows missing snapshot error state with retry.

Output tab tests:

- Reports table shows readiness tag, source count, alert count, and evidence availability.
- Preview drawer can open an evidence section for outputs with snapshots.
- Regenerate keeps existing behavior and can opt into Stage 5 metadata defaults.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx
```

Expected: FAIL because UI components are not implemented.

- [ ] **Step 3: Build `ReportEvidencePanel`**

Component requirements:

- Accept `outputId`, optional preloaded evidence response, and `compact` mode.
- Load immutable snapshot through `getWatchlistOutputEvidence`.
- Render a compact readiness summary at the top.
- Render source summary and evidence table with stable columns:
  - Update
  - Source
  - Published
  - Alert evidence
  - Review/queue state
  - Link
- Render excluded item trail behind progressive disclosure.
- Use Ant Design `Table` only where it fits; switch to list/detail rows below narrow widths using CSS classes already used in Watchlists.
- Handle loading, error, empty, legacy, and missing-snapshot states.

- [ ] **Step 4: Build `ReportBuilderDrawer`**

Component requirements:

- User can choose preset: `cti_osint`, `news_briefing`, `general_research`.
- User can choose run, title, format/template, include evidence table, include excluded item trail, generate TTS/audio where current output creation supports it.
- It must surface report readiness before final generation. If no readiness endpoint exists for a draft, compute a client preflight from queued items as an approximation and label it "Preflight"; after generation, trust backend snapshot readiness.
- It must fetch queued Updates for the selected run using existing item filters, not local page state.
- It must show "Proceed with warnings" when readiness is warning and `allow_weak_evidence=true`.
- It must prevent generation only when readiness is blocking or no run/items are selected.
- It must send `item_ids` for the queued included items so the backend can snapshot the same set.

- [ ] **Step 5: Integrate `OutputsTab`**

Add:

- Primary toolbar action: "Create report".
- Readiness column in advanced mode and compact readiness badge in core title column.
- Evidence availability action from each row.
- Banner for reports with blocking/missing snapshot states.
- Empty-state CTA that points to Updates queue when no reports exist.

Preserve:

- Existing job/run/delivery filters.
- Existing preview/download/regenerate actions.
- Existing delivery issue banner.
- Existing focus restoration behavior.

- [ ] **Step 6: Integrate `OutputPreviewDrawer`**

Add:

- Evidence/provenance section or tab after template/provenance metadata.
- Lazy-load evidence only when drawer opens and user expands evidence.
- Keep download/chat actions unchanged.
- Make audio outputs show evidence metadata even when rendered text content is unavailable.

- [ ] **Step 7: Add copy**

Suggested copy:

- `Create report`
- `Report readiness`
- `Ready`
- `Needs review`
- `Blocked`
- `Evidence snapshot`
- `Immutable snapshot captured at {{time}}`
- `Live provenance only`
- `This older report was created before evidence snapshots were available.`
- `Weak evidence`
- `Only one source is represented. Add corroborating updates or proceed with a warning.`
- `No alert evidence`
- `This CTI report has no matching alert evidence.`
- `Included updates`
- `Excluded trail`
- `Proceed with warnings`
- `Generate defensible report`
- `Open Updates queue`

- [ ] **Step 8: Run UI tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.smoke.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.focus-management.test.tsx
```

Expected: PASS.

- [ ] **Step 9: Commit Stage 5D**

Commit:

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportEvidencePanel.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx \
  apps/packages/ui/src/store/watchlists.tsx \
  apps/packages/ui/src/assets/locale/en/watchlists.json \
  apps/packages/ui/src/public/_locales/en/watchlists.json \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.focus-management.test.tsx \
  backlog/tasks/<stage-5d-task-file>
git commit -m "feat: add watchlist report builder"
```

## Task 5: Stage 5E Presets, Docs, Real-Server QA, And Closeout

**Files:**
- Modify: `tldw_Server_API/app/core/Watchlists/templates/*` as needed
- Modify: `Docs/API-related/Watchlists_API.md`
- Modify: `Docs/Published/API-related/Watchlists_API.md`
- Modify as needed: locale files touched by Stage 5D
- Update: `backlog/tasks/<stage-5-task-files>`

- [ ] **Step 1: Add or update report presets**

Preserve existing template names where possible. If new templates are needed, add:

- CTI/OSINT report preset with:
  - Executive summary
  - Key findings
  - Evidence table
  - Alert matches
  - Source provenance
  - Gaps and weak-evidence warnings
  - Included/excluded trail
- News briefing preset with:
  - What changed
  - Timeline/recency
  - Source diversity
  - People/organizations/topics if present in tags or summaries
  - Follow-up links
  - Evidence caveats

Templates must consume `report.readiness`, `report.source_summary`, `report.included_items`, and `report.excluded_items` from the output context.

- [ ] **Step 2: Document API**

Update API docs with:

- `POST /outputs` Stage 5 fields and examples.
- Evidence snapshot metadata fields.
- `GET /outputs/{output_id}/evidence`.
- `GET /outputs/{output_id}/readiness`.
- Legacy/live-only provenance behavior.
- Report readiness warning codes.
- CTI and news examples.
- Compatibility note: Markdown/HTML/Chatbook/audio still flow through output artifacts.

Mirror `Docs/Published/API-related/Watchlists_API.md`.

- [ ] **Step 3: Run full focused verification**

Backend:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlist_reports_evidence.py \
  tldw_Server_API/tests/Watchlists/test_watchlist_reports_api.py \
  tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py::test_watchlist_scopes_outputs_by_job_and_records_output_provenance \
  tldw_Server_API/tests/Watchlists/test_watchlist_items_triage_api.py \
  -q
```

Frontend:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/watchlists-reports.test.ts \
  src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts \
  src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/ReportEvidencePanel.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.smoke.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.focus-management.test.tsx
```

Docs/static:

```bash
git diff --check
cmp -s Docs/API-related/Watchlists_API.md Docs/Published/API-related/Watchlists_API.md
```

Bandit on touched backend scope:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/app/core/Watchlists/report_evidence.py \
  tldw_Server_API/app/core/DB_Management/Watchlists_DB.py \
  -f json -o /tmp/bandit_watchlists_stage5_reports.json
```

Expected: no new findings in touched code.

- [ ] **Step 4: Real-server CDP smoke**

Use the real server and real WebUI. Do not mock the server.

Minimum smoke:

1. Start real FastAPI on an available local port with a real test database.
2. Start real Next WebUI on an available local port pointed at that API.
3. Seed or create through API:
   - one CTI Watchlist with two sources, one monitor, one finished run, two queued Updates, one unqueued same-run Update, and one critical content alert.
   - one news Watchlist with two sources, one monitor, one finished run, queued Updates, and no critical alert.
4. Open `/watchlists?tab=outputs` in Playwright/CDP.
5. Create a CTI report from queued Updates.
6. Confirm readiness appears, weak-evidence warnings are meaningful, and evidence snapshot opens.
7. Confirm the included evidence table contains source/title/link/alert evidence.
8. Confirm excluded trail includes the unqueued Update.
9. Preview and download the report.
10. Create or inspect a news briefing preset and confirm source diversity/recency language fits news use.
11. Repeat the core create/preview/evidence flow at approximately 420x760 viewport.
12. Capture screenshots under `/private/tmp/tldw-watchlists-stage5/`.
13. Stop all servers and verify ports are clear.

Record observed console/network errors in the Backlog task. Rate-limit warnings from rapid dev reloads are acceptable only if the flow remains usable and the issue is documented as a follow-up.

- [ ] **Step 5: Close Backlog tasks**

For each Stage 5 task:

- Check all acceptance criteria.
- Add verification notes.
- Add known skips/blockers.
- Add final summary.
- Mark Done.

- [ ] **Step 6: Commit Stage 5E**

Commit:

```bash
git add \
  tldw_Server_API/app/core/Watchlists/templates \
  Docs/API-related/Watchlists_API.md \
  Docs/Published/API-related/Watchlists_API.md \
  backlog/tasks/<stage-5-task-files>
git commit -m "docs: close watchlist defensible reports stage"
```

## Stage 5 Exit Criteria

- Users can create a report from queued Updates without leaving `/watchlists`.
- New reports preserve an immutable evidence snapshot separate from mutable current item state.
- Reports expose readiness state and warning reasons before and after generation.
- CTI/OSINT users can see alert evidence, source provenance, included/excluded item trail, and weak-evidence warnings.
- News users can see recency/source-diversity context and follow-up evidence links.
- Existing Markdown, HTML, download, regenerate, Chatbook, TTS, and audio output paths remain compatible.
- Legacy outputs are clearly labeled when they have live-only provenance rather than immutable snapshots.
- Reports table and preview drawer remain keyboard/focus-safe and usable in constrained viewports.
- API docs and published docs describe the Stage 5 contract.
- Focused backend/frontend tests, Bandit, docs checks, and real-server CDP smoke are recorded.

## Known Deferrals To Later Stages

- Full constrained-viewport redesign across every Watchlists tab is Stage 6.
- Rich trust/calibration explanations, source credibility scoring, and analyst confidence calibration are Stage 7 unless backed by current evidence fields.
- LLM claim extraction, deduplicated claims, and contradiction detection require a separate enrichment model and should not be smuggled into Stage 5 helper logic.
- Scheduled delivery gating based on readiness requires a product decision. Stage 5 only surfaces readiness and lets users proceed deliberately.
- Cross-report comparison, report diffing, and historical trend dashboards are outside Stage 5.
