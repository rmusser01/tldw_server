# Research Workspace Trust and Status Telemetry Runbook

Operational guidance for validating and troubleshooting Research Workspace trust/status UX instrumentation.

## Scope

This runbook covers:
- Workspace-level telemetry contract for trust/status UX.
- Validation gates for Stage 4 rollout readiness.
- Incident triage for provenance, status visibility, cancelability, and recovery flows.

Primary implementation references:
- `apps/packages/ui/src/utils/research-workspace-telemetry.ts`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/undo-manager.ts`
- `apps/packages/ui/src/store/workspace.ts`

The implementation references retain `workspace-playground` names as internal
compatibility identifiers; user-facing routes and labels should use
`/research-studio` and Research Studio.

Plan reference:
- `2026-05-23-research-workspace-trust-panel-api-wiring-plan.md`

## Capability-Aware Health Contract

Research Studio now has a backend-owned capability contract for action-level
status. Keep route entry permissive for reachable degraded backend health, then
gate or warn only at the relevant Research Studio action boundary.

Primary endpoint:

```text
GET /api/v1/research-studio/capabilities
```

Operational properties:
- Authenticated and permission-gated with `media.read`.
- Rate-limited under `research_studio.capabilities`.
- Derived server-side from local health collectors; the WebUI must not infer
  action capability from raw subsystem health payloads.
- Short-lived client cache via `ttl_seconds`; clients refresh before expensive
  generation actions when the payload is stale.
- Sanitized response fields only: stable status/mode/dependency/reason-code
  values, no raw exceptions, local paths, provider keys, or secret values.

Capability payload shape:

```json
{
  "status": "degraded",
  "ttl_seconds": 30,
  "capabilities": {
    "source_browse": {
      "status": "ready",
      "mode": "allow",
      "dependencies": ["database", "chacha_notes"]
    },
    "chat": {
      "status": "degraded",
      "mode": "warn",
      "dependencies": ["source_browse", "rag", "llm"],
      "reason_code": "rag_degraded"
    },
    "artifact_text_generation": {
      "status": "ready",
      "mode": "allow",
      "dependencies": ["source_browse", "llm"]
    },
    "slides_generation": {
      "status": "unavailable",
      "mode": "block",
      "dependencies": ["source_browse", "llm", "slides"],
      "reason_code": "slides_unavailable"
    },
    "audio_summary": {
      "status": "unknown",
      "mode": "warn",
      "dependencies": ["source_browse", "llm", "tts"],
      "reason_code": "tts_unknown"
    },
    "export_download": {
      "status": "ready",
      "mode": "allow",
      "dependencies": ["local_artifact_state"]
    },
    "sync_share": {
      "status": "unknown",
      "mode": "warn",
      "dependencies": ["sync"]
    }
  },
  "timestamp": "2026-05-13T00:00:00Z"
}
```

Stable values:
- `status`: `ready`, `degraded`, `unavailable`, or `unknown`.
- `mode`: `allow`, `warn`, or `block`.
- `reason_code`: optional machine-readable user-safe reason.
- `dependencies`: stable dependency identifiers, not raw implementation
  details.

Mode semantics:
- `allow`: action proceeds normally.
- `warn`: action remains available with degraded or unknown status copy.
- `block`: only the matching action is blocked; adjacent independent actions
  remain available.

### Current Health Sources

| Source | Current Payload Evidence | Research Studio Use | Limitations |
|---|---|---|---|
| `GET /api/v1/health` | Aggregate `status` plus `checks.database`, `checks.metrics`, and `checks.chacha_notes`; HTTP `206` is used for aggregate degraded state. | Broad app entry/status and source-health input to capability derivation. | Still not enough by itself for action-level gating. |
| `GET /api/v1/health/live` | Lightweight `status: alive`. | Safe for liveness/recovery probing only. | Liveness does not imply any Research Studio workflow capability. |
| `GET /api/v1/rag/health` | RAG `status` plus component statuses for cache, metrics, batch processor, and circuit breakers. | Input to chat and source-grounded generation capability derivation. | Does not prove LLM provider availability by itself. |
| `GET /api/v1/llm/health` | LLM subsystem `status` plus provider manager, queue, and rate-limiter details. | Input to chat, text artifact, slides, and audio summary capability derivation. | Does not prove source access, slides DB, or TTS provider availability by itself. |
| TTS config readiness collector | Configured provider count and enabled-provider count without provider initialization. | Input to Audio Summary capability derivation. | This intentionally avoids downloading or initializing TTS models; exact synthesis failures still surface at request time. |
| Slides DB collector | Per-user slides database availability. | Input to Slides output capability derivation. | Does not prove LLM generation or source retrieval readiness for slide content. |

### Capability Matrix

| Research Studio Capability | Frontend Behavior | Backend Contract Source |
|---|---|---|
| Open Research Studio route | Allow when aggregate health is `ok`, `healthy`, or `degraded`; continue to block only unreachable, malformed, or explicit unhealthy readiness states. | Implemented by `ServerReadinessGate`. |
| Browse local workspace shell, notes, and locally persisted selection state | Allow with broad degraded/disconnected status visible. | Local/browser state. |
| Browse backend-backed source details | Allow or warn from `source_browse`; request-level errors still own exact source-detail failures. | `source_browse`. |
| Chat over selected sources | `warn` keeps Send available with degraded copy; `block` disables Send without showing the generic disconnected banner. | `chat`. |
| Generate work products or raw text artifacts | Source selection remains first; then `warn` shows inline degraded copy and `block` disables text output buttons/regeneration. | `artifact_text_generation`. |
| Generate Slides | Slides blocks apply only to Slides and do not block text artifacts. | `slides_generation`. |
| Generate Audio Summary | Audio blocks apply only to Audio Summary and do not block text artifacts or slides. | `audio_summary`. |
| Export/download generated artifacts | Local downloads remain available when they do not need backend services; backend-dependent exports respect the capability. | `export_download`. |
| Sync/share collaborative state | Warn or block only from explicit sync/share capability state. | `sync_share`. |

### Frontend Implementation Decision

The frontend consumes the backend contract but does not duplicate derivation
rules. If the capability endpoint is unreachable, malformed, unauthorized, or
times out, Research Studio stays open when aggregate readiness already allowed
entry. Capabilities become `unknown/warn`, read-only UI remains available, and
request-level errors still own exact action failures.

The source-selection gate remains higher priority than capability gating. If no
source is selected, show no-source guidance first. Capability warnings and
blocks appear only after the user is otherwise eligible to attempt an action.

### Local Verification Checklist

Use this checklist for PR evidence when capability gating or generation
behavior changes. Do not paste secrets, full keys, raw provider error bodies, or
full generated artifact text into the PR.

1. Confirm authenticated backend access:
   - Load the existing local single-user or JWT credentials from saved config.
   - Verify `GET /api/v1/health` returns reachable `ok`, `healthy`, or
     `degraded` state.
   - Verify `GET /api/v1/research-studio/capabilities` returns `200` and all
     expected capability IDs.
2. Confirm route behavior with CDP or Playwright:
   - `/research-studio` renders the Research Studio shell.
   - `/workspace-playground` redirects or aliases to `/research-studio`.
   - `/workspace-studio?tab=studio` normalizes to
     `/research-studio?tab=studio`.
   - Mobile `/research-studio?tab=studio` opens the Studio tab.
3. Confirm real local text generation:
   - Use existing saved provider credentials; do not replace this with a mock
     provider or fake key.
   - Seed or select a small deterministic source.
   - Select the source in Research Studio.
   - Generate a `Summary` artifact.
   - Record provider family and model only when visible and non-sensitive.
   - Record source type/title, artifact type, completion status, output
     character count, and screenshot path if captured.
4. Document caveats:
   - If live generation fails because credentials are absent or invalid, record
     that as a PR verification blocker.
   - If Slides or Audio Summary are not fully configured, keep their live
     certification deferred and rely on capability-mode tests for this PR.

## Ownership and Escalation

- Product Analytics owner: validates KPI definitions and dashboard parity.
- Platform owner: validates instrumentation integrity and storage/recovery behavior.
- Research Workspace on-call engineer: handles runtime incidents and rollback execution.

Escalate as P1 immediately for:
- Confirmed user data loss.
- Repeated stuck generation states after reload.
- Conflict handling paths that overwrite data unexpectedly.

## Telemetry Storage and Access

Current UI contract persists telemetry to browser local storage key:
- `tldw:research-workspace:telemetry`

Use either:
- Workspace menu: `Workspaces -> Telemetry summary`.
- Browser console:

```js
JSON.parse(localStorage.getItem("tldw:research-workspace:telemetry") || "{}")
```

Telemetry summary now includes an ops-only rollout execution panel with:
- current rollout subject ID,
- one-click subject regeneration,
- per-flag rollout override presets (`0%`, `10%`, `50%`, `100%`),
- reset and refresh controls for local override state.

Reset telemetry snapshot (for test/verification only):

```js
localStorage.setItem(
  "tldw:research-workspace:telemetry",
  JSON.stringify({
    version: 1,
    counters: {
      status_viewed: 0,
      citation_provenance_opened: 0,
      token_cost_rendered: 0,
      diagnostics_toggled: 0,
      quota_warning_seen: 0,
      conflict_modal_opened: 0,
      undo_triggered: 0,
      operation_cancelled: 0,
      artifact_rehydrated_failed: 0,
      source_status_polled: 0,
      source_status_ready: 0,
      connectivity_state_changed: 0
    },
    last_event_at: null,
    recent_events: []
  })
)
```

## Event Contract

Stage 2 sign-off requires the baseline trust/status 12 events.
Stage 4 adds 3 confusion indicators for operational diagnostics.

| Event | Emitted From | Minimum Fields | Purpose |
|---|---|---|---|
| `status_viewed` | Workspace shell activity rail | `workspace_id`, `operations_count`, `status` | Tracks whether active processing/generation status is visible. |
| `citation_provenance_opened` | Chat citation click handler | `workspace_id`, `has_media_id` | Tracks provenance drill-down usage. |
| `token_cost_rendered` | Retrieval diagnostics render | `has_tokens`, `has_cost` | Verifies token/cost transparency renders. |
| `diagnostics_toggled` | Retrieval details expand/collapse | `expanded` | Tracks diagnostics discoverability and cognitive load. |
| `quota_warning_seen` | Quota warning listener | `workspace_id`, `reason` | Verifies proactive storage risk warning visibility. |
| `conflict_modal_opened` | Cross-tab conflict surface | `workspace_id`, `changed_fields_count` | Tracks conflict resolver activation. |
| `undo_triggered` | Shared undo manager | none | Tracks recovery usage after destructive actions. |
| `operation_cancelled` | Chat stop + artifact cancel | `workspace_id`, `operation`, optional `artifact_type` | Verifies cancel affordance usage and availability. |
| `artifact_rehydrated_failed` | Store rehydrate migration | `workspace_id`, `interrupted_count` | Tracks interrupted generation recovery path. |
| `source_status_polled` | Source status poll loop | `workspace_id`, `processing_count` | Verifies background polling activity. |
| `source_status_ready` | Poll completion transition | `workspace_id`, `media_id` | Tracks readiness transition from processing to ready. |
| `connectivity_state_changed` | Header connectivity indicator | `workspace_id`, `from`, `to` | Verifies backend health signal transitions. |
| `confusion_retry_burst` | Chat retry banner action | `workspace_id`, `retry_count`, `window_ms` | Detects repeated retry attempts over a short window. |
| `confusion_refresh_loop` | Cross-tab "Use latest" reload path | `workspace_id`, `refresh_count`, `window_ms` | Detects potential panic refresh loops caused by conflict uncertainty. |
| `confusion_duplicate_submission` | Chat submit handler | `workspace_id`, `duplicate_count`, `window_ms`, `source_scope_count`, `message_length` | Detects repeated duplicate prompts in a short interval. |

## Confusion Indicator Thresholds

- Retry burst: `3+` retry clicks within `30s`.
- Refresh loop: `3+` "Use latest" reload actions within `45s`.
- Duplicate submission: identical normalized prompt within identical source scope within `12s` (first duplicate signal at count `2`).

These signals are diagnostic indicators, not hard failures by themselves. Treat sustained increases as UX clarity regressions and correlate with support ticket tags.

## KPI Definitions and Targets

Use 14-day trailing window, minimum 500 eligible sessions per variant.

| KPI | Formula | Target |
|---|---|---|
| Citation provenance open rate | `citation_provenance_opened / sessions_with_citations` | `>= 40%` |
| Model/token transparency coverage | `token_cost_rendered / responses_with_usage_metadata` | `>= 95%` |
| Diagnostics overload signal | `diagnostics_toggled(expanded=false within 5s) / diagnostics_toggled(expanded=true)` | `<= 15%` |
| Quota warning lead coverage | `quota_exceeded_sessions_with_prior_warning / quota_exceeded_sessions` | `>= 95%` |
| Conflict resolver coverage | `conflict_modal_opened_with_action / conflict_modal_opened` | `>= 95%` |
| Cancel affordance success | `operation_cancelled_with_request_abort / operation_cancelled` | `>= 95%` |
| Rehydrate interruption recovery | `artifact_rehydrated_failed_followed_by_retry_or_clear / artifact_rehydrated_failed` | `>= 90%` |
| Undo discoverability | `undo_triggered / reversible_destructive_actions` | `>= 50%` |
| Retry burst incidence | `sessions_with_confusion_retry_burst / active_sessions` | `<= 5%` |
| Refresh loop incidence | `sessions_with_confusion_refresh_loop / sessions_with_conflicts` | `<= 3%` |
| Duplicate submission incidence | `sessions_with_confusion_duplicate_submission / active_sessions` | `<= 8%` |

Release pass rule:
- KPI targets hold for two consecutive weekly cuts.
- No rollout guardrail breach.

## Dashboard Query Cookbook

Use these query mappings in the ops dashboard and weekly trust review:

| Panel | Query | Notes |
|---|---|---|
| Retry burst incidence | `sessions_with_confusion_retry_burst / active_sessions` | Elevated values indicate repeated retry behavior under uncertainty. |
| Refresh loop incidence | `sessions_with_confusion_refresh_loop / sessions_with_conflicts` | Spikes indicate cross-tab conflict resolution confusion. |
| Duplicate submission incidence | `sessions_with_confusion_duplicate_submission / active_sessions` | Tracks repeated same-prompt submissions in short windows. |
| Retry burst trend (24h) | `count(confusion_retry_burst where timestamp >= now-24h)` | Use for on-call watch windows. |
| Refresh loop trend (24h) | `count(confusion_refresh_loop where timestamp >= now-24h)` | Pair with conflict modal volume. |
| Duplicate submission trend (24h) | `count(confusion_duplicate_submission where timestamp >= now-24h)` | Pair with backend latency/availability signals. |

In-product support:
- `Workspaces -> Telemetry summary` now exposes:
  - confusion indicator counters and rates,
  - 24h and 7d confusion event totals,
  - query-formula hints for dashboard parity.

## Export Workflow for Ops Review

From `Workspaces -> Telemetry summary`:

1. Click `Export JSON` to download full telemetry snapshot.
2. Click `Export confusion CSV` to download confusion-focused event rows.

Expected exports:
- JSON filename prefix: `workspace-telemetry-summary-`
- CSV filename prefix: `workspace-telemetry-confusion-`

CSV columns:
- `event_type`, `timestamp_iso`, `timestamp_ms`
- `workspace_id`, `operation`, `artifact_type`
- `retry_count`, `refresh_count`, `duplicate_count`
- `window_ms`, `source_scope_count`, `message_length`
- `details_json`

## Daily Operational Checks

1. Open `Workspaces -> Telemetry summary` on a fresh session and confirm counters increment for:
   - `status_viewed`
   - `diagnostics_toggled`
   - `operation_cancelled`
2. Trigger one source ingestion and confirm:
   - `source_status_polled` increments.
   - `source_status_ready` increments after readiness.
3. Trigger one connectivity transition (disconnect/reconnect backend) and confirm:
   - `connectivity_state_changed` includes `from` and `to`.
4. Trigger one destructive action with undo and confirm:
   - `undo_triggered` increments when Undo is clicked.

## Incident Triage Playbooks

### 1) Missing provenance or diagnostics telemetry

- Check `recent_events` in telemetry summary modal.
- Verify retrieval diagnostics panel renders and can be toggled.
- Verify citation click handler runs and source focus path resolves.
- If events still missing, inspect browser console for:
  - `[research-workspace-telemetry] Failed to record event`

### 2) Cancel action appears non-functional

- Verify `operation_cancelled` event increments.
- Verify request abort path:
  - Chat uses `stopStreamingRequest()`.
  - Artifact generation uses `AbortController`.
- If telemetry increments but operation does not stop, treat as client cancellation defect.

### 3) Retry spikes or duplicate prompt spikes

- Check `confusion_retry_burst` and `confusion_duplicate_submission` counts in telemetry summary.
- Verify connection banner retry CTA is not repeatedly shown due stale connectivity state.
- Verify submit disable/loading behavior is consistent during active processing/streaming.
- If spikes coincide with backend degradation, classify as infrastructure-induced confusion.
- If spikes persist without backend degradation, classify as UX clarity regression and open a product issue.

### 4) Cross-tab conflicts not surfacing

- Confirm storage events or broadcast channel updates fire.
- Confirm `conflict_modal_opened` increments when dual-tab writes occur.
- Validate modal options show with consequence text:
  - `Use latest`
  - `Keep mine`
  - `Fork copy`

### 5) Stuck generating artifacts after refresh

- On reload, inspect artifacts with prior `generating` state.
- Confirm they migrate to recoverable failed state.
- Confirm `artifact_rehydrated_failed` increments with `interrupted_count`.

### 6) Storage risk not visible before quota exceed

- Confirm header storage indicator is present and updates with usage.
- Confirm quota event listener emits `quota_warning_seen`.
- If warnings are absent before quota errors, treat as guardrail breach.

## Rollout and Rollback

Active rollout flags:
- `research_workspace_provenance_v1`
- `research_workspace_status_guardrails_v1`

Flag surface map:
- `research_workspace_provenance_v1` controls provenance-facing trust surfaces:
  - citation-to-source navigation,
  - retrieval diagnostics panel visibility,
  - model badge/model picker visibility,
  - provenance telemetry summary access in workspace menu.
- `research_workspace_status_guardrails_v1` controls status/recovery guardrails:
  - storage quota and cross-tab conflict warning surfaces,
  - source status polling loop,
  - global activity rail and connectivity/storage indicators,
  - retry/refresh-loop confusion telemetry instrumentation.

Cohort assignment controls:
- Rollout assignment is deterministic per client via local subject key:
  - `tldw:feature-rollout:subject-id:v1`
- Percentage resolution precedence for each rollout flag:
  1. Runtime window override (`window.__TLDW_RESEARCH_WORKSPACE_ROLLOUT__`)
  2. Local storage percentage override
  3. Build/runtime env percentage
  4. Default `100`
- Local storage percentage keys:
  - `tldw:feature-rollout:research_workspace_provenance_v1:percentage`
  - `tldw:feature-rollout:research_workspace_status_guardrails_v1:percentage`
- Environment percentage keys:
  - `VITE_RESEARCH_WORKSPACE_PROVENANCE_V1_ROLLOUT_PERCENTAGE`
  - `VITE_RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1_ROLLOUT_PERCENTAGE`
  - `NEXT_PUBLIC_RESEARCH_WORKSPACE_PROVENANCE_V1_ROLLOUT_PERCENTAGE`
  - `NEXT_PUBLIC_RESEARCH_WORKSPACE_STATUS_GUARDRAILS_V1_ROLLOUT_PERCENTAGE`

In-product execution controls (preferred for ops):
1. Open `Workspaces -> Telemetry summary`.
2. In `Rollout execution controls`, verify the current subject ID.
3. For each flag, choose `0%`, `10%`, `50%`, or `100%` to set local override.
4. Use `Regenerate subject` to simulate a different deterministic cohort assignment.
5. Use `Reset to 100%` to return local overrides to full rollout.
6. Use `Refresh` to re-read local storage after external/manual changes.

Operational note:
- These controls affect the local browser session overrides only and do not modify server-side global rollout configuration.

Ops quick controls (browser console):
- Set 10% cohort:
  - `localStorage.setItem("tldw:feature-rollout:research_workspace_provenance_v1:percentage", "10")`
  - `localStorage.setItem("tldw:feature-rollout:research_workspace_status_guardrails_v1:percentage", "10")`
- Expand to 50%:
  - same keys with `"50"`.
- Full rollout:
  - same keys with `"100"`.

Expansion gates:
- `10% -> 50% -> 100%` only when no data-loss incidents and conflict resolution success remains `>= 95%`.

Abort criteria:
- Any confirmed data-loss incident.
- Quota/conflict failure rate `> 5%` for 6 consecutive hours.
- Sustained guardrail breach for more than 24 hours after mitigation attempts.

Rollback procedure:
1. Disable both feature flags.
2. Revert to legacy conflict/undo behavior path.
3. Publish incident summary with user-facing recovery instructions.

## Verification Checklist

- [ ] Event contract includes all 12 required trust/status events.
- [ ] Telemetry summary modal exposes counters and recent event details.
- [ ] Reset path works for QA sessions.
- [ ] Cross-tab conflict handling emits telemetry and presents all actions.
- [ ] Cancel and undo actions emit expected telemetry.
- [ ] Rehydrate recovery emits interruption telemetry.
