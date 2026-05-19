# Monitoring Backend Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Monitoring backend review and deliver one findings-first, evidence-backed report covering correctness, auth and permission issues, state-model and source-of-truth risks, persistence and backend-compatibility hazards, operational concerns, maintainability problems, and test gaps across the scoped monitoring backend surface.

**Architecture:** This is a read-first, risk-first review plan. Execution starts by locking the dirty-worktree baseline and final report contract, then inspects API and schema boundaries, then traces alert identity and overlay state, then inspects the core topic-monitoring and notification code, and only after that runs targeted pytest slices to confirm or weaken candidate findings. No repository source changes are part of execution; the deliverable is the final in-session review plus a prioritized follow-up plan.

**Tech Stack:** Python 3, FastAPI, SQLite, PostgreSQL, pytest, git, rg, sed, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not only `HEAD`
- label any finding that depends on uncommitted local changes
- keep code scope inside topic monitoring services, the monitoring and admin-monitoring backend routes, the admin overlay helper, and the admin monitoring repo
- exclude Guardian/self-monitoring, claims monitoring, metrics internals, and all monitoring UI code unless a backend contract issue forces a brief reference
- treat `tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py` as an in-scope persistence edge, not as a general AuthNZ review
- inspect auth dependency code only where needed to validate monitoring-specific permission or principal behavior
- separate `Confirmed finding`, `Probable risk`, `Improvement`, and `Open question` in working notes, even if the final response groups items differently
- do not modify repository source files during the review itself
- use the smallest targeted pytest slices needed to confirm a concrete claim
- treat PostgreSQL-only monitoring repo coverage as optional verification: run it if the fixture environment is available, otherwise call out the resulting confidence limit explicitly
- keep blind spots explicit instead of implying excluded or unverified areas are safe

## Review File Map

**No repository source files should be modified during execution.**

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-07-monitoring-backend-review-design.md`
- `Docs/superpowers/plans/2026-04-07-monitoring-backend-review-execution-plan.md`

**Primary implementation files to inspect first:**
- `tldw_Server_API/app/core/Monitoring/README.md`
- `tldw_Server_API/app/core/Monitoring/__init__.py`
- `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`
- `tldw_Server_API/app/core/Monitoring/notification_service.py`
- `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
- `tldw_Server_API/app/api/v1/endpoints/monitoring.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py`
- `tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py`
- monitoring-specific definitions in `tldw_Server_API/app/api/v1/schemas/admin_schemas.py` only:
  `AdminAlertAssignRequest`, `AdminAlertEscalateRequest`, `AdminAlertEventResponse`,
  `AdminAlertHistoryListResponse`, `AdminAlertRuleCreateRequest`,
  `AdminAlertRuleCreateResponse`, `AdminAlertRuleDeleteResponse`,
  `AdminAlertRuleListResponse`, `AdminAlertRuleResponse`,
  `AdminAlertSnoozeRequest`, `AdminAlertStateMutationResponse`,
  and `AdminAlertStateResponse`
- `tldw_Server_API/app/services/admin_monitoring_alerts_service.py`
- `tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py`

**Direct-edge files to inspect only if an active trace requires them:**
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/core/AuthNZ/permissions.py`
- `tldw_Server_API/app/core/AuthNZ/principal_model.py`
- `tldw_Server_API/app/core/AuthNZ/database.py`
- `tldw_Server_API/app/main.py`

**Primary tests to inspect and selectively run:**
- `tldw_Server_API/tests/Monitoring/test_topic_monitoring.py`
- `tldw_Server_API/tests/Monitoring/test_notification_service.py`
- `tldw_Server_API/tests/Monitoring/test_notification_endpoint.py`
- `tldw_Server_API/tests/Monitoring/test_monitoring_notifications_settings.py`
- `tldw_Server_API/tests/Monitoring/test_monitoring_root_resolution.py`
- `tldw_Server_API/tests/Admin/test_admin_monitoring_alerts_service.py`
- `tldw_Server_API/tests/Admin/test_admin_monitoring_api.py`
- `tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py`
- `tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py`

**Optional PostgreSQL-backed verification if the test fixture environment is available:**
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_admin_monitoring_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_monitoring_repo_postgres.py`

**Nearby tests to inspect only if a concrete claim requires them:**
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_mcp_monitoring_invariants.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_monitoring_metrics_summary.py`

**Scratch artifacts allowed during execution:**
- `/tmp/monitoring_backend_review_inventory.txt`
- `/tmp/monitoring_backend_core_pytest.log`
- `/tmp/monitoring_backend_admin_pytest.log`
- `/tmp/monitoring_backend_auth_pytest.log`
- `/tmp/monitoring_backend_postgres_pytest.log`

## Stage Overview

## Stage 1: Baseline and Review Contract
**Goal:** Lock the dirty-worktree baseline, exact scope, and final response structure before deep reading starts.
**Success Criteria:** The active code surface, test surface, review exclusions, and final findings contract are fixed before candidate findings are recorded.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: API and Schema Boundary Pass
**Goal:** Inspect monitoring and admin-monitoring routes, schemas, startup hooks, audit behavior, and permission assumptions.
**Success Criteria:** Candidate findings about route semantics, auth boundaries, error shaping, and contract drift are tied to exact files and tests.
**Tests:** Read route and permission tests after the static pass; defer execution to Stage 5.
**Status:** Not Started

## Stage 3: Alert Identity and Overlay State Pass
**Goal:** Trace how persisted topic alerts become admin-facing identities and how overlay state and event history mutate those alerts.
**Success Criteria:** Source-of-truth, identity-model, and write-order risks are traced end to end with exact evidence.
**Tests:** Read overlay and repo tests after the static pass; defer execution to Stage 5.
**Status:** Not Started

## Stage 4: Core Topic Monitoring and Notification Pass
**Goal:** Inspect watchlist loading, rule compilation, dedupe, alert creation, notification delivery, and path-resolution behavior.
**Success Criteria:** Candidate findings about runtime behavior, concurrency, config/path handling, and operational safety are captured with exact file references.
**Tests:** Read topic-monitoring and notification tests after the static pass; defer execution to Stage 5.
**Status:** Not Started

## Stage 5: Persistence, Backend Compatibility, and Targeted Verification
**Goal:** Validate DB semantics, backend parity, schema bootstrapping, and the highest-value candidate findings using targeted pytest slices.
**Success Criteria:** Major claims are either supported by executed verification or explicitly downgraded in confidence.
**Tests:** Only the targeted pytest slices named in this plan, plus a directly adjacent test when needed to settle a disputed invariant.
**Status:** Not Started

## Stage 6: Final Synthesis and Test-Gap Pass
**Goal:** Convert evidence into one final report that distinguishes confirmed defects, probable risks, improvements, and blind spots.
**Success Criteria:** Every major claim in the final output is tied to source inspection, test inspection, executed verification, or an explicit confidence limit.
**Tests:** No new tests unless a final disputed claim still requires one narrow slice.
**Status:** Not Started

### Task 1: Lock the Baseline and Final Output Contract

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-monitoring-backend-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-monitoring-backend-review-execution-plan.md`
- Inspect: `tldw_Server_API/app/core/Monitoring`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/monitoring.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py`
- Test: none

- [ ] **Step 1: Capture the dirty-worktree baseline**

Run:
```bash
git status --short
```

Expected: a list of uncommitted files, making it clear whether any monitoring-related files already differ from committed history.

- [ ] **Step 2: Record the commit baseline used for the review**

Run:
```bash
git rev-parse --short HEAD
```

Expected: one short commit hash to cite when a finding depends on committed behavior rather than only local edits.

- [ ] **Step 3: Capture recent churn in the monitoring backend surface**

Run:
```bash
git log --oneline -n 20 -- \
  tldw_Server_API/app/core/Monitoring \
  tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py \
  tldw_Server_API/app/api/v1/endpoints/monitoring.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py \
  tldw_Server_API/app/services/admin_monitoring_alerts_service.py \
  tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py
```

Expected: a recent history snapshot showing where this backend surface has been actively changing and which files deserve extra skepticism during review.

- [ ] **Step 4: Enumerate the exact source review surface**

Run:
```bash
cat > /tmp/monitoring_backend_review_inventory.txt <<'EOF'
tldw_Server_API/app/core/Monitoring/__init__.py
tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py
tldw_Server_API/app/core/Monitoring/notification_service.py
tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py
tldw_Server_API/app/api/v1/endpoints/monitoring.py
tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py
tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py
tldw_Server_API/app/api/v1/schemas/admin_schemas.py (monitoring-specific definitions only)
tldw_Server_API/app/services/admin_monitoring_alerts_service.py
tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py
EOF
cat /tmp/monitoring_backend_review_inventory.txt
```

Expected: one stable file inventory for the scoped backend review surface, with `admin_schemas.py` explicitly narrowed to only the monitoring-specific definitions.

- [ ] **Step 5: Enumerate the primary test surface before deep reading**

Run:
```bash
printf '%s\n' \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py \
  tldw_Server_API/tests/Monitoring/test_notification_service.py \
  tldw_Server_API/tests/Monitoring/test_notification_endpoint.py \
  tldw_Server_API/tests/Monitoring/test_monitoring_notifications_settings.py \
  tldw_Server_API/tests/Monitoring/test_monitoring_root_resolution.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_alerts_service.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_api.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py \
  tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py
```

Expected: the exact test files that anchor later verification choices, without drifting into claims monitoring or Guardian coverage.

- [ ] **Step 6: Freeze the final response contract before recording findings**

Use this exact final response structure:
```markdown
## Findings
- severity-ordered items with confidence, type, impact, reasoning, and file references

## Structural Improvements
- targeted design or maintainability changes that materially reduce future defects

## Open Questions / Assumptions
- only unresolved items that materially affect confidence

## Test / Docs Gaps
- missing invariants, misleading tests, and meaningful documentation drift

## Prioritized Next Steps
- ordered follow-up actions that reduce the highest-risk uncertainty or defect exposure first

## Verification
- files inspected, tests run, and what remains unverified
```

- [ ] **Step 7: Confirm excluded surfaces stay excluded**

Run:
```bash
printf '%s\n' \
  tldw_Server_API/app/core/Monitoring/self_monitoring_service.py \
  tldw_Server_API/app/api/v1/endpoints/self_monitoring.py \
  tldw_Server_API/tests/Guardian/test_self_monitoring.py \
  tldw_Server_API/tests/Claims/test_claims_monitoring_api.py
```

Expected: a short explicit excluded-surface list that prevents scope creep during execution.

### Task 2: Execute the API and Schema Boundary Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/api/v1/endpoints/monitoring.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py`
- Inspect: monitoring-specific definitions in `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
- Inspect: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_monitoring_api.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py`

- [ ] **Step 1: Map the route surface and startup hooks**

Run:
```bash
rg -n "router = APIRouter|@router\\.|on_event\\(\"startup\"\\)" \
  tldw_Server_API/app/api/v1/endpoints/monitoring.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py
```

Expected: the exact route and startup anchors for the monitoring and admin-monitoring backend surfaces.

- [ ] **Step 2: Read the monitoring endpoint file in behavioral slices**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/api/v1/endpoints/monitoring.py
sed -n '261,620p' tldw_Server_API/app/api/v1/endpoints/monitoring.py
```

Capture:
- route semantics for watchlists, alerts, read or acknowledge, dismiss, settings, and test notification
- permission dependency shape and route-level responsibility boundaries
- audit emission behavior and error shaping

Expected: a candidate finding list for API semantics, auth assumptions, and contract drift.

- [ ] **Step 3: Read the admin monitoring endpoint file in full control-plane order**

Run:
```bash
sed -n '1,420p' tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py
```

Capture:
- alert rule CRUD semantics
- assign, snooze, escalate, and history behavior
- whether overlay actions require the same existence guarantees as topic-alert actions

Expected: a candidate finding list for control-plane authority and route-to-state mismatches.

- [ ] **Step 4: Read the monitoring-specific schemas that define the backend contract**

Run:
```bash
sed -n '1,220p' tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py
rg -n "class AdminAlertRule|class AdminAlertState|class AdminAlertEvent|class AdminAlertHistoryListResponse|class AdminAlertAssignRequest|class AdminAlertSnoozeRequest|class AdminAlertEscalateRequest" tldw_Server_API/app/api/v1/schemas/admin_schemas.py
sed -n '994,1117p' tldw_Server_API/app/api/v1/schemas/admin_schemas.py
```

Expected: the exact request and response contract for topic alerts plus admin overlay state.

- [ ] **Step 5: Read only the auth dependency edges needed for monitoring claims**

Run:
```bash
rg -n "def get_auth_principal|def require_permissions|SYSTEM_LOGS" \
  tldw_Server_API/app/api/v1/API_Deps/auth_deps.py \
  tldw_Server_API/app/core/AuthNZ/permissions.py \
  tldw_Server_API/app/api/v1/endpoints/monitoring.py
sed -n '1,220p' tldw_Server_API/tests/Admin/test_admin_monitoring_api.py
sed -n '1,220p' tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py
```

Expected: enough evidence to judge whether the monitoring routes are permission-gated consistently, without expanding into a general AuthNZ audit.

### Task 3: Execute the Alert Identity and Overlay State Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/services/admin_monitoring_alerts_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/monitoring.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_monitoring_alerts_service.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py`
- Test: `tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py`

- [ ] **Step 1: Map the alert identity and overlay write/read helpers**

Run:
```bash
rg -n "def build_alert_identity|def merge_runtime_alert_with_overlay|def list_alert_states|def upsert_alert_state|def append_alert_event|def list_alert_events" \
  tldw_Server_API/app/services/admin_monitoring_alerts_service.py \
  tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py
```

Expected: exact anchors for how runtime topic alerts become admin-facing stateful entities.

- [ ] **Step 2: Read the helper and repository implementations in semantic slices**

Run:
```bash
sed -n '1,220p' tldw_Server_API/app/services/admin_monitoring_alerts_service.py
sed -n '1,420p' tldw_Server_API/app/core/AuthNZ/repos/admin_monitoring_repo.py
```

Capture:
- whether alert identity is stable, synthetic, or partially reconstructed
- how SQLite and PostgreSQL code paths differ
- how partial overlay updates are merged and persisted

Expected: a candidate finding list for source-of-truth, identity, and backend-parity risks.

- [ ] **Step 3: Trace one alert lifecycle through both route layers**

Run:
```bash
sed -n '261,430p' tldw_Server_API/app/api/v1/endpoints/monitoring.py
sed -n '225,420p' tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py
```

Trace:
- list alert
- acknowledge or dismiss persisted topic alert
- assign, snooze, and escalate overlay state
- list history

Expected: one end-to-end mental model of where alert truth lives and how it mutates.

- [ ] **Step 4: Read the overlay and repo tests before running any of them**

Run:
```bash
sed -n '1,220p' tldw_Server_API/tests/Admin/test_admin_monitoring_alerts_service.py
sed -n '1,260p' tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py
sed -n '1,260p' tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py
sed -n '1,220p' tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py
sed -n '1,220p' tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py
```

Expected: a clear map of which overlay invariants are truly tested and which are only implied.

### Task 4: Execute the Core Topic Monitoring and Notification Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/Monitoring/__init__.py`
- Inspect: `tldw_Server_API/app/core/Monitoring/README.md`
- Inspect: `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`
- Inspect: `tldw_Server_API/app/core/Monitoring/notification_service.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
- Test: `tldw_Server_API/tests/Monitoring/test_topic_monitoring.py`
- Test: `tldw_Server_API/tests/Monitoring/test_notification_service.py`
- Test: `tldw_Server_API/tests/Monitoring/test_notification_endpoint.py`
- Test: `tldw_Server_API/tests/Monitoring/test_monitoring_notifications_settings.py`
- Test: `tldw_Server_API/tests/Monitoring/test_monitoring_root_resolution.py`

- [ ] **Step 1: Read the local Monitoring README before implementation details**

Run:
```bash
sed -n '1,120p' tldw_Server_API/app/core/Monitoring/__init__.py
sed -n '1,240p' tldw_Server_API/app/core/Monitoring/README.md
```

Expected: the module export surface and intended Monitoring contract are both visible before source-level reasoning begins.

- [ ] **Step 2: Map the core topic monitoring hotspots**

Run:
```bash
rg -n "def _resolve_paths|def _seed_watchlists_from_file|def _load_watchlists_from_db|def _compile_rule|def list_watchlists|def upsert_watchlist|def delete_watchlist|def reload|def _snippet_around|def _iter_scan_chunks|def _find_match_span|def _applicable_watchlists|def evaluate_and_alert" tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py
```

Expected: the line anchors for path handling, watchlist loading, rule compilation, chunk scanning, scope filtering, and alert generation.

- [ ] **Step 3: Read the topic monitoring service in semantic slices**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py
sed -n '260,620p' tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py
sed -n '620,860p' tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py
```

Capture:
- config and path resolution rules
- config-file seeding versus DB truth
- regex safety and rule compilation behavior
- dedupe and stream-specific suppression behavior
- singleton lifecycle and reload semantics

Expected: a candidate finding list for operational behavior, concurrency, and alert-generation correctness.

- [ ] **Step 4: Read the notification and DB implementations in focused slices**

Run:
```bash
rg -n "class NotificationService|def _resolve_file_path|def get_settings|def update_settings|def notify\\(|def notify_generic|def notify_or_batch|def flush_digest|def _send_webhook|def _send_email" tldw_Server_API/app/core/Monitoring/notification_service.py
sed -n '1,360p' tldw_Server_API/app/core/Monitoring/notification_service.py
rg -n "class TopicMonitoringDB|def _ensure_schema|def list_watchlists|def get_watchlist_by_key|def upsert_watchlist|def replace_watchlist_rules|def insert_alert|def recent_duplicate_exists|def list_alerts|def mark_read" tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py
sed -n '1,760p' tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py
```

Expected: evidence about persistence semantics, path anchoring, write ordering, duplicate suppression, and runtime-notification behavior.

- [ ] **Step 5: Read the dedicated topic-monitoring and notification tests**

Run:
```bash
sed -n '1,320p' tldw_Server_API/tests/Monitoring/test_topic_monitoring.py
sed -n '1,260p' tldw_Server_API/tests/Monitoring/test_notification_service.py
sed -n '1,260p' tldw_Server_API/tests/Monitoring/test_notification_endpoint.py
sed -n '1,220p' tldw_Server_API/tests/Monitoring/test_monitoring_notifications_settings.py
sed -n '1,220p' tldw_Server_API/tests/Monitoring/test_monitoring_root_resolution.py
```

Expected: a map of which runtime and config invariants are already protected and which are not.

### Task 5: Execute Targeted Verification for the Highest-Value Claims

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/tests/Monitoring`
- Inspect: `tldw_Server_API/tests/Admin`
- Inspect: `tldw_Server_API/tests/AuthNZ`
- Test: `tldw_Server_API/tests/Monitoring/test_topic_monitoring.py`
- Test: `tldw_Server_API/tests/Monitoring/test_notification_service.py`
- Test: `tldw_Server_API/tests/Monitoring/test_notification_endpoint.py`
- Test: `tldw_Server_API/tests/Monitoring/test_monitoring_notifications_settings.py`
- Test: `tldw_Server_API/tests/Monitoring/test_monitoring_root_resolution.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_monitoring_alerts_service.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_monitoring_api.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py`
- Test: `tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py`

- [ ] **Step 1: Activate the project virtual environment before running pytest**

Run:
```bash
source .venv/bin/activate
python --version
```

Expected: the project virtual environment is active and Python reports successfully.

- [ ] **Step 2: Run the core monitoring unit and endpoint slices**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py \
  tldw_Server_API/tests/Monitoring/test_notification_service.py \
  tldw_Server_API/tests/Monitoring/test_notification_endpoint.py \
  tldw_Server_API/tests/Monitoring/test_monitoring_notifications_settings.py \
  tldw_Server_API/tests/Monitoring/test_monitoring_root_resolution.py \
  -q | tee /tmp/monitoring_backend_core_pytest.log
```

Expected: the dedicated topic-monitoring and notification slices pass, or any failures directly refine confidence in the candidate findings.

- [ ] **Step 3: Run the admin overlay and API slices**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Admin/test_admin_monitoring_alerts_service.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_api.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py \
  tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py \
  -q | tee /tmp/monitoring_backend_admin_pytest.log
```

Expected: admin monitoring overlay and route tests pass, or any failures expose backend control-plane drift.

- [ ] **Step 4: Run the monitoring-specific auth and backend-selection slices**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py \
  -q | tee /tmp/monitoring_backend_auth_pytest.log
```

Expected: permission-claim and backend-selection behavior is exercised locally without requiring PostgreSQL.

- [ ] **Step 5: Probe PostgreSQL parity only if the fixture environment is available**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_admin_monitoring_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_monitoring_repo_postgres.py \
  -q | tee /tmp/monitoring_backend_postgres_pytest.log
```

Expected: either PASS, or an explicit skip or fixture-unavailable signal that must be called out in the final verification section before making backend-parity claims.

### Task 6: Synthesize the Final Review and Explicitly Call Out Blind Spots

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-monitoring-backend-review-design.md`
- Inspect: `/tmp/monitoring_backend_core_pytest.log`
- Inspect: `/tmp/monitoring_backend_admin_pytest.log`
- Inspect: `/tmp/monitoring_backend_auth_pytest.log`
- Inspect: `/tmp/monitoring_backend_postgres_pytest.log`
- Test: none

- [ ] **Step 1: Sort candidate issues into confirmed findings, probable risks, improvements, and open questions**

Use this rubric:
- `Confirmed finding`: supported by source inspection plus either strong direct evidence or passing/failing targeted verification
- `Probable risk`: source evidence is real but execution evidence is incomplete or the environment blocked confirmation
- `Improvement`: worthwhile cleanup that is not presently a defect
- `Open question`: unresolved ambiguity that materially affects confidence

- [ ] **Step 2: Downgrade claims that lack executed support**

Before writing the final report, verify that:
- persistence or backend-parity claims mention whether PostgreSQL verification ran
- permission claims mention whether they came from source inspection, tests, or both
- any claim affected by local uncommitted monitoring changes is labeled as such

- [ ] **Step 3: Write the final report using the frozen response contract**

Use:
```markdown
## Findings
- severity-ordered items with confidence, type, impact, reasoning, and file references

## Structural Improvements
- targeted design or maintainability changes that materially reduce future defects

## Open Questions / Assumptions
- only unresolved items that materially affect confidence

## Test / Docs Gaps
- missing invariants, misleading tests, and meaningful documentation drift

## Prioritized Next Steps
- ordered follow-up actions that reduce the highest-risk uncertainty or defect exposure first

## Verification
- files inspected, tests run, and what remains unverified
```

Expected: one concise, evidence-backed backend review that separates immediate defects from longer-horizon cleanup.

- [ ] **Step 4: Verify the review stayed inside the approved scope**

Before final delivery, confirm:
- no Guardian/self-monitoring findings are mixed into the report
- no claims monitoring findings are mixed into the report
- UI observations appear only if they directly expose a backend contract issue

Expected: the final output remains a single coherent monitoring backend review rather than a catch-all monitoring audit.
