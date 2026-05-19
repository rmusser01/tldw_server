# Audit Module Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Audit module review and deliver one consolidated, evidence-backed review covering reliability risks, operational failure modes, tenant-boundary issues, test gaps, and targeted improvement suggestions across the Audit subsystem and its main integrations.

**Architecture:** This is a read-first, risk-first review plan. Execution starts by locking the worktree baseline, then inspects the Audit core and migration path, then the DI/config/export surfaces, then representative integrations, and only after that runs the dedicated Audit suite plus selected integration tests to validate or weaken candidate findings. No source changes are expected during execution; the deliverable is the final review output in-session.

**Tech Stack:** Python 3, pytest, git, find, grep, sed, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not just `HEAD`
- label findings that depend on uncommitted local changes
- prioritize reliability and operational risk over style or broad refactoring advice
- separate `Confirmed issue`, `Likely risk`, and `Improvement suggestion`
- do not modify repository source files during the review itself
- do not run unrelated blanket test suites; use the representative test-selection rule from the spec
- do not propose fixes yet unless a finding cannot be explained without a remediation sketch

## Review File Map

**No repository files should be modified during execution.**

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-07-audit-module-review-design.md`
- `Docs/superpowers/plans/2026-04-07-audit-module-review-execution-plan.md`

**Primary implementation files to inspect:**
- `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- `tldw_Server_API/app/core/Audit/audit_shared_migration.py`
- `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- `tldw_Server_API/app/api/v1/endpoints/audit.py`
- `tldw_Server_API/app/core/config.py`
- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- `Docs/Code_Documentation/Guides/Audit_Module_Code_Guide.md`
- `tldw_Server_API/app/core/Audit/README.md`

**Representative integration files to inspect:**
- `tldw_Server_API/app/core/Embeddings/audit_adapter.py`
- `tldw_Server_API/app/core/Evaluations/audit_adapter.py`
- `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- `tldw_Server_API/app/core/Evaluations/webhook_manager.py`
- `tldw_Server_API/app/core/Chat/chat_service.py`
- `tldw_Server_API/app/core/Sharing/share_audit_service.py`
- `tldw_Server_API/app/core/Sharing/workspace_deletion_hook.py`
- `tldw_Server_API/app/core/MCP_unified/protocol.py`
- `tldw_Server_API/app/core/RAG/rag_service/security_filters.py`
- `tldw_Server_API/app/core/Resource_Governance/coverage_audit.py`
- `tldw_Server_API/app/core/Workflows/adapters/control/flow.py`
- `tldw_Server_API/app/core/Workflows/adapters/llm/moderation.py`

**High-value tests to inspect and selectively run:**
- `tldw_Server_API/tests/Audit/test_unified_audit_service.py`
- `tldw_Server_API/tests/Audit/test_audit_shared_migration.py`
- `tldw_Server_API/tests/Audit/test_audit_db_deps.py`
- `tldw_Server_API/tests/Audit/test_audit_endpoints.py`
- `tldw_Server_API/tests/Audit/test_audit_export_endpoint.py`
- `tldw_Server_API/tests/Audit/test_audit_export_pagination.py`
- `tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py`
- `tldw_Server_API/tests/Audit/test_audit_fallback_lock.py`
- `tldw_Server_API/tests/Audit/test_audit_service_init_race.py`
- `tldw_Server_API/tests/Audit/test_audit_storage_mode.py`
- `tldw_Server_API/tests/Audit/test_audit_pii_overrides.py`
- `tldw_Server_API/tests/Audit/test_pii_pattern_groups.py`
- `tldw_Server_API/tests/Audit/test_risk_settings_overrides.py`
- `tldw_Server_API/tests/Audit/test_sqlite_runtime_pragmas.py`
- `tldw_Server_API/tests/AuthNZ/test_audit_chain_integration.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_api_key_rotation_audit.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_admin_membership_audit_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_Postgres/test_admin_membership_audit_org_pg.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_service_audit_tasks.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_chat_command_audit.py`
- `tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py`
- `tldw_Server_API/tests/Embeddings/test_dlq_audit_redact_encryption.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_audit_adapter.py`
- `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_audit_hooks.py`
- `tldw_Server_API/tests/Jobs/test_jobs_audit_bridge.py`
- `tldw_Server_API/tests/Sharing/test_share_audit_service.py`
- `tldw_Server_API/tests/MCP_unified/test_mcp_hub_audit_findings.py`
- `tldw_Server_API/tests/Resource_Governance/test_coverage_audit.py`
- `tldw_Server_API/tests/Admin/test_admin_account_audit_events.py`
- `tldw_Server_API/tests/UserProfile/test_user_profile_admin_audit.py`
- `tldw_Server_API/tests/UserProfile/test_user_profile_user_audit.py`

**Scratch artifacts allowed during execution:**
- `/tmp/audit_module_review_notes.md`
- `/tmp/audit_review_core_pytest.log`
- `/tmp/audit_review_integration_pytest.log`

## Stage Overview

## Stage 1: Baseline and Checklist
**Goal:** Lock the worktree baseline, confirm the exact Audit review surface, and fix the final report structure before deep reading starts.
**Success Criteria:** The review scope, dirty-worktree note, file buckets, representative test list, and final report template are fixed.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: Core and Migration Reliability Pass
**Goal:** Inspect the Audit core and migration path for event-loss, fallback, shutdown, schema, retention, and resumability risks.
**Success Criteria:** Candidate findings are recorded with exact file references and evidence type, and weak concerns are explicitly separated from confirmed failures.
**Tests:** Read dedicated Audit tests relevant to core, migration, fallback, and shutdown behavior.
**Status:** Not Started

## Stage 3: Lifecycle, Config, and Export Pass
**Goal:** Inspect the DI, configuration, DB-path, and endpoint layers for tenant-boundary, storage-mode, lifecycle, and export-scaling risks.
**Success Criteria:** Shared/per-user storage rules, admin scoping, cross-loop shutdown behavior, and export filtering/streaming assumptions are traced end to end.
**Tests:** Read and run focused DI and endpoint tests after the static pass.
**Status:** Not Started

## Stage 4: Integration Pass
**Goal:** Trace how major backend modules emit audit events and where adapter/caller patterns can lose context, swallow failures, or create brittle lifecycle behavior.
**Success Criteria:** Each priority integration area has at least one representative caller/test pair reviewed when coverage exists, code-only areas are called out explicitly as test gaps, and the evidence is strong enough to support confirmed findings or explicit likely-risk labels.
**Tests:** Representative cross-module audit tests only.
**Status:** Not Started

## Stage 5: Test Execution and Evidence Reconciliation
**Goal:** Run the dedicated Audit suite and the representative integration tests needed to validate or weaken candidate findings.
**Success Criteria:** Test outcomes are captured, failures are interpreted cautiously, and every major claim in the final review is tied to code inspection, test execution, or both.
**Tests:** Dedicated Audit suite plus selected integration slices.
**Status:** Not Started

## Stage 6: Final Synthesis
**Goal:** Produce the final review with findings first, clear evidence labels, open questions, and non-bug improvements separated cleanly.
**Success Criteria:** The final output matches the approved spec, includes baseline/evidence notes, and does not overstate claims unsupported by code or tests.
**Tests:** No new tests unless a disputed claim still needs confirmation.
**Status:** Not Started

### Task 1: Lock the Review Baseline and Output Format

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-audit-module-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-audit-module-review-execution-plan.md`
- Inspect: `tldw_Server_API/app/core/Audit`
- Inspect: `tldw_Server_API/tests/Audit`
- Test: none

- [ ] **Step 1: Capture the dirty-worktree baseline**

Run:
```bash
git status --short
```

Expected: a list of any uncommitted files, including whether Audit-related files are currently modified.

- [ ] **Step 2: Record the commit baseline used for the review**

Run:
```bash
git rev-parse --short HEAD
```

Expected: one short commit hash to cite when a finding depends on committed behavior rather than only local edits.

- [ ] **Step 3: Enumerate the core Audit surface**

Run:
```bash
find tldw_Server_API/app/core/Audit -maxdepth 2 -type f | sort
find tldw_Server_API/app/api/v1/API_Deps -maxdepth 2 -type f | grep 'Audit_DB_Deps.py'
find tldw_Server_API/app/api/v1/endpoints -maxdepth 1 -type f | grep '/audit.py$'
```

Expected: the exact core Audit implementation and API entrypoints used in the review.

- [ ] **Step 4: Enumerate the dedicated Audit tests**

Run:
```bash
find tldw_Server_API/tests/Audit -maxdepth 1 -type f | sort
```

Expected: the dedicated Audit test inventory that will anchor the later verification pass.

- [ ] **Step 5: Fix the final response template before reading deeply**

Use this structure for the final review:
```markdown
## Findings
- severity-ordered confirmed issues first with file references and evidence notes

## Open Questions / Assumptions
- only unresolved items that materially affect confidence

## Improvements
- non-bug suggestions that reduce operational fragility

## Verification
- tests run, important files inspected, and what remains unverified
```

### Task 2: Execute the Core Service and Migration Reliability Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/Code_Documentation/Guides/Audit_Module_Code_Guide.md`
- Inspect: `tldw_Server_API/app/core/Audit/README.md`
- Inspect: `tldw_Server_API/app/core/Audit/unified_audit_service.py`
- Inspect: `tldw_Server_API/app/core/Audit/audit_shared_migration.py`
- Test: `tldw_Server_API/tests/Audit/test_unified_audit_service.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_shared_migration.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_eviction_shutdown.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_fallback_lock.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_service_init_race.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_pii_overrides.py`
- Test: `tldw_Server_API/tests/Audit/test_pii_pattern_groups.py`
- Test: `tldw_Server_API/tests/Audit/test_risk_settings_overrides.py`
- Test: `tldw_Server_API/tests/Audit/test_sqlite_runtime_pragmas.py`

- [ ] **Step 1: Read the operator-facing design docs first**

Run:
```bash
sed -n '1,260p' Docs/Code_Documentation/Guides/Audit_Module_Code_Guide.md
sed -n '1,220p' tldw_Server_API/app/core/Audit/README.md
```

Expected: the intended contracts for storage mode, fallback behavior, export, shutdown, and test mode.

- [ ] **Step 2: Locate the core service landmarks before reading full sections**

Run:
```bash
grep -n "class UnifiedAuditService\|async def initialize\|async def _init_database\|async def flush\|async def stop\|async def export_events\|async def query_events\|def _append_events_to_fallback\|async def _migrate_legacy_audit_events" tldw_Server_API/app/core/Audit/unified_audit_service.py
grep -n "def main\|async def migrate\|checkpoint\|resume\|skip\|locked\|corrupt" tldw_Server_API/app/core/Audit/audit_shared_migration.py
```

Expected: a stable reading map for the highest-risk code paths.

- [ ] **Step 3: Read the core service sections in reliability order**

Read and trace:
- initialization and schema creation
- buffer ownership and flush triggers
- fallback queue writing
- export/query paths
- shutdown and background-task behavior
- legacy migration or compatibility paths

Expected: a candidate finding list that distinguishes event-loss risks, shutdown races, export hazards, and migration hazards.

- [ ] **Step 4: Read the core-focused tests before running them**

Capture for each test file:
- which invariant it is trying to protect
- whether it covers the production failure mode directly or only indirectly
- what adjacent behavior still appears untested

- [ ] **Step 5: Search for suspicious reliability patterns in the core service**

Run:
```bash
grep -nE "except Exception|create_task|run_coroutine_threadsafe|sleep\(|warning\(|error\(|return None|pass$|TODO|FIXME"   tldw_Server_API/app/core/Audit/unified_audit_service.py   tldw_Server_API/app/core/Audit/audit_shared_migration.py
```

Expected: a short list of branches to inspect manually, not a final finding list by itself.

### Task 3: Execute the Lifecycle, Config, and Export Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Inspect: `tldw_Server_API/app/core/config.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/audit.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_db_deps.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_endpoints.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_export_endpoint.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_export_pagination.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_storage_mode.py`
- Test: `tldw_Server_API/tests/Audit/test_audit_result_semantics.py`

- [ ] **Step 1: Locate the lifecycle and storage-mode landmarks**

Run:
```bash
grep -n "get_audit_service_for_user\|shutdown_user_audit_service\|shutdown_all_audit_services\|_resolve_audit_storage_mode\|_schedule_service_stop\|EVICTION_" tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py
grep -n "get_audit_db_path\|get_shared_audit_db_path" tldw_Server_API/app/core/DB_Management/db_path_utils.py
grep -n "AUDIT_" tldw_Server_API/app/core/config.py
grep -n "export_audit_events\|count_audit_events\|_shared_storage_enabled\|_principal_is_admin\|_map_event_types\|_map_categories" tldw_Server_API/app/api/v1/endpoints/audit.py
```

Expected: the exact functions that decide tenant scoping, path selection, and export semantics.

- [ ] **Step 2: Read the lifecycle and endpoint sections in one pass**

Read and trace:
- per-user vs shared storage resolution
- rollback precedence
- owner-loop and cross-loop shutdown paths
- admin vs non-admin filtering behavior
- stream forcing and max-row behavior
- filename and timestamp parsing behavior

Expected: a candidate finding list for tenant leakage, stale service reuse, and export correctness or scaling risks.

- [ ] **Step 3: Compare static behavior to the dedicated DI and endpoint tests**

Capture:
- whether the tests assert the exact behavior the code claims
- whether shared-storage and non-admin branches are both covered
- whether the most expensive export paths are only lightly covered

- [ ] **Step 4: Search for broad exception handling or silent branch behavior**

Run:
```bash
grep -nE "except Exception|return False|return None|logger\.warning|logger\.debug|create_task|Thread\(|run_coroutine_threadsafe"   tldw_Server_API/app/api/v1/API_Deps/Audit_DB_Deps.py   tldw_Server_API/app/api/v1/endpoints/audit.py   tldw_Server_API/app/core/config.py
```

Expected: a shortlist of lifecycle and export branches that need manual scrutiny.

### Task 4: Execute the Integration Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/Embeddings/audit_adapter.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/audit_adapter.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/webhook_manager.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_service.py`
- Inspect: `tldw_Server_API/app/core/Sharing/share_audit_service.py`
- Inspect: `tldw_Server_API/app/core/Sharing/workspace_deletion_hook.py`
- Inspect: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/security_filters.py`
- Inspect: `tldw_Server_API/app/core/Resource_Governance/coverage_audit.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/control/flow.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/llm/moderation.py`
- Test: `tldw_Server_API/tests/AuthNZ/test_audit_chain_integration.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_api_key_rotation_audit.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_audit_tasks.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_command_audit.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py`
- Test: `tldw_Server_API/tests/Embeddings/test_dlq_audit_redact_encryption.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_audit_adapter.py`
- Test: `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_audit_hooks.py`
- Test: `tldw_Server_API/tests/Jobs/test_jobs_audit_bridge.py`
- Test: `tldw_Server_API/tests/Sharing/test_share_audit_service.py`
- Test: `tldw_Server_API/tests/MCP_unified/test_mcp_hub_audit_findings.py`
- Test: `tldw_Server_API/tests/Resource_Governance/test_coverage_audit.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_account_audit_events.py`
- Test: `tldw_Server_API/tests/UserProfile/test_user_profile_admin_audit.py`
- Test: `tldw_Server_API/tests/UserProfile/test_user_profile_user_audit.py`

- [ ] **Step 1: Enumerate direct unified-audit callers and adapters**

Run:
```bash
grep -RIn "unified_audit_service\|Audit_DB_Deps\|AuditEventType\|AuditContext"   tldw_Server_API/app/core   tldw_Server_API/app/api/v1 | sort | sed -n '1,240p'
```

Expected: the direct caller map for representative integrations without drifting into unrelated frontend or docs scope.

- [ ] **Step 2: Read one representative caller path per priority area**

For each area, capture:
- how the service instance is obtained
- whether audit writes are awaited or deferred
- whether `user_id`, endpoint, and method context are populated
- whether failures are re-raised, logged, or swallowed
- whether the integration assumes per-user or shared storage implicitly

Expected: a consistent matrix across AuthNZ, Chat, Embeddings, Evaluations, Jobs, Sharing, MCP, RAG, and Workflows.

- [ ] **Step 3: Compare representative tests against the caller behavior**

Capture:
- which integration expectations are explicitly tested
- where a test only checks that a call was made, not that the context or tenant scoping is correct
- which modules appear to rely on the audit service without dedicated regression coverage
- which priority areas have no direct audit-focused tests and therefore must be reported as coverage gaps

- [ ] **Step 4: Search for unsafe fire-and-forget and broad exception patterns in integration code**

Run:
```bash
grep -RInE "create_task|except Exception|logger\.error|logger\.warning|pass$|return None"   tldw_Server_API/app/core/Embeddings/audit_adapter.py   tldw_Server_API/app/core/Evaluations/audit_adapter.py   tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py   tldw_Server_API/app/core/Evaluations/webhook_manager.py   tldw_Server_API/app/core/Chat/chat_service.py   tldw_Server_API/app/core/Sharing/share_audit_service.py   tldw_Server_API/app/core/Sharing/workspace_deletion_hook.py   tldw_Server_API/app/core/MCP_unified/protocol.py   tldw_Server_API/app/core/RAG/rag_service/security_filters.py   tldw_Server_API/app/core/Resource_Governance/coverage_audit.py   tldw_Server_API/app/core/Workflows/adapters/control/flow.py   tldw_Server_API/app/core/Workflows/adapters/llm/moderation.py
```

Expected: a shortlist of integration branches that need manual scrutiny for dropped events or brittle error handling.

### Task 5: Run the Dedicated Audit Suite and Representative Integration Tests

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/tests/Audit`
- Inspect: representative integration test files listed above
- Test: `tldw_Server_API/tests/Audit`
- Test: representative integration test files listed above

- [ ] **Step 1: Run the dedicated Audit suite and capture the output**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audit -v | tee /tmp/audit_review_core_pytest.log
```

Expected: a full Audit-suite result log with either all-pass output or named failures that can be correlated to static findings.

- [ ] **Step 2: Run the representative AuthNZ and Chat audit tests**

Run:
```bash
source .venv/bin/activate
python -m pytest   tldw_Server_API/tests/AuthNZ/test_audit_chain_integration.py   tldw_Server_API/tests/AuthNZ/integration/test_api_key_rotation_audit.py   tldw_Server_API/tests/Chat/unit/test_chat_service_audit_tasks.py   tldw_Server_API/tests/Chat_NEW/integration/test_chat_command_audit.py -v | tee /tmp/audit_review_integration_pytest.log
```

Expected: a focused result slice covering authentication and chat audit integrations.

- [ ] **Step 3: Run the representative Embeddings, Evaluations, Sharing, Jobs, MCP, Admin, and UserProfile audit tests**

Run:
```bash
source .venv/bin/activate
python -m pytest   tldw_Server_API/tests/Embeddings/test_embeddings_audit_adapter.py   tldw_Server_API/tests/Embeddings/test_dlq_audit_redact_encryption.py   tldw_Server_API/tests/Evaluations/test_evaluations_audit_adapter.py   tldw_Server_API/tests/Evaluations/test_embeddings_abtest_audit_hooks.py   tldw_Server_API/tests/Jobs/test_jobs_audit_bridge.py   tldw_Server_API/tests/Sharing/test_share_audit_service.py   tldw_Server_API/tests/MCP_unified/test_mcp_hub_audit_findings.py   tldw_Server_API/tests/Admin/test_admin_account_audit_events.py   tldw_Server_API/tests/UserProfile/test_user_profile_admin_audit.py   tldw_Server_API/tests/UserProfile/test_user_profile_user_audit.py -v
```

Expected: one representative integration pass across the remaining priority modules.

- [ ] **Step 4: Run storage-backend-specific audit tests only if the earlier evidence points there**

Run if needed:
```bash
source .venv/bin/activate
python -m pytest   tldw_Server_API/tests/AuthNZ_SQLite/test_admin_membership_audit_sqlite.py   tldw_Server_API/tests/AuthNZ_SQLite/test_admin_membership_audit_team_sqlite.py   tldw_Server_API/tests/AuthNZ_Postgres/test_admin_membership_audit_org_pg.py   tldw_Server_API/tests/AuthNZ_Postgres/test_admin_membership_audit_team_pg.py -v
```

Expected: additional backend-specific evidence only when tenant or storage behavior remains disputed.

- [ ] **Step 5: Reconcile the test results against the candidate findings**

For each major candidate finding, record:
- confirmed by static inspection only
- confirmed by static inspection plus test behavior
- weakened by passing coverage
- still ambiguous because the relevant path is untested or environment-gated

### Task 6: Produce the Final Findings Report

**Files:**
- Create: none
- Modify: none
- Inspect: all files and test outputs referenced above
- Test: none beyond disputed-claim reruns

- [ ] **Step 1: Convert the candidate list into the approved finding classes**

For each item, classify it as:
- `Confirmed issue`
- `Likely risk`
- `Improvement suggestion`

Expected: no unsupported suspicion remains labeled as a confirmed issue.

- [ ] **Step 2: Order confirmed findings by operational severity**

Use this order when applicable:
- cross-tenant leakage
- silent event loss or fallback corruption
- shutdown or lifecycle races that can drop events
- export/filtering behavior that can expose or omit data incorrectly
- integration inconsistencies that weaken audit integrity
- lower-risk maintainability improvements

Expected: the final report starts with the most production-relevant failures, not the easiest fixes.

- [ ] **Step 3: Add evidence and baseline notes to every major finding**

Each major finding must say whether it is based on:
- current worktree behavior
- committed behavior
- static inspection
- test execution
- or a combination

Expected: readers can tell exactly how strong each claim is and whether local uncommitted changes matter.

- [ ] **Step 4: Add only material open questions**

Open questions are allowed only when they materially affect confidence or severity.

Expected: the final review does not pad the report with minor curiosities or speculative tangents.

- [ ] **Step 5: Deliver the final in-session review**

The final response must:
- present findings first with file references
- separate improvements from bug findings
- include a short verification section listing tests run and anything still unverified
- avoid proposing code edits unless the user asks for fixes next

Expected: one concise, evidence-backed Audit review that can immediately drive remediation work if needed.
