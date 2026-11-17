# C11 current managed policy at external execution admission

Associated task TASK-13260.278.18.83.45. Root recorded/read back the actual policy revocation failure before edits under human all-43 authorization and approved this exact manager admission design. Parent task remains root-owned.

## Source ruling and minimal design

The actual saved deny-all policy still allowed one artifact.read. McpHubService.update_external_server commits/audits managed storage; the external manager executes from its initial/reconciled cached policy. Reconcile intentionally preserves last-good inventory/adapter on replacement or partial-loader failure. Preserve that inventory availability, but never authorize an execution with stale policy or changed adapter-bound settings.

Inside the existing manager execution lock, load the canonical current snapshot immediately before external admission. Missing/disabled/unavailable/malformed current configuration refuses. Use current allow/deny/write/confirmation policy, preserving its defaults. Compare adapter-bound configuration against the initialized/reconciled config, excluding policy, enabled and display name; drift refuses until successful explicit reconciliation. Keep standalone file-loader support, exact managed loader filtering, owner/principal and credential checks, breaker/timeout/concurrency, last-good inventory and already-admitted call completion. No service/runtime coupling, auto-enable, new schema/dependency/config/default or reviewed refresh-endpoint change.

ADR assessment: no new ADR required. Reuse [ADR-018](../ADR/018-resource-governance-endpoint-policy-and-route-map.md), [ADR-052](../ADR/052-mcp-admin-claims.md) and [existing federation design](../Design/MCP_External_Server_Federation_Design.md): local policy applies before forwarding through existing canonical authority. This repairs freshness at that seam, not an alternate source of authority. [ADR-006](../ADR/006-bandit-report-path-portability.md) governs scoped Python security checks.

Scope: production external_servers/manager.py; new tests/MCP_unified/test_mcp_external_policy_revocation.py; genuinely affected existing manager in-flight/reconcile fixture if its consumptive loader wrongly advances saved configuration just because execution now rereads it. All old substantive assertions remain. Prior targeted-discovery endpoint and UI unit frozen; do not replay unchanged19/12 controls.

## Stage 1: Causal current-authority regression
**Goal**: Reproduce persistence followed by stale real manager execution.
**Success Criteria**: Real managed SQLite repository/service/router/module/registry lifecycles reproduce revoked policy/disable/removal/invalid or unavailable authority and transport drift; unchanged valid controls stay admitted.
**Tests**: New module before production. External adapter/storage failure seams only; no application import/auth shim/raw SQL.
**Status**: Complete

## Stage 2: Admission correction and qualification
**Goal**: Apply current authority under the existing execution lock without transport restart for policy-only changes.
**Success Criteria**: New cases and genuinely affected current-manager controls pass; compile/Ruff/Bandit baseline/current show no new findings.
**Tests**: Configured Python3.12.13 env, outer600/per-case300, normal plugins. Existing primary Bandit1.9.3/Python3.11 AST scanner only, baseline reports retained in memory.
**Status**: Complete

## Stage 3: Review and live recovery
**Goal**: Freeze exact source/tests for independent review and root controlled reload/recovery.
**Success Criteria**: Independent review clear; root actual persisted policy revocation/disable behavior succeeds after qualified reload.
**Tests**: Root alone owns runtime/browser/profile/model/request actions.
**Status**: In Progress

## Attempt ledger and limits

Maximum3 issue attempts, fixture/operator failures distinct from causal red; no cap reset/unchanged passing replay. In-flight calls admitted before a persisted update may finish; later calls must reread current authority after acquiring the lock. Readiness/inventory may retain last-good entries, but actual execution refuses when current configuration cannot authorize it. No cancellation of already-admitted effects or multi-process atomic commit/dispatched-effect guarantee is introduced. No parent-task, runtime/profile/API/build/Git/proof/dependency operation.

### Before-production attempt reassessment

1. Original production/full new15: natural1, 4.637s outer /3.04s pytest, 12 expected stale-authority failures and two positive passes. One additional real-stdio positive failed, excluded from causal evidence; default captured migration output obscured its exact traceback.
2. Exact failed real-stdio positive only with short/no-capture trace: natural0, 1pass, 2.068s outer /0.96s pytest on unchanged source/test. No reproducible fixture cause, so no speculative test edit. A preceding wrong test selector collected0/exit4 (operator failure, no behavior exercised). Initial real-stdio failure remains disclosed, not claimed causally resolved.

Retained12 causal reds support approved minimal manager correction. Next declared third issue phase is current-source new15 plus genuinely affected execution/reconcile/last-good controls; no prior targeted-discovery19/UI12 replay. Existing in-flight fixture's consumptive loader would advance saved transport merely on the added read. Before green, change only that loader to return current state and explicitly advance to new config after the old call enters; all existing assertions remain. Bare new-test B101 comments receive narrow pytest-oracle explanations only. This does not reset the cap.

### Final current-source qualification and freeze

Third declared current-source phase: all25 selected cases passed, natural0, 4.272s outer /2.94s pytest, seed1446782837, normal configured plugins, Python3.12.13, outer600/per-case300, no skips. New15 complete bodies plus these ten genuinely affected manager controls: legacy adapter runtime-auth compatibility; filtered discovery/unknown virtual tool; writes disabled; confirmation and marker stripping; in-flight reconciliation; replacement construction isolation; replacement discovery failure/last-good preservation; targeted construction count; partial-loader last-good preservation; telemetry. No unchanged targeted-discovery19 or UI12 replay. The real installed stdio control passes both its sole original-source diagnostic and the current-source full15 run; the first full-red stdio failure remains unexplained, without a speculative source/fixture claim.

Production one manager admission block: canonical loader read and fingerprint comparison inside existing lock, fresh policy for existing guards, safe PermissionError on unavailable/current-invalid/current-disabled/current-removed authority or adapter-bound drift. No cached runtime mutation, source filtering change, service/endpoint/UI/config/default/auth change. Existing in-flight test only returns current loader state and explicitly advances persisted state after first admission; every original assertion and outside-fixture AST remains unchanged.

Scoped final Ruff: production0/new test0; same one inherited I001 in existing manager test as baseline. New test import-format I001 corrected by formatting only after green, no behavioral change/replay. An initial non-escalated Ruff cache write failed with sandbox EPERM (operator/static invocation failure); final no-cache scan ran normally. Scoped AST parse+compile all three Python paths passed. Bandit existing primary Python3.11.13/Bandit1.9.3 STATIC only: baseline/current exact normalized-code fingerprints identical, five inherited B101 in existing manager tests, zero scan errors and no production/new-test finding. New assertions have narrow explained pytest-oracle nosec B101 comments. No dependency/environment reconstruction or proof file.

Frozen source/test hashes:
- manager.py: 5e15b19bc7d778b1de971d7b99ccde1a09d9310babb831817465f2a2e0865260
- existing test_external_server_manager.py: 6f8e3663020946d6c79ad0a107b0249ad1a476b4f99e5db90cb73f7d325ca85b
- new test_mcp_external_policy_revocation.py: 083e6360aa89ab7d424df945c603091daf35f460ee8af4110c79aa4899cc9088

Self-review read every production line/new body/canonical loader, saved service/router, module/registry dispatch and reconciliation/broker consumers. No substantiated new Critical/Important/Minor finding. Root independent source review requested; actual private-runtime reload and live deny-all/disable acceptance remain open. Coverage is real temporary SQLite managed storage/module/registry and fixed-principal actual update route, plus installed stdlib stdio positive; no full HTTP authentication/CSRF/multi-user owner/RLS/PostgreSQL/live transport/filesystem-write acceptance claim. Last-good inventory can remain visible; already-admitted calls can finish. New requests reload authority after the lock. No multi-process commit-to-dispatch atomicity or cancellation of prior admitted effects is claimed.
