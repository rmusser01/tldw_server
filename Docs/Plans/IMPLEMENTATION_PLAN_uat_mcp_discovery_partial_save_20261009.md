# C11 managed MCP discovery and acknowledged-save repair

Associated task: TASK-13260.278.18.83.45. Root recorded the supported before-edit note and approved this unit before this plan/tests/source work. Standing human all-43 fix/validate authorization applies.

## Evidence and scope

Actual one Simple stdio create succeeded; subsequent per-server refresh returned 503, and the UI called that a create failure while retaining its form. Root independently verified the owned live runtime had zero loaded modules because its private working directory lacked the relative default module YAML and both module override variables were unset. Root owns explicit private federation opt-in and later runtime qualification. This unit never auto-enables federation.

Per-server refresh currently executes external.tools.refresh against the manager adapter snapshot. The existing aggregate path reconciles current managed storage, invalidates capability caches and refreshes module registry mappings. A newly committed server requires that same targeted reconciliation. Preserve the current module breaker, timeout and semaphore, principal/mutation/visibility/row admission and same-server lock.

Production scope is only endpoints/mcp_hub_management.py and shared UI MCPHub/ExternalServersTab.tsx. New backend/UI test modules qualify the actual manager/module/registry/router and mounted component/service at external transport seams. One existing Simple test whose former expectation codifies the partial-save bug is genuinely affected. No backend transport, auth, owner, parser, dependency, policy/default, profile/runtime, Git, parent task or unrelated source change.

ADR check: ADR required: no. Reuse [ADR-018](../ADR/018-resource-governance-endpoint-policy-and-route-map.md), [ADR-052](../ADR/052-mcp-admin-claims.md), and [existing federation design](../Design/MCP_External_Server_Federation_Design.md). Existing claim-derived authority, module opt-in, managed registry, public wire formats and execution limits remain; this corrects their existing consumer path and acknowledged-save UI.

## Stage 1: Causal contracts
**Goal**: Establish causal red for a saved-after-startup server and acknowledged save followed by discovery refusal.
**Success Criteria**: Canonical real module/manager/registry/router with owned temporary DB reproduces absent new tools; real mounted UI/service shows incorrect create failure. Preserve unavailable module and authority/config/error controls.
**Tests**: New targeted backend module and mounted partial-save module using existing configured plugins; external adapter/request seams only.
**Status**: Complete

## Stage 2: Minimal correction and qualification
**Goal**: Targeted reconcile under existing execution limits, and truthful saved lifecycle with explicit same-ID discovery retry.
**Success Criteria**: New regressions pass; acknowledged save closes form/selects row even on discovery refusal; actual create failure retains draft; readiness error/no-tools not presented as Ready. Affected tests pass without broad earlier-control replay; scoped lint/compile/Bandit reports no new finding.
**Tests**: New causal cases, genuinely affected route/guard and partial-save cases, first configured UI/Web affected qualifications; baseline/current static checks.
**Status**: Complete

## Stage 3: Independent review and actual recovery
**Goal**: Freeze exact source/tests for independent connected review and root-controlled runtime recovery.
**Success Criteria**: Independent review clear; root combined types/build and explicit private federation activation followed by one actual existing-server discovery recovery/read-only fixture tool qualification.
**Tests**: Root-owned actual runtime/browser and configured types/build only; no agent runtime or provider invocation.
**Status**: In Progress

## Bounded attempt ledger

Normal Python3.12 activated environment; configured pytest plugins; outer cap600/per-case300. UI/Web Node20.19.5/Vitest4.0.18, 8192MB, normal configured plugins, outer600. Maximum three issue attempts before STOP/reassessment; fixture/setup failures never labelled causal red. No unchanged passing replay to obtain a prettier combined count. Evidence and exact byte/delta inventory retained in memory; no proof files.

### Phase 1 evidence and fixture assessment (before production)

Backend first command natural pytest1, outer6.374s: four new-runtime discovery/admission cases failed, six controls passed; one additional foreign-owner oracle was invalid and is excluded from causal evidence. A system.configure principal intentionally has unrestricted list visibility under existing _resolve_visible_scope_filters. Correct only this new control to an ordinary read principal validating a foreign row (404); mutation refusal remains separately 403. No authority source change. Backend primary expected tool_count1 saw the saved server remain absent from manager/registry; external adapter entry was not reached for new-server controls. Concurrency wait timeout reflects that absent adapter entry, not a fixture scheduling defect.

UI first command natural Vitest1, outer7.958s: five causal failures and three controls passed. Actual mounted DOM shows Failed to create after discovery refusal and retained create form; saved-result/retry absent; returned needs_attention/no_tools show false Ready. No provider/model/runtime invocation.

Operator/static preparation failures are separate: initial full-source JSON output exceeded the transport output budget, so pre-unit exact bytes were retained successfully compressed in memory; first Ruff lacked worktree cache permission; first ESLint used the wrong installed executable path. None is a causal test failure. Valid baseline Ruff production I001/UP035/BLE001 are inherited, and baseline static Bandit1.9.3/Python3.11.13 reports0; this scanner reads AST only, current app/tests remain Python3.12.13. Valid apps-base ESLint baseline has0errors/2 inherited exhaustive-deps warnings.

The existing default-executor initialization test fixture is genuinely affected: adapt only its manager/registry reconciliation fixture and result assertion while retaining initialization-before-lookup and exactly-one-initialize oracles. Other backend bodies unchanged. Existing Simple refusal expectation must now retain Saved identity instead of codifying the bug. Source changes remain exactly two production paths.


### Corrected-source qualification and frozen review handoff

Backend phase 2 natural0/4.889s outer (3.54s pytest): 18 passed =13 new plus five genuinely affected existing route cases. The positive case uses the repository's existing stdlib stdio fixture through real managed storage, module/manager/registry/router and verifies newly registered tool identity. The additional malformed owned-config control received its first and sole execution natural0/3.097s outer (1.39s pytest):1 passed. Retain 19 distinct backend passes =14 new plus5 existing; no combined19 execution or full app/auth/CSRF/PostgreSQL/provider/live acceptance claim. Configured inherited warnings remain (2667 main phase;238 sole case).

UI phase 2 natural0/16.938s outer:12 selected passed. First Web phase natural0/18.569s outer (18.22s Vitest):the same12 selected passed. Each includes8 new, the affected prior Simple refusal case, and3 existing managed create/edit/delete/manual/guided-save cases. Both commands pattern-filtered30 unrelated cases; do not describe this as full42 qualification or no skips. No earlier Simple34 replay. Real mounted component/service/Form/StatePanel and external HTTP seam verify acknowledged save, closed draft, selected persisted ID, same-ID explicit retry/no duplicate create, repeated discovery refusal, truthful needs_attention/no_tools/ready, save without discovery and genuine create refusal.

Final static qualification: Python compile/AST syntax passes3 touched Python paths; TypeScript transpile syntax passes3 touched TSX paths (not project types). Scoped Ruff new backend module0; existing production retains I001/UP035/BLE001 and existing test retains I001/twoF821(anext)/UP037 exactly baseline, no new findings. Apps-base ESLint current source0errors/2 inherited exhaustive-deps warnings; new and affected tests0errors/0warnings. Bandit1.9.3 using the confirmed existing primary Python3.11.13 AST-only scanner: production baseline/current0 and new test0. Existing test baseline B101276/B10534/B1061; current B101275/B10534/B1061, all inherited,0 scan errors. Exact equal-line mapping reports0 new findings.32 new-module and3 changed-body assertion-only nosec B101 comments explain pytest behavioral oracles; comment-only AST equality preserved. No assertion, runtime guard or security code is suppressed.

Static operator issues, excluded from defect evidence: JavaScript orchestrator has no btoa; no command or mutation ran on that failed call. An overlarge inherited Bandit report exceeded output budget; corrected bounded comparison retained counts. A first comparison keyed code excerpts containing line numbers and therefore falsely classified shifted inherited lines; exact byte-equal-line mapping corrected that comparator and reports0 new findings. No behavioral test replay followed these static-output corrections.

Self-review read both production deltas, full new14 backend and8 mounted UI bodies, the changed existing executor fixture and changed Simple refusal case. No actionable finding in owned scope. Existing backend bytes outside the single owned test body are identical to retained pre-unit bytes; first13 original Simple bodies are identical. Source/test freeze is ready for independent connected review. Root combined UI/Web types/build, private explicit federation opt-in/reload, actual persisted-server recovery and one read-only fixture tool call remain open. No source auto-enable, application process, provider/model or private-profile operation performed by this agent.
