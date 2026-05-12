---
id: TASK-287
title: Implement ACP downstream entrypoint strategy stages 1-3
status: In Progress
assignee: []
created_date: '2026-05-12 03:51'
updated_date: '2026-05-12 06:00'
labels:
  - ACP
  - implementation
  - certification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1563'
  - 'https://github.com/rmusser01/tldw_server/issues/1564'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md
  - Docs/Development/ACP_Compatibility_Matrix.md
  - Docs/Development/ACP_Certification_Checklist.md
  - tldw_Server_API/Config_Files/agents.yaml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved ACP downstream entrypoint strategy design for the first product slice. This work adds explicit ACP entrypoint strategy metadata, classification, profile-specific certification manifests, and setup/status/API visibility while keeping live certification, downstream agent installation, and adapter implementation out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Registry entries, YAML rows, API registration/update schemas, and DB-backed dynamic registrations preserve entrypoint strategy metadata with conservative defaults for legacy rows.
- [ ] #2 A deterministic classifier reports probe state, ACP command/args, primary blocker, blockers, status message, and docs URL without running live agent commands.
- [x] #3 Certification smoke helper can render profile-specific dry-run manifests for native, adapter-backed, documented-candidate, and custom-template profiles and refuses unsafe live runs without required env.
- [ ] #4 ACP agents, health, and setup-guide surfaces expose strategy and blocker metadata consistently for YAML, API-backed, runner, and static fallback rows.
- [ ] #5 Focused unit, helper, integration, docs, and security checks pass for the touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan saved at Docs/superpowers/plans/2026-05-12-acp-entrypoint-strategy-implementation-plan.md.

Plan review completed after two fix rounds. Added explicit no-inference guardrail tests, legacy DB migration/default coverage, dynamic API parity tests, and initialize-gated session/new plus session/prompt manifest sequencing. Final reviewer status: approved.

Task 1 complete. Added registry/API/DB ACP entrypoint strategy metadata, built-in YAML seeds, migration/default handling, and focused tests. Reviews: spec compliant and code quality approved after null-clearing and mutable-default fixes. Final scoped Task 1 tests: 67 passed, 5 warnings. Bandit reports only existing ACP_Sessions_DB.py baseline findings outside changed lines.

Task 2 complete. Added deterministic classify_agent_entrypoint output, immutable AgentEntrypointClassification value object, blocker precedence tests, no acp_command inference tests, and caveat taxonomy rows. Reviews: spec compliant and code quality approved after tuple immutability fix. Focused classifier tests: 19 passed, 5 warnings; Bandit on agent_registry.py reported 0 findings.

Task 3 execution plan recorded before edits: add requested failing helper tests; run focused pytest with shared venv for red evidence; implement only certification smoke profile manifest and stdio runner changes; re-run focused pytest plus scoped Bandit and diff review; commit only Task 3 helper and test files.

Task 3 red evidence: focused helper pytest initially failed with 4 missing-helper failures for build_agent_profile_manifest, run_manifest_dict, and _run_stdio_jsonrpc_sequence while 8 existing tests passed. Task 3 green evidence: focused helper pytest now reports 12 passed, 5 warnings. Scoped Bandit on acp_certification_smoke.py reports 0 results. git diff --check on Task 3 files is clean.

Task 3 follow-up review fix plan: add regression coverage for interleaved JSON-RPC notifications before matching responses; verify red against current stdio runner; update runner to wait for the response id matching the just-sent frame while ignoring notifications and other ids within the existing timeout; re-run focused helper tests and commit a scoped follow-up.

Task 3 follow-up red evidence: helper pytest failed with len(written)==2 after an interleaved notification before initialize error, proving session/new was written too early. Follow-up green evidence: helper pytest now reports 14 passed, 5 warnings. The stdio reader now waits for a matching JSON-RPC response id and ignores notifications and other ids inside the existing timeout. Scoped Bandit on acp_certification_smoke.py reports 0 results and git diff --check is clean.

Task 3 code-quality follow-up plan: add red tests for partial-line timeout cleanup, JSON-RPC error cleanup with sanitized stderr, and broken-pipe cleanup; replace select/readline main-thread response reading with a reader-thread queue bounded by the existing deadline; add one cleanup helper that closes stdio, terminates or kills, waits, and is used on all timeout/error/broken-pipe paths; re-run focused helper tests, Bandit, and diff checks before committing.

Task 3 code-quality follow-up red evidence: helper pytest failed 3 tests for partial-line timeout and missing cleanup on JSON-RPC error and broken pipe. Final green evidence: helper pytest reports 17 passed, 5 warnings in 2.87s. Bandit on acp_certification_smoke.py reports 0 findings. git diff --check is clean. Implementation now uses a daemon stdout reader thread plus queue/deadline matching, central cleanup for kill or terminate plus wait and stdio close, and sanitized bounded JSON-RPC error output that omits error.data.

Task 3 final robustness follow-up red evidence: helper pytest failed 4 tests before implementation for bounded stdout queue behavior, write OSError cleanup, flush ValueError cleanup, and success cleanup ordering. Final green evidence: helper pytest reports 22 passed and 5 warnings in 38.12s. Bandit on acp_certification_smoke.py reports 0 findings. git diff --check is clean. Implementation bounds stdout line and queue size, drops notifications and wrong-id responses before enqueueing, pauses read-ahead after matching the expected id, handles OSError and ValueError write or flush failures through centralized cleanup, and closes stdin before waiting on successful probe shutdown.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
