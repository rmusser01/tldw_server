---
id: TASK-2303
title: Add MCP policy simulation CLI and admin harness
status: Done
updated_date: '2026-06-11'
labels:
- mcp
- policy
- cli
- admin
- followup
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
dependencies:
- TASK-2349
- TASK-2351
- TASK-2301
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add policy simulation tooling, for example `mcp policy simulate --profile backend-engineer --tool fs.patch --path src/foo.py --action edit`, so operators can validate profile/path policies before exposing them to agents. The harness should reuse the effective permission preview and path-enforcer decision contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Operators can simulate one profile policy decision for a hypothetical tool call (tool, paths, urls, command/argv, raw arguments JSON) through the gateway CLI and receive a deterministic JSON verdict.
- [x] #2 The simulation pipeline mirrors runtime ordering (legacy tool/capability decision, then permission-rule subjects, then approval-lease and TTL path-grant overlays) and reuses the same enforcement and merge code so verdicts cannot drift; a parity test asserts harness verdicts match actual runtime call outcomes.
- [x] #3 Output reports overall status (allowed/denied/approval_required), legacy policy status, per-subject decisions with matched rules and reason codes, approval lease markers, and merged effective path scopes including TTL grants.
- [x] #4 Simulation is read-only: it never writes audit events or consumes leases; unknown profiles produce a structured profile_not_found error.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Anti-drift extraction: moved the permission-rule subject extraction (subject key sets, bounds constants, limit error, and helpers) from `gateway/profile_runtime.py` into a new shared `mcp_unified/profiles/subjects.py` with public names (`extract_permission_rule_subjects`, `PermissionSubjectLimitError`, `MAX_PERMISSION_SUBJECTS` / `MAX_SUBJECT_VALUE_LENGTH` / `MAX_COMMAND_ARGV_TOKENS`). The runtime imports from there, so enforcement and simulation share one extractor. Refactor verified by the existing gateway suite staying green (191 tests) before the harness work continued.

Harness: new `mcp_unified/gateway/policy_simulation.py` `simulate_tool_call_policy(profile, tool_name, arguments, *, capability, policy_grant_store, session_id)`. It runs `build_effective_policy_result()` for the legacy decision, compiles permission rules (compile failure simulates as denied/invalid_permission_rules), reports informational per-subject decisions via `evaluate_permission_rule_decision` (unmatched subjects shown as effective "allow"), then derives the actual verdict by calling the runtime's own `_enforce_permission_rules_for_tool_call` (catching `GatewayPolicyDenied` and surfacing its `to_error_data()` payload), and finally reports merged path scopes via the runtime's `_policy_with_ttl_path_grants`. The parity test runs five (tool, arguments) cases through both the harness and a real `ProfileAwareGatewayRuntime.call_tool` and asserts identical statuses, covering allow, path deny, ask without lease, ask satisfied by lease, and legacy tool denial.

CLI: `simulate-policy --profile --tool [--path]... [--url]... [--command] [--argv-json] [--arguments-json] [--capability] [--session-id] --config`. The handler loads the gateway config, seeds `config.profiles` into the resolved profile store (so pure-JSON configs are simulatable), loads the profile (structured `profile_not_found` error otherwise), builds the optional policy grant store from `policy_grants`, and emits the harness payload as JSON. Simulation does not require a persistent grant store; without one, ask decisions simulate as approval_required.

TDD evidence:
- Red: 8 tests in new test_gateway_policy_simulation.py and 2 CLI tests failed with missing modules/subcommands; green after.
- Refactor safety: test_gateway_fastapi_package.py (191 tests incl. all subject-bound and permission-rule cases) green after the subjects.py extraction.

Verification:
- Combined suite (test_gateway_fastapi_package, test_gateway_cli_package, test_policy_grant_stores, test_gateway_policy_grant_manager, test_gateway_policy_simulation, test_profile_permission_rules, test_profile_policy_decisions, test_filesystem_lock_managers) passed with 402 tests; re-run 5x green. One unreproducible single failure of test_gateway_cli_path_grant_lifecycle was observed in one combined run immediately after the slice landed (passed in isolation, file-level, and 8 subsequent attempts) — recorded here as a watch item.
- Ruff over all touched modules and tests passed.
- `python -m compileall -q` over touched modules passed.
- Bandit over profiles/subjects.py and gateway/policy_simulation.py reported no findings.
- `git diff --check` passed.

Deferred: human-readable table output (CLI emits JSON like all other verbs); `--action` evaluation against merged path scopes (path scopes are reported for operator inspection; action-level path-enforcer simulation needs the downstream enforcer contract); remote admin endpoint for simulation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a read-only policy simulation harness and simulate-policy CLI verb that mirror the gateway runtime decision pipeline by reusing the shared subject extractor (newly extracted to profiles/subjects.py), the runtime enforcement function, and the TTL path-grant merge, with a parity test pinning harness verdicts to actual runtime outcomes across allow/deny/ask/lease/legacy-denial cases.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
