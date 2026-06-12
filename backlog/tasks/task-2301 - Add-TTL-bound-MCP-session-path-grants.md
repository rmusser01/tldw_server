---
id: TASK-2301
title: Add TTL-bound MCP session path grants
status: Done
updated_date: '2026-06-11'
labels:
- mcp
- policy
- security
- session
- followup
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
dependencies:
- TASK-2351
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add temporary path grants scoped to a session, run, or short TTL so operators can grant one-off elevated filesystem access without permanently changing a profile. The grant resolution path should merge temporary grants into effective policy with auditability, expiry, and safe preview output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Operators can create TTL-bound path grants (profile, workspace-relative prefix, file-policy actions, optional session scope) without editing the profile document.
- [x] #2 Path grant prefixes are normalized and validated like authored path grants (no absolute paths, drive prefixes, or '..' segments); actions are validated against the file-policy action taxonomy.
- [x] #3 Active applicable grants are merged into the delegated effective policy's path_scopes with source/grant_id/expiry provenance; expired and session-mismatched grants are excluded; base profile scopes are preserved.
- [x] #4 Grants expire automatically and can be listed and revoked through the gateway CLI, with audit events on create and revoke.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Built on the TASK-2351 `mcp_unified/policy_grants/` store using `grant_type="path"` (schema already carried `actions`/`effect`, so no migration was needed). Validation lives in `policy_grants/models.py`: `validate_grant_request` now normalizes path-grant values through the new public `normalize_path_grant_prefix()` wrapper in `profiles/path_grants.py` (rejecting absolute paths, Windows drive prefixes, and `..` segments), and the new `validate_grant_actions()` requires at least one action and checks each against `PATH_GRANT_ACTIONS`. Both memory and SQLite stores call it in `create_grant`.

Runtime merge in `gateway/profile_runtime.py`: `_policy_with_ttl_path_grants()` runs in `_call_backend_tool_through_policy` after policy resolution and before `_context_with_effective_policy`, keeping `build_effective_policy_result()` pure. It lists active `path` grants for the profile, filters by `matches_session(_context_session_id(context))`, and appends flat scopes `{prefix, actions, effect, source: "ttl_grant", grant_id, expires_at}` to `policy.path_scopes` via `model_copy`. Store failures leave the base policy unchanged (never widen on error); no applicable grants means the policy object passes through untouched.

Management: `GatewayPolicyGrantManager.grant_path()` with the same TTL clamps as approval leases ([60, 86400], default 900) and `policy_grant.path.created` audit events. CLI verb `create-path-grant` (--profile, --prefix, --actions CSV, --ttl-seconds, --session-id, --granted-by, --reason); `list-approval-grants` gained a `--grant-type` filter; `revoke-approval-grant` works for both grant types.

TDD evidence:
- Store red: 3 new tests (prefix normalization, unsafe values, invalid actions) failed against the permissive TASK-2351 validation; green after.
- Manager red: 2 grant_path tests failed with AttributeError (method missing); green after.
- Runtime red: merge test failed (no ttl_grant scopes in delegated effective policy); green after. The inapplicable-grants test (expired + session mismatch) pins that such grants stay excluded.
- CLI red: 2 tests failed on unknown create-path-grant subcommand; green after.

Verification:
- `python -m pytest` over test_gateway_fastapi_package.py, test_gateway_cli_package.py, test_policy_grant_stores.py, test_gateway_policy_grant_manager.py, test_profile_permission_rules.py, test_profile_policy_decisions.py, test_filesystem_lock_managers.py passed with 392 tests.
- Ruff over all touched modules and tests passed.
- `python -m compileall -q` over touched modules passed.
- Bandit over mcp_unified/policy_grants/ and gateway modules reported no findings.
- `git diff --check` passed.

Deferred: temporary deny-effect grants (permanent deny belongs in the profile); pattern-valued prefixes beyond prefix semantics; surfacing TTL grants in the effective permission preview endpoint; ToolUseEvent marker when a merged grant was actually consulted by the downstream path enforcer.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added TTL-bound, optionally session-scoped path grants on the shared policy grant store: validated workspace-relative prefixes and file-policy actions, runtime merge of active applicable grants into delegated effective policy path_scopes with ttl_grant provenance, and CLI create/list/revoke management with audit events and TTL clamps.
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
