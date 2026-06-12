---
id: TASK-2351
title: Add MCP approval lease flow for ask permission decisions
status: Done
assignee: []
created_date: '2026-06-11'
updated_date: '2026-06-11'
labels:
  - mcp
  - policy
  - security
  - approvals
  - followup
dependencies:
  - TASK-2349
  - TASK-2350
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the top deferred item from TASK-2349: an approval lease flow so matched `ask` permission-rule decisions can be satisfied by a TTL-bound operator-issued grant instead of being hard-blocked. Operators grant a short-lived approval for one (profile, subject_type, normalized value) tuple, optionally scoped to a session; the gateway runtime converts matching `ask` decisions to allow until expiry. `deny` outcomes are never overridden.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reusable TTL-bound policy grant store exists with memory and SQLite backends, exact-match on normalized subject values, session scoping, expiry, and periodic cleanup.
- [x] #2 The gateway runtime consults active approval leases for matched ask decisions before raising approval_required; deny rules are never overridden; expired or session-mismatched leases do not apply.
- [x] #3 Ask denials report redacted approval availability metadata; approved calls carry a redacted lease marker (not the raw grant id) in delegated context metadata.
- [x] #4 Operators can create, list, and revoke approval leases through the gateway CLI against a persistent (sqlite) grant store configured via the gateway config `policy_grants` section, with TTL clamps and audit events.
- [x] #5 Bootstrap wiring exposes the grant store on bootstrap results and config-driven bootstrap builds it from `policy_grants`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
New `mcp_unified/policy_grants/` package modeled on `filesystem_locks/`: frozen `PolicyGrant` dataclass (grant_id, profile_id, grant_type approval|path, subject_type, value, actions, effect, session_id, user_id, granted_by, reason, expires_at, ttl_seconds, safe_payload), `InMemoryPolicyGrantStore` (RLock + periodic sweep), `SQLitePolicyGrantStore` (SQLAlchemy, epoch_us expiry, batched cleanup, secrets.token_urlsafe ids), and `create_policy_grant_store()` factory (grant_store_backend memory|sqlite). The schema includes `actions`/`effect` now so TASK-2301 path grants need no migration. Subject values are normalized at grant and lookup time via the new public `normalize_permission_subject_value()` wrapper in `profiles/permission_rules.py`, so a grant for `https://Example.com/x` and a runtime subject `example.com` agree. Session semantics: grant with session_id=None applies profile-wide; a session-scoped grant matches only its session and wins over a global grant when both match.

Runtime hookup in `gateway/profile_runtime.py`: `ProfileAwareGatewayRuntime` accepts optional `policy_grant_store` (absent preserves the previous hard-block behavior). On a matched ask decision, `_active_approval_lease_marker()` consults `find_active_grant`; store failures fail closed to denial (noqa BLE001 documented inline). Approved subjects contribute a redacted SHA256[:16] marker of the grant id, attached to the delegated context metadata under `mcp_policy_approval_grants` so tool-use reporting can record it without leaking the revocation token. Ask denials gain `provenance.approval = {available, grant_type, subject_type}`.

Management surface: `gateway/policy_grants.py` `GatewayPolicyGrantManager` (TTL clamped to [60, 86400], default 900; audit events policy_grant.approval.created / policy_grant.revoked via the AuditStore pattern from credential grants). CLI verbs `create-approval-grant`, `list-approval-grants`, `revoke-approval-grant` in `gateway/cli.py`, requiring a persistent sqlite `policy_grants` store from the gateway config (reason_code policy_grant_store_unavailable otherwise). Config: `GatewayPolicyGrantStoreConfig` + `policy_grants` field on `GatewayProfileBootstrapConfig` + `build_gateway_policy_grant_store()`; `bootstrap_profile_gateway` passes the store to the runtime and exposes it on `GatewayProfileBootstrap`.

TDD evidence:
- Store red: 13 tests in new test_policy_grant_stores.py failed with ModuleNotFoundError before the package existed; green after.
- Runtime red: 5 tests in test_gateway_fastapi_package.py failed with TypeError (unexpected policy_grant_store kwarg); green after hookup.
- Manager/config/bootstrap red: 7 tests in new test_gateway_policy_grant_manager.py failed on missing modules/fields; green after.
- CLI red: 3 tests in test_gateway_cli_package.py failed on unknown subcommands; green after.

Verification:
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_policy_grant_stores.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_grant_manager.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_policy_decisions.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py -q` passed with 383 tests.
- Ruff over all touched modules and tests passed (fixed one pre-existing F821 missing Mapping import in test_gateway_cli_package.py).
- `python -m compileall -q` over touched modules passed.
- Bandit over mcp_unified/policy_grants/ and gateway/policy_grants.py reported no findings.
- `git diff --check` passed.

Deferred: pattern-valued approval grants (exact normalized match only); REST admin endpoints for grants (CLI only); CLI audit events currently use the profile storage bundle's audit store; consuming/limiting lease use counts.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a TTL-bound policy grant store (memory + SQLite) and wired approval leases into gateway runtime permission-rule enforcement so operator-granted, optionally session-scoped leases convert matched ask decisions to allow until expiry, never overriding deny. Includes CLI grant/list/revoke management with TTL clamps and audit events, config/bootstrap wiring, and redacted lease markers in delegated context metadata.
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
