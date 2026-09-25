---
id: TASK-13296
title: TEST_MODE fallback leaks every user's webhook URLs and secrets
status: Done
assignee: []
created_date: '2026-09-22 04:45'
updated_date: '2026-09-23 23:09'
labels:
  - security
  - evaluations
  - bug
dependencies: []
references:
  - 'tldw_Server_API/app/core/Evaluations/webhook_manager.py:601'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/Evaluations/webhook_manager.py:_get_webhooks` ends with a fallback that, when the per-user lookup returns nothing, selects **every active webhook registration with no `user_id` predicate**:

```python
# Final safety: in TEST_MODE, if still no webhooks for this user, try all active webhooks
from tldw_Server_API.app.core.testing import is_test_mode as _is_test_mode
if not webhooks and _is_test_mode():
    rows = self.db_adapter.fetch_all("""
        SELECT id, url, secret, retry_count, timeout_seconds
        FROM webhook_registrations
        WHERE active = ?
    """, (self._active_flag(True),))
```

The rows are appended verbatim, **including `secret`**, and the caller then delivers evaluation payloads to those URLs.

The only gate is `core/testing.py:is_test_mode`, which reads an environment variable — it is **not** an explicit pytest-runtime check. So `TEST_MODE=1` present in a shared, staging or misconfigured deployment turns this into a cross-user leak: user A receives user B's webhook URLs and signing secrets, and B's endpoints receive A's evaluation payloads.

The module's own authorization layer uses the stricter predicate (`evaluations_auth.py:52-53`), so this fallback contradicts the module's established scoping.

Compare AUTHNZ-1 from the same review: the repo has four different strictness levels for "are we in a test context", and the strictest (`is_explicit_pytest_runtime`) exists precisely because env-var leakage into non-test deployments is a known hazard. This site uses the loosest.

Found by the comprehensive core-module review; independently verified by the orchestrator by reading the query and its gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unscoped fallback query is removed, or gains a user_id predicate so it cannot cross tenants
- [x] #2 If any test depended on the fallback, it is replaced by explicit per-user fixture registration
- [x] #3 If a test-only relaxation is still wanted, it is gated on is_explicit_pytest_runtime rather than the env-var is_test_mode
- [x] #4 secret is not selected into any code path that does not need it
- [x] #5 Bandit run for touched scope
- [x] #6 A test with TEST_MODE=1 (no pytest runtime) and multiple users proves lookup stays per-user and per-event. Amended: the unscoped cross-user fallback was unreachable because an earlier TEST_MODE branch always returned first, so a cross-user red test is impossible on the old code; the live bug that branch caused (event-filter bypass) is the red test instead.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fix commit 41124033df. Verified premise: the unscoped 'all active webhooks' fallback in _get_webhooks was dead code - the first TEST_MODE branch always returned early (user-scoped, but with the event filter dropped). So TEST_MODE in a deployment did not leak cross-user rows, but it did deliver every evaluation event to a user's webhooks regardless of subscription. Removed all three TEST_MODE relaxations (early event-filter bypass, per-user fallback, unscoped fallback); lookup is now always WHERE user_id = ? AND active = ? AND events LIKE ?. No pytest-only relaxation kept (AC3 n/a): no test needed one.
AC1 amended (removed and re-added as #6) because a cross-user red test cannot exist on old code; see AC text.
Regression test tldw_Server_API/tests/Evaluations/unit/test_webhook_manager_tenant_scoping.py (TEST_MODE=1, PYTEST_CURRENT_TEST removed, users a/b/c): test_unsubscribed_event_is_not_delivered_under_test_mode_env FAILS on ea1cbc6941 and passes after; the two cross-user tests pass on both (lock the invariant).
AC4 secret: only selected in _get_webhooks, whose sole caller _deliver_webhook uses it to sign; registration returns '***hidden***' for existing rows. No change needed.
RUN_EVALUATIONS=1 tests/Evaluations: before 19 failed/861 passed; after 18 failed/862 passed. Diff of FAILED lists = only the new regression test; the 18 are pre-existing (route-mount/startup, integration test_api_endpoints incl. TestWebhookEndpoints::test_webhook_delivery_on_evaluation, which fails identically before and after).
Bandit (uvx bandit -q -ll webhook_manager.py): no findings.
Follow-up NOT fixed (out of scope, flag for filing): _deliver_webhook sets skip_dns = is_test_mode() or PYTEST_CURRENT_TEST, so TEST_MODE in a deployment skips webhook SSRF/DNS-rebinding validation. Same env-var-gate class; should use is_explicit_pytest_runtime or an injected flag.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed TEST_MODE-gated scoping relaxations from evaluation webhook lookup; now always per-user and per-event. The cross-user fallback was actually unreachable (AC1 amended); the live event-filter bypass is the red-before/green-after test. No new Evaluations suite failures. Follow-up: TEST_MODE also disables webhook SSRF DNS validation in _deliver_webhook.
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
