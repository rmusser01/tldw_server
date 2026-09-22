---
id: TASK-13296
title: Remove unreachable unscoped webhook query before a refactor activates it
status: To Do
assignee: []
created_date: '2026-09-22 04:45'
updated_date: '2026-09-22 05:19'
labels:
  - security
  - evaluations
  - bug
dependencies: []
references:
  - 'tldw_Server_API/app/core/Evaluations/webhook_manager.py:601'
priority: medium
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
- [ ] #1 A failing test with TEST_MODE=1 and two users proves user A receives user B's webhook rows including secret
- [ ] #2 The unscoped fallback query is removed, or gains a user_id predicate so it cannot cross tenants
- [ ] #3 If any test depended on the fallback, it is replaced by explicit per-user fixture registration
- [ ] #4 If a test-only relaxation is still wanted, it is gated on is_explicit_pytest_runtime rather than the env-var is_test_mode
- [ ] #5 secret is not selected into any code path that does not need it
- [ ] #6 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CORRECTION by the filer, 2026-09-21. This task was filed as a live cross-user leak. That was wrong, and the error was mine: I read the unscoped query and its is_test_mode() gate but did not trace reachability.

The block is UNREACHABLE. _get_webhooks opens with 'if _is_test_mode():' whose branch returns unconditionally (webhook_manager.py:548 guard, return at :555-564). So the two later fallbacks -- both guarded by 'if not webhooks and _is_test_mode()' -- can only be evaluated on the path where _is_test_mode() is False, where their own guard is therefore False. Neither ever executes, in either mode.

Proven, not reasoned: tests/Evaluations/unit/test_webhook_manager_user_scoping.py asserts that no query lacking a user_id predicate is issued, parametrized over TEST_MODE set and unset. All six tests pass against the UNFIXED code -- which is the demonstration that there is no live leak.

What remains true and why this is still worth doing: an unscoped 'SELECT id, url, secret, ... FROM webhook_registrations WHERE active = ?' sat in the file, dormant, behind a condition that can never be true. The early return keeping it dead is itself a TEST_MODE hack and a plausible refactor target; removing it would have activated a cross-user disclosure of webhook URLs and signing secrets. Removed rather than left dormant, with the parametrized test as the guard against reactivation.

Severity corrected High -> Medium: latent hazard and dead code, not a live disclosure. The is_test_mode-reads-an-env-var observation still stands as a general concern (see AUTHNZ-1, which documents four strictness levels for test-context detection), but it is not exploitable here.
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
