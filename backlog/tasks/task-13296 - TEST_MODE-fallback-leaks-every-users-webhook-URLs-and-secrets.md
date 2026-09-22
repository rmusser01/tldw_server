---
id: TASK-13296
title: TEST_MODE fallback leaks every user's webhook URLs and secrets
status: To Do
assignee: []
created_date: '2026-09-22 04:45'
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
- [ ] #1 A failing test with TEST_MODE=1 and two users proves user A receives user B's webhook rows including secret
- [ ] #2 The unscoped fallback query is removed, or gains a user_id predicate so it cannot cross tenants
- [ ] #3 If any test depended on the fallback, it is replaced by explicit per-user fixture registration
- [ ] #4 If a test-only relaxation is still wanted, it is gated on is_explicit_pytest_runtime rather than the env-var is_test_mode
- [ ] #5 secret is not selected into any code path that does not need it
- [ ] #6 Bandit run for touched scope
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
