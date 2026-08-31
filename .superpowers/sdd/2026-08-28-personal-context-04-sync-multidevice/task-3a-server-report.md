# Task 3a server correction report

## Status

Ready for controller review. TASK-13148 status was not changed.

Commit: this report is included in the remediation commit; the exact SHA is
reported in the controller handoff.

## RED evidence

Command:

```text
PYTHONDONTWRITEBYTECODE=1 /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -k 'personal_context_bootstrap_endpoint_exposes_exact_content_free_attention or personal_context_bootstrap_endpoint_omits_untrusted_attention'
```

Result: `3 failed, 3 passed, 86 deselected`. The intended failures proved that
an unsupported required quota was absent instead of reported as zero, malformed
attention fields crossed the HTTP boundary, and a reason/kind mismatch crossed
the boundary.

## GREEN evidence

- Focused RED selection after implementation: `6 passed, 86 deselected`.
- Full affected endpoint/bootstrap modules: `125 passed`.
- Ruff over all four touched Python files: `All checks passed!`.
- Python 3.11 compilation of all four touched Python files: exit `0`.
- Bandit over the three touched production modules: exit `0`, `0` findings,
  `0` errors.
- Chatbook strict bootstrap-error parser accepted the unknown required quota
  with `available_quotas[name] == 0`.
- `git diff --check` and the base-range diff check: exit `0`.

## Changed files

- `tldw_Server_API/app/core/Sync/v2/profile.py`
- `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- `tldw_Server_API/app/api/v1/endpoints/sync.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- `IMPLEMENTATION_PLAN_personal_context_bootstrap.md`
- `backlog/tasks/task-13148 - Bootstrap-Personal-Context-canonical-profile.md`
- This report.

## Implementation summary

Quota incompatibility now reports every requested quota name in the available
map, using a safe zero when the server does not support the name. The HTTP error
mapper no longer trusts service-provided attention mappings: it validates exact
strict schema structure, bounded scalar values and names, semantic consistency,
and stable reason-code/kind agreement. Validation failure preserves the existing
status/reason/message response while omitting attention entirely.

## Self-review

- No canonical profile identifiers, wrapped keys, ciphertext, or arbitrary
  mappings can cross through invalid attention.
- Existing authentication, status codes, stable reason codes, and successful
  bootstrap response shapes are unchanged.
- The compatibility correction is additive for strict clients and preserves
  deterministic sorted insufficient-quota names.

## Known limitations and skips

The repository-wide suite was not run because repository guidance requires
explicit opt-in. Verification was limited to the affected authenticated endpoint
and Personal Context bootstrap modules. Existing Bandit `nosec` warnings in the
large endpoint module were reported by the scanner but produced no findings and
were not introduced by this correction.

## Final minor review correction

Status: ready for controller review; TASK-13148 status remains unchanged.

RED command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py -k 'unknown_zero_minimum_quota or has_no_materializing_sync_profile_helper'
```

RED result: `2 failed, 33 deselected`. The failures proved that an unsupported
zero minimum was incorrectly rejected and the obsolete materializing helper
remained exposed.

GREEN results:

- Focused selection: `2 passed, 33 deselected`.
- Complete affected bootstrap and authenticated endpoint modules: `127 passed`.
- Ruff: `All checks passed!` across the three touched Python files.
- Python 3.11 `py_compile`: exit `0` across the three touched Python files.
- Bandit: exit `0`, with `0` findings/errors across both production modules.
- `git diff --check`: exit `0`; production/test search found no remaining
  `ensure_sync_profile` definition or caller.

Implementation: quota compatibility now compares requested minima against a
zero default for unknown names. This makes zero a satisfied minimum while
preserving the existing positive-unknown typed attention contract. The unused
`PersonalContextService.ensure_sync_profile()` helper and its unused fake were
removed; absent-profile bootstrap remains reservation-only until the reviewed
completion boundary.

Changed files for this correction:

- `tldw_Server_API/app/core/Sync/v2/profile.py`
- `tldw_Server_API/app/core/Personalization/personal_context_service.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py`
- `IMPLEMENTATION_PLAN_personal_context_bootstrap.md`
- `backlog/tasks/task-13148 - Bootstrap-Personal-Context-canonical-profile.md`
- This report.

Known skip: the repository-wide suite was not run; verification remained scoped
to the affected bootstrap and authenticated endpoint modules.
