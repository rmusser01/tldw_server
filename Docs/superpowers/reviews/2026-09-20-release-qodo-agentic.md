# Release candidate deep Qodo review

Tracking: TASK-13263.1. Review: [PR2972 agentic review](https://github.com/rmusser01/tldw_server/pull/2972#issuecomment-5752955730).

This is a separate set of 18 findings following the 23 original PR2761 findings and the later Notes reviews. A row is closed only after its fix or disposition is verified.

| # | Finding | Status and evidence |
|---|---|---|
| 1 | Release source disagreement | Reconciled current inventory, plan and Backlog authority with release.json; old checkpoints explicitly historical. New CI test fails on the old documents. Final source/manifest refresh follows all protected edits. |
| 2 | Missing chat loading state | Fixed; focused regressions pass. See the frontend/backend reports below. |
| 3 | Media account authority | Fixed and independently checked, including search/detail cache retirement, page remount, bulk/undo operation ownership, owner-stamped collections/favorites and removal of unowned media-type cache. See the media-security follow-up. |
| 4 | User message fallback name | Fixed; focused regressions pass. See the frontend/backend reports below. |
| 5 | Image-detail retry persistence | Fixed; focused regressions pass. See the frontend/backend reports below. |
| 6 | OSCE endpoint helper docstrings | Fixed; source and owning tests verified. |
| 7 | OSCE focus timers | Fixed; focused regressions pass. See the frontend report below. |
| 8 | Shared UI coverage omitted | Added separate bounded coverage report under apps/packages/ui, using its own Vitest setup/aliases. The WebUI exclusion prevents running UI under the wrong package configuration. Required package-owned UI unit shards already enforce results. Existing report-only coverage policy remains: AGENTS says aim for >80%, not a repository-wide blocking threshold; this repair does not claim 80% has been achieved or silently expand the separately scoped global frontend work. Workflow regression was red before this change, green afterward. |
| 9 | Two host test classifications | Rejected: integration is the sole test-type classification; vz_linux_host_failure_drill is a registered orthogonal manual-selection marker (pyproject.toml). skipif is a platform condition. Removing the selector would weaken operator selection; removing integration would omit the real scenario from category collection. |
| 10 | OSCE loose null checks | Fixed with strict checks retaining null/undefined behavior. |
| 11 | Prompt focus timer | Fixed; focused regressions pass. See the frontend/backend reports below. |
| 12 | OSCE schema helper docstrings | Fixed; source and owning tests verified. |
| 13 | Real VM operations | Intentional approved integration acceptance, not a unit test. TASK-13243.3 explicitly requires real VSock missing-exec metadata and a negative control that actually executes when the guard is disabled. Separate E2E and fault-injection opt-ins, disposable bundles, explicit isolated helper, empty initial VM inventory, owned-resource cleanup and retained errors constrain the test. Portable fake-client tests cover cleanup failures and lost create replies. Replacing the live boundary with a fake would invalidate its purpose; existing live evidence is in Docs/Sandbox/vz-linux-prepared-host-evidence.md. No live VM operation was performed for this review. |
| 14 | OSCE untranslated guidance | Fixed; focused regressions pass. See the frontend/backend reports below. |
| 15 | Flashcard loose null checks | Fixed with strict checks retaining session ownership behavior. |
| 16 | OSCE finite request limit | Fixed: router-wide finite ingress limiter uses a Request-only wrapper; 429 and unchanged OpenAPI query contract verified. |
| 17 | Public setup model paths | Confirmed local anonymous exposure; remote setup already requires admin. Fix preserves model input validation/execution and authenticated configuration resume while redacting the public state. Independent security review completed; its client-resume finding is fixed with regressions. |
| 18 | Untracked host skips | Added TASK-13243.3 / GitHub issue1442 to each local prerequisite skip and documented intentional manual acceptance. Host eligibility is a prerequisite, not a disabled failing test. |

Parent verification: 14 workflow/portable host tests passed, one live integration deselected. Scoped Bandit (excluding test assertions) reports zero findings. Ruff has the same five pre-existing broad cleanup exception diagnostics as HEAD; those catches intentionally retain every cleanup error and do not swallow success. Current authority consistency test is separately validated. Final combined verification is recorded below; remote CI is tracked separately.

## Verified implementation reports and follow-up

[Frontend report](2026-09-20-release-qodo-agentic-frontend.md): 241 passing tests. [Backend report](2026-09-20-release-qodo-agentic-backend.md): 327 passing tests and one official PostgreSQL-unavailable skip, zero production Bandit findings. Parent rechecked 22 media boundary tests and 24 setup privacy/control cases. Parent workflow/source/portable host suite: 27 passed, live test deselected; cached actionlint1.7.12 passed. Candidate0.1.43 wheel/sdist, Twine and backend-only checks pass.

Fresh independent review identified surviving media mutations/persisted collections and a setup-client authenticated resume gap. The media follow-up now guards mutations/undo and scopes persistent records, including positive same-owner reload controls. Setup now sends available credentials and falls back to anonymous progress only for401; anonymous redacted resume offers provider reselection instead of a blank first-chat step. Three new regressions failed before the fixes, then105 setup-domain/hook/wizard/transport tests passed. The final source record is refreshed after these corrections; current authority lives in the release plan, inventory, Backlog description and release.json.


## Final integrated verification

- Parent package-owned frontend run: 255 passed across ten suites. Full WebUI TypeScript check passed. Touched tracked TypeScript ESLint baseline138/current135: no new diagnostics; new authority suite checked separately.
- Initial storage tests passed under the WebUI localStorage adapter but exposed a missing extension storage backend under the package config. Tests now provide browser.storage while retaining the real Plasmo hook; no production behavior, assertions, timeouts or skips were weakened to resolve the harness discrepancy.
- Backend verification remains327 passed/one official PostgreSQL-unavailable skip; setup frontend105 passed. Parent source/workflow/portable-host verification27 passed/one live integration deselected. Explicit protected-checkout verification is recorded in the release plan.
- Strict documentation build and curated regeneration pass. Candidate0.1.43 wheel/sdist, Twine and backend-only packaging pass. Scoped production Python Bandit reports zero new findings. Bandit does not analyze TypeScript; boundary tests, TypeScript/ESLint and independent source review are the relevant frontend evidence.
- Companion Chatbook PR2763 follow-up4030d6d58d:165 focused tests; parent separately reran29 mounted/HTTP controls; Bandit0; all seven inline Qodo threads resolved and the eighth summary-only item answered.

These focused counts overlap and are not a full repository test or fresh four-configuration UAT certification. No0.1.43 merge/publication occurred. Original PR2761 has23 resolved Qodo threads; recovery PR2973 is merged under explicit approval and has no unresolved review threads. All18 candidate Qodo threads received individual verified replies against pushed52e311431f and are resolved. A fresh complete GitHub thread inventory confirms no unresolved threads on original2761, candidate2972, recovery2973 or companion2763 (no additional pages). Required remote CI remains a release gate.

[Media security follow-up](2026-09-20-release-qodo-media-security-followup.md) records account boundaries, legacy storage migration impact and test configuration evidence. [Independent review](2026-09-20-release-qodo-agentic-security.md) retains the original findings plus their verified disposition.
