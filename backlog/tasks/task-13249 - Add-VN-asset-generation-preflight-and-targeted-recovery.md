---
id: TASK-13249
title: Add VN asset generation preflight and targeted recovery
status: In Progress
assignee: []
created_date: '2026-09-13 18:27'
updated_date: '2026-09-13 19:30'
labels:
  - vn-assets
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
  - 'https://github.com/rmusser01/tldw_server/pull/2954'
documentation:
  - Docs/Design/2026-09-13-vn-generation-readiness.md
  - Docs/Evidence/VN_Generation_Readiness_2026_09_13.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First productization slice for issue 2021: expose server-owned generation preflight and actionable configuration diagnostics, wire targeted slot retry using existing idempotent API, and verify state recovery. Reconcile issues 2021-2027 and parent links as accompanying tracking work. Full recipe snapshot and worker crash-replay hardening remain in 2021.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pack generation preflight reports effective backend availability and worker configuration without asserting external-worker health.
- [x] #2 The WebUI displays actionable generation failures and supports targeted slot retry with duplicate submission protection.
- [x] #3 Focused backend and frontend tests plus browser QA verify changed behavior; Bandit and relevant type checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented advisory owner-scoped preflight, stable generation keys, targeted retry, active progress polling and pack-switch guards. Mobile browser QA found and fixed implicit grid overflow. Independent review identified four refresh races; failing regressions reproduced each and all are fixed. Final re-review found no remaining issues in those fixes. Verification: 90 backend tests, 31 frontend tests, 3 Chromium smoke tests; touched ESLint and Bandit clean. Full typecheck fails identically on unchanged dev (90 existing diagnostics). See evidence document for commands, before/after observations and limits. Draft PR packaging in progress; human-written Change summary required before merge.

Draft PR #2954 opened against dev. Implementation and verification complete; task remains In Progress pending PR review/merge. Required human-written Change summary is explicitly outstanding in the draft. Temporary execution plan removed after all four stages completed; design and verification evidence are retained.

Requester supplied the human-written Change summary and authorized rebase, all PR/Qodo review fixes, and merge. Fresh fetch/rebase confirms dev c70387f496 is already the base (no rewrite). PR marked ready; Qodo review is running. backend-required failed because the new endpoint and two schemas were not reflected in the generated OpenAPI fingerprint/types. Regenerate using existing tooling, review the diff, verify the drift gate, and include any Qodo repairs before waiting on final-head reviews/checks.

PR #2954 Qodo review: fixed pack-scoped command locking, finite authenticated preflight rate policy, shared adapter/default-model resolution, direct core tests, and missing type/doc annotations. Matrix refresh finding disproven by a regression that passed before production changes; successful matrix application already replaces selectedPack. Regenerated omitted OpenAPI fingerprint with CI-compatible schema libraries; exporter --check passes exact CI hash. Verification: 86 preflight/model/API + 46 adapter/jobs + 9 catalog tests; 33 frontend tests; 3 Chromium desktop/mobile scenarios; scoped ESLint and Bandit clean. Typecheck retains exactly the 90 baseline diagnostics. Detailed disposition is in Docs/Evidence/VN_Generation_Readiness_2026_09_13.md. Awaiting final independent review, push, refreshed Qodo review and required CI before authorized merge.

Independent review also found JWT and API-key principals could receive separate preflight budgets for one user. Added opt-in per_user buckets to the shared RBAC factory/enforcer, enabled only for VN preflight; legacy endpoint semantics remain unchanged. Reproduced with failing API identity assertions, then verified 136 auth-hardening/VN API/core/model tests passing, including frozen-clock aggregate-budget and cross-user isolation tests. Scoped production Ruff and expanded Bandit both clean; final OpenAPI check still passes. All original required PR checks except the repaired fingerprint passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the bounded generation-readiness and targeted-recovery implementation for #2021. All scoped tests pass. Issues #2021-#2027 were reconciled and registered as children of #1391. Recipe snapshots, reload recovery and worker crash replay remain outside this slice; #2021 stays open. No live provider/GPU/worker deployment or extension build was exercised.

Delivery: https://github.com/rmusser01/tldw_server/pull/2954 (draft). No merge performed.
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
