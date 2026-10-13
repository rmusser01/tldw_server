---
id: TASK-13598
title: Rebase structured Notes PR 3226 onto dev and review the integration
status: In Progress
created_date: 2026-10-12 19:44
assignee:
- '@codex'
labels:
- notes
- structured-editing
- review
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/3226
updated_date: 2026-10-13 01:09
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the already reviewed local structured-document checks and reversible formatting with current dev so the existing draft PR has no merge conflicts. Preserve current upstream owners and source/metadata safety, then obtain a fresh independent review before updating the PR. Physical IME/spoken output qualification and separately deferred durable Notes sync remain outside this integration claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The PR branch is rebased onto the recorded current dev base, resolves its conflicts and retains the intended structured-Notes changes without deleting unrelated upstream behavior or documentation.
- [ ] #2 Targeted verification of affected real storage, source and language preservation, recovery, native editor and import/export paths qualifies the integrated source; conflict resolutions do not weaken authority, exact-source or undo safeguards.
- [ ] #3 Schema and migration allocations are compatible with current dev and scoped lint, formatting, type and security checks introduce no unexplained findings; real upgrade/bootstrap controls cover migration conflicts.
- [ ] #4 An independent requesting-code-review review of the rebased PR range is recorded; Critical and Important findings are fixed or explicitly reconciled with technical evidence before publication.
- [ ] #5 The existing PR is updated with the reviewed rebased head using an exact expected-head force-with-lease, retains dev as its base and documents actual checks and remaining qualification limits without enabling merge or claiming unverified physical coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR for rebase/direct existing-owner integration; ADR path: Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md; Docs/ADR/071-074 for native/editor boundaries. Reason: retain approved existing boundaries and adapt only verified conflict consequences; record a new ADR before adopting any genuinely new architecture.
1. Read this task, current AGENTS/Architecture references, original design, accepted final review, publication provenance and current dev conflict owners.
2. Preserve exact published head/backup; rebase onto recorded dev, replay only structured-feature changes, resolve conflicts by combining actual upstream/feature behavior, preserve original design/source safety and unrelated dev documents. Correct migration/identifier collisions with actual current allocations and upgrade controls.
3. Verify actual rebased module/config/dependency binding in private owned environments before targeted tests; run affected source/storage/recovery/native/worker/import/export and relevant scoped static/security checks with no broad sweep or user data.
4. Save exact resolution decisions, paths/commands/output/source hashes and limitations in a report; obtain Root-dispatched independent review of dev-to-rebased-head, address confirmed findings and verify amended controls.
5. Commit explicit integration/task paths, leave original physical qualification tasks open, and let Root publish the independently reviewed branch using exact expected-head force-with-lease; no merge or auto-merge.
6. Current-owner disposition (Root, 2026-10-12): retain the single current wait/flush navigation owner from G01/PR3164, documented in Docs/superpowers/plans/2026-10-03-notes-chat-ux-remediation-plan.md and Docs/Development/2026-10-04-ux-remediation-integration-log.md (NS-01/02/03/05/06). No distinct canonical save/navigation ADR exists in current Docs/ADR; ADR070/071 still govern structured source/recovery/native boundaries. The old immediate source-navigation expectation is obsolete; preserve its RED evidence. Revised controls must retain newest exact source/mode/revision/base durably while the old request is pending, forbid navigation until settlement, then verify newer accepted source or explicit Stay/Discard/Continue recovery and restart. Do not add a source-only bypass or indefinite wait policy; preserve current transport timeout/authority cancellation controls.
7. Independent review fixes: settle active save promise and reducer on local storage failure; verify repaired-storage Save and navigation retries without loss/bypass. Preserve canonical, legacy, deleted and unsupported Knowledge head precedence through existing ordinary Markdown and portable export/import, while source downloads and structured bodies stay exact. Regenerate canonical API-types path union/fingerprint under repository-equivalent route policy with the supported generator and backend-required drift check. Correct bounds E2E to assert unavailable Format and unchanged source. Focused actual-owner regressions, scoped checks, additive evidence and independent scoped re-review are required before Root expected-head lease publication; no new carrier/transport or qualification waiver.
8. Additional confirmed integration defect from real active-Sync export/import control: the immutable compound response projection omits content_language from the durable note-step payload, so POST ACK returns null although GET returns the persisted structured mode. ADR required: no new ADR; ADR070 and the existing durable compound-response owner apply. Preserve payload field presence, explicit null and known/future language exactly in that owner; do not read the current row, infer language or weaken replay/receipt authority. Before/after real active compound POST/update/idempotent-replay controls must retain exact source/language/head including newer-row changes and legacy omission. Preserve initial RED evidence, run focused touched-owner Ruff/Bandit/format checks, and include this finding and fix in the independent scoped re-review before Root publication. Existing source/language/currentness ACs cover this regression; no new carrier, schema, transport or policy.
9. Current-head CI follow-up under the user's continuing approval: retain complete failed GitHub job logs and reports s-backend-ci-diagnosis.md / s-frontend-ci-diagnosis.md. Preserve prior source reviews and failures; reopen verification/review/publication accounting. ADR required: no new ADR for restoring existing contracts/fixtures and idle-query admission; ADR paths: 031,034,070-074. The capability-query activation signal remains ephemeral within the existing page/editor interface and preserves authenticated scope, currentness and write-time negotiation; no new durable rule, API, schema, dependency or cache.
10. Correct obsolete source-bound save mocks and known Markdown/capability fixtures, held-ACK and encoded-ID settlement expectations, module-relative raw fixture binding, unknown-mode duplicate affordance and precise Export-submenu selector. Add real idle/active editor query controls and activate the existing display capability lookup only for actual draft/selection/restored content, preserving immediate Create-before-typing and current write/offline guards. Never weaken ACK/version/language/source/authority checks, budgets, first-attempt E2E ratchets or unknown-mode protection.
11. Seed deliberate historical SQLite58/59/60 and PostgreSQL64 fixtures with schema-specific real SQL as existing tests do; preserve all old catalogs, graph/constraint/rollback/concurrency assertions and current constructor upgrades. Update exact-title/error, supported DB signatures, current PostgreSQL83 and Sync-v2 capability/readiness/safety-query expectations. Retain no-secret refusal, v1 shape, immutable replay and cursor/append fences. No production old-schema fallback or new migration is justified. Repair isolated PDF/Monaco dependency fixtures using real preparation owner, not mocked helper or skipped copy checks.
12. Regenerate Published docs through the canonical refresh script; retain exact source/private boundaries, commit generated mirrors for revision-date history and verify deterministic refresh/strict build without suppressions. Use isolated current-source private environments and serialized focused RED/GREEN checks, selected real PostgreSQL fixtures and targeted hosted Notes print/request-budget controls where practical; no local full sweep or physical IME/spoken claim. Run scoped formatting/lint/type/security, independently review actual changed range, and update existing draft PR with exact expected-head lease and truthful residual limits; no merge/auto-merge.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased the structured feature onto dev 2a8ffa78, preserving current Notes save/receipt/provenance, wait-and-flush navigation, Sync readiness/transactions, WYSIWYG and API-types owners. Allocated SQLite 79 / PostgreSQL 83 with real upgrade/bootstrap/lifecycle proof. ADR-070 through ADR-074 retain the approved boundaries; no new architecture, schema or transport was added during the review fixes.

Independent full rebased-PR review found three Important issues and one Minor issue. Focused real export/import checks found a fourth Important issue: the immutable compound acknowledgment omitted persisted language. Existing-owner fixes handle storage refusal before publishing a save promise, retain authoritative provenance through actual export/import actions, regenerate canonical API contracts, conditionally project language from the durable acknowledgment payload, and assert disabled Format after a resource-limited Check. Ordinary portable JSON preserves marker-based overwrite behavior; structured/future records use the existing validated carrier and fail-closed optimistic overwrite. Exact source downloads and immutable replay after newer rows remain protected.

Fresh final controls passed 5 storage/single-inflight/unchecked-Format cases, 36 export/import cases and 28 acknowledgment/overwrite cases. Supported frontend typecheck, scoped ESLint (0 errors / 68 inherited warnings), amended Python Ruff/format and zero-finding Bandit passed. Canonical generation/drift uses the repository-equivalent route policy: 2,114 paths / 3,267 schemas, including Notes capabilities. The actual typecheck consumes index.d.ts and paths.d.ts; generated rich schema.d.ts was not a frontend check input. Prior real PostgreSQL and worker evidence remains separately source-bound; unchanged owners were not rerun.

Independent scoped re-review of 5733e139 to f0fa621d3e4cd8be5b293a867520e539fed5add7 accepted I1/I2/I3/M1/I4 with 0 new Critical, Important or Minor source findings. Root fully read both reports and independently verified 12 source files, 5 generated artifacts, 187 receipts, exact package hashes, clean head, dev ancestry and diff checks. Source gate PASS is separate from test, merge and release qualification.

Published the reviewed source head to existing draft PR https://github.com/rmusser01/tldw_server/pull/3226 with exact force-with-lease against original head 3cfe8fc8d622c501e1f5ad1dacda2242b71673dc. Updated its description with actual checks and limitations. GitHub confirmed dev base, exact head, MERGEABLE, draft and no auto-merge. Subsequent task-only bookkeeping preserves reviewed source/tests.

All integration acceptance criteria are accounted for, but status remains In Progress under repository DoD: broader recovery attempts remain 184 pass / 14 fail and 9 pass / 5 fail; four controls passed unchanged in isolation, while the Studio uninstrumented timeout remains unqualified. Matched pre-fix/current diagnostics passed original assertions/deadlines with exact source restoration; they do not establish a load-only cause or a clean recovery-suite result. Bounds discovery predates the final title and is not frozen-head/hosted proof. Whole-file formatting debt, physical IME/spoken output, current-head human Change summary/CI/review gates and separately deferred Chatbook durable Notes sync remain open. No full suite or gate waiver. Reports, raw failed attempts, scoped confirmation and immutable manifests are retained under /private/tmp/structured-notes-pr-rebase-2026-10-12. Existing physical tasks TASK-13587 and TASK-13597 remain open.
Current-head CI follow-up is in progress after the user requested continuation. Previous source reviews/publication remain historical; current failed-job evidence and root-cause reports are under /private/tmp/structured-notes-pr-followup-2026-10-13. Verification, review and publication are reopened. The repository-local Python CLI lacks --plan; the installed Python backlog-py CLI supplies this field, followed by the repository canonical-format check. No Node CLI, MCP or hand-edit was used.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed rebase and four Important source corrections published to draft PR3226 on dev2a8ffa78. Scoped re-review accepts all findings with no new source issue. Fresh 5/36/28 selections and scoped static/security checks passed; reliable uninstrumented recovery, whole-file formatting, physical qualification and merge gates remain open.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
