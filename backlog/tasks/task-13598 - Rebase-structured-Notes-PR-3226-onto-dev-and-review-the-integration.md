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
updated_date: 2026-10-12 21:21
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate the already reviewed local structured-document checks and reversible formatting with current dev so the existing draft PR has no merge conflicts. Preserve current upstream owners and source/metadata safety, then obtain a fresh independent review before updating the PR. Physical IME/spoken output qualification and separately deferred durable Notes sync remain outside this integration claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The PR branch is rebased onto the recorded current dev base, resolves its conflicts and retains the intended structured-Notes changes without deleting unrelated upstream behavior or documentation.
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
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Replayed the structured feature onto recorded dev 2a8ffa78, preserving current Notes save/receipt/provenance, Sync readiness/transactions, WYSIWYG lifecycle and API-types fingerprint owners. Allocated SQLite79/PostgreSQL83 with real upgrade/bootstrap proof. Complete language snapshots now compose into newer provenance capture/bootstrap, with legacy-null compatibility and exact structured source fences. Root reconciled source navigation with current G01/NS01 wait/flush behavior before revised tests; obsolete immediate-navigation RED is retained. ADR070–074 apply; no new boundary or ADR.

Fresh private source/config/dependency binding, targeted storage/API/provenance/recovery/native/worker/import/export checks, current typecheck, scoped Ruff and zero-finding Bandit are recorded in /private/tmp/structured-notes-pr-rebase-2026-10-12/s-rebase-report.md and s-evidence. Real owned PostgreSQL18.6 fixture passed79selected checks, including82→83 and current bootstrap/lifecycle; exact fixture cleanup succeeded. Existing current-dev ESLint/whole-file formatting debt is baseline-accounted rather than suppressed. Physical IME/spoken output and deferred C durable Notes sync remain unqualified. Root independent review/publication remain pending; task and ACs stay open.

Review round1 corrected local-storage refusal before publishing the current shared save reservation, preserving synchronous durable-write and wait/flush ownership. Actual authority-bound single-note export now carries the validated Knowledge head; portable exports reuse existing marker and independent provenance import carriers without altering structured source or source downloads. Canonical API-types package generation under checked-in route policy regenerated both committed artifacts and rich local types (2114paths/3267schemas); the required drift command and supported typecheck pass. Bounds E2E now asserts disabled Format. Additive red/green and actual-action import evidence, timeout attempts and their scoped disposition are in s-review-fix-report.md; scoped independent re-review remains pending.

The actual active-Sync import control also exposed a missing language field in immutable compound ACKs. The existing response owner now conditionally projects content_language from its durable payload, retaining omission/null/known/future values without consulting a newer row. Real create/update/idempotent replay controls after newer source changes passed; the original RED failures remain recorded. Ordinary portable JSON stays marker-only and preserves the destination canonical head on overwrite; structured/future records reuse the validated independent carrier and refuse stale overwrite without changing source/language/head. Matched Studio baseline/current diagnostics passed original assertions/deadline after exact temporary instrumentation restoration; original uninstrumented/batch failures remain disclosed. No further Studio product change or qualification waiver. Root scoped re-review/publication remain pending.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
