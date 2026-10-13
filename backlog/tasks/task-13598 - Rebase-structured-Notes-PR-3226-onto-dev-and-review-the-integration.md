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
updated_date: 2026-10-13 04:11
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
13. Published-head CI b8d8e298 revealed two feature-related verification defects (logs and read-only proposals retained under new-head-failures). ADR required: no new ADR; existing ADR070 durable recovery and unchanged CI partition ownership apply. Bind the canonical quota-failure spy to Object.getPrototypeOf(window.localStorage), the existing house pattern, and prove actual native JSDOM plus MemoryStorage receiver interception while preserving every source/selection/no-navigation/notice/finally assertion and original deadlines. Verify the owning deterministic CI config and focused canonical/authority storage controls without production or setup changes. Allocate test_historical_note_seed.py explicitly to chacha-content-persona in all five existing platform matrices; keep baseline44/ignore4, required Linux PostgreSQL and all markers/fixtures unchanged. Run the canonical coverage checker without baseline writes, scoped static checks and independent two-path review before exact-lease publication. No full sweep, host repetition, gate waiver or unrelated CI repair.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Structured YAML/JSON/JSONL checks and reversible formatting remain integrated on dev 2a8ffa78, with SQLite79/PostgreSQL83 and ADR031/034/070–074 contracts. Earlier review fixes preserve durable storage before save reservation, canonical Knowledge export/import provenance, generated API contracts and language in immutable acknowledgments.

CI corrections suppress the display capability query for inactive empty Notes while preserving Create, authenticated ownership and write/offline negotiation. Save fixtures acknowledge the exact submitted source/language/version; contract fixtures use the consuming module path, recovery failure injection uses actual localStorage, and Export selects the semantic submenu. Historical fixtures use real pre-language SQL and FTS/graph/cascade controls; runtime upgrades and exact titles remain unchanged. Seven Published mirrors reproduce unchanged canonical docs and have Git history. No new architecture, migration, dependency or guard exemption.

Fresh frontend selections passed56 owner/39 remediation/182 document/24 alternate-consumer cases, supported typecheck and lint comparison (0errors/116inheritedwarnings/0new). The source/ACK selection remains120pass/1storage-fixturefail plus a separate corrected1pass, not a clean121-case aggregate; its three uninstrumented Studio controls passed at original deadlines. Backend selections passed13 focused/79 historical/77 contract/15 real PostgreSQL zero-skip/34 docs-and-strict-build cases, source main/spawn binding and public/private boundary. Scoped Python comparisons introduced no Ruff/nonassert Bandit findings; changed-range/new-file format and AST checks passed, with ten inherited whole-file formatting debts retained.

Independent CI-delta review accepted33 paths against a39fe39b with no new Critical/Important/Minor finding. Root verified current review/source/evidence hashes; source606616eb contains the exact25 reviewed paths after generated-only9480bfe. Actual hosted606616eb diagnostics collected two cases, both first-attempt with0retry/skip: Notes held23 requests in the unchanged15s window with0idle capability requests; NE04 reached Export but reproduced its known blank-print-window defect. This establishes menu traversal, not functional printing. Real backend/readiness/auth, fresh extension build/current shared-source binding and complete service/private-runtime cleanup are recorded.

Status remains In Progress under repository DoD: inherited whole-file static/formatting debt, original broader recovery aggregates and current-head GitHub/human/merge gates remain separate. Physical tasks13587/13597 stay open; Chatbook durable Notes sync34818 stays deferred. No full sweep, physical waiver, merge or auto-merge. Immutable reports/manifests retain actual execution heads, format equivalence and failed attempts under /private/tmp/structured-notes-pr-followup-2026-10-13. Reviewed source is ready for lease-protected publication; publication accounting remains pending.
Published the reviewed source606616eb and accounting head0974c8dc to existing draft PR3226 using the exact expected-head lease against a39fe39b. GitHub confirmed exact head, dev base, draft, no auto-merge and matching truthful validation notes. This final task-only accounting preserves all reviewed source/tests; status remains In Progress for the documented repository DoD limits.
Published-head CI follow-up reopened: unit shard7/8 returned174pass/1native-storage-spy failure; coverage guard reports the one new historical seeder module unassigned. The dependent frontend-required failure is an aggregate. Prior review/host/source receipts remain historical and immutable; this plan is recorded before the two corrections.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed CI corrections published to draft PR3226 against dev2a8ffa78. Targeted frontend/backend/PostgreSQL/strict docs and first-attempt hosted23-request budget passed; known print defect, inherited static/formatting debt, physical and current-head GitHub/merge gates remain open.
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
