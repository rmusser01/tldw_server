---
id: TASK-13522
title: Qualify structured Notes across editors sync files and recovery
status: In Progress
created_date: 2026-10-11 22:07
dependencies:
- TASK-13521
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-12 15:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove the complete structured-document workflow with real participating editors, storage and services before release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Real editors demonstrate type/format/type/undo/undo/redo, earlier typing, current identity, IME/focus/read-only/navigation and source-safe rendering in supported environments.
- [ ] #2 Real local/server saves, negotiated clear/private sync, language-only conflicts, copies, recovery and portable import/export retain exact source and language.
- [ ] #3 Mixed-version peers refuse unsupported operations without clearing metadata or claiming synchronization; empty rejected drafts recover after restart.
- [ ] #4 Real worker/deadline/navigation tests and1MiB/depth100/100-diagnostic boundaries preserve responsiveness and leak no source or credentials in logs.
- [ ] #5 Durable evidence names actual runtime versions, commands, screenshots/traces and unsupported/unrun cases; no full test sweep without explicit user approval.
- [ ] #6 Relevant ADR assessment, scoped static/security evidence and completion documentation are recorded without hiding unfinished release gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: C backlog/decisions/194-structured-note-language-and-local-editor-validation.md; S Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md, 066-local-monaco-asset-owner.md, 067-notes-temporary-language-mode.md, 068-exact-source-secondary-note-capture.md, 069-native-monaco-bom-source-adapter.md.
Reason: direct release qualification and scoped defect fixes of the approved source/metadata/transport/native-editor boundaries; a new boundary decision must be recorded before adoption.
1. Read the exact task, approved I5 brief/spec AC1–AC15, reviewed reports/scope resolutions and runtime ownership records; assemble a source-bound evidence matrix separating covered profiles from missing actual-owner release outcomes.
2. Exercise exact YAML/JSON/JSONL through actual local/server stores, APIs, each participating sync direction, copies/recovery/portable carriers, including omitted/null/future/language-only conflicts and mixed-version behavior. For the existing unavailable private Notes server ingress, qualify truthful refusal/no append and encrypted recoverable local pending metadata rather than invent transport.
3. Resolve each of the six inherited canonical-save failures with focused actual-owner evidence: cancelled source-open retention; newer typing during confirmation; unavailable linked-note retry; pending read across same-owner credential rotation; pending create across lineage-valid rotations/versioned PUT; acknowledgement during canonical refresh. Fix real defects within AC or prove invalid fixture assumptions before correcting them.
4. Fill missing native Database/File/web typing-format-typing-undo-undo-redo, stale/read-only/IME composition/focus/problem navigation and exact BOM/CRLF/final-newline/external-hash/recovery controls using actual editor entry paths. Verify compact TUI body paint and browser narrow/mobile/status accessibility; report real physical/platform scope precisely.
5. Qualify outstanding actual-app worker bounds/debounce/deadline/crash/startup/navigation churn and source/log privacy with responsive finite controls and owned-process/settled-promise cleanup. Reuse already source-qualified unit matrices; do not replay unrelated suites or infer physical zero overlap from native asynchronous termination.
6. Run targeted affected selections and scoped lint/format/types/security comparisons against the immutable Task15 bases. Use only owned synthetic runtime/browser/DB fixtures, verify fresh readiness, recreate owned PostgreSQL only if required, redact credentials without obscuring failures, stop/reap only exact owned resources.
7. Record durable per-AC1–AC15 outcomes, commands/exits/versions/fixture/source/artifact hashes, limitations and cleanup in the release evidence and full report; update relevant docs/task notes and commit explicit paths separately. Leave Done/AC closure to controller review; no full sweep, push, merge or new feature scope.
8. Whole-branch final fix wave (one combined dispatch and one scoped re-review): bound existing Chatbook recovery to one active write plus the latest pending snapshot while preserving flush, original base, scope, failure/retry and late-ACK semantics; expose existing raw-source versus note-container import through the incumbent Chatbook and web controls with supported extensions and exact-source/metadata tests; disable web Format for identity-current non-formatable results without stale-result veto; display logical CR/CRLF diagnostic coordinates without normalizing source. Use existing ADR194/065-069 and current save/import/recovery owners, targeted actual coordinator/UI/held-storage/native guards and scoped static/security/diff checks. Retain all originals and append final-wave evidence; no new transport/store/dependency, full sweep, deferred durable sync work or release/task closure. Root owns review.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
I5 qualification and scoped fixes submitted for independent review; keep In Progress and AC/DoD unchecked. Web source deep-links await existing scoped recovery hydration; empty/newer drafts survive navigation/restart. Chatbook preserves omitted language, negotiates explicit language writes, and validates current provider generation/client and exact advancing source/mode ACK; diagnostics retain truthful pending/no-op status. Native YAML/JSONL history and real SQLite/API/portable/private-refusal/PostgreSQL paths qualified, with original REDs and fixture-contract corrections retained separately. Targeted tests only; current clean owner/new-file checks plus exact inherited Ruff/Bandit/mypy and edited-region formatter comparisons are recorded. Existing ADR194/065–069 apply; no new architecture/dependency/transport. Full per-AC1–15 evidence: Chatbook Docs/superpowers/reviews/2026-09-27-structured-document-release-evidence.md and programme task-15-report.md/task-15-evidence/manifest.json. Owned Notes zero, process groups/ports and exact PostgreSQL container absent. Physical OS IME/clipboard/TTY/screenreader/hardware, deployed historical incapable server, and unexercised installed-extension profiles remain incomplete; all-four-file Library mypy retains389 inherited errors after minimal comment barrier correction. Root owns final review and closure.

Pre-review historical-peer follow-up: exact baseline d22529db9fcf8c0a8f8116531f6d4fe09ba4bfd6 ran as an owned loopback API; all2411 imported module files match baseline Git blobs. Current Chatbook service/client passed27 actual requests: real capability404, absent language projection, explicit null/known/future create/update no-write refusal, and exact ordinary omitted source writes with trailing spaces/newline and a&b <tag>. No product defect reproduced or production/test edit made. Owned Notes zero; exact process group and3196 absent. Additive evidence/report: task-15-old-peer-followup-report.md and task-15-old-peer-followup-evidence/manifest.json; tracked Chatbook review2026-09-27-structured-document-old-peer-followup.md. Original535 artifacts/report remain immutable. Initial and two earlier C helpers consulted default config before isolation; exact config/dictionary metadata proves pre-existing/no ordinary update during those probes, with no pre-probe content hash. Raw initial stderr stays private; final run binds task config/data before imports. Physical/platform gates remain incomplete; Root owns review/closure, all criteria remain unchecked.

Formal fix1 qualification: isolated current REST5 and JSON-items portable5 profiles passed. Actual runtime-provider/server Notes scope create/read/versioned update passed YAML/JSON/JSONL; this is service admission, not a native server Library editor. Real clear-sync enrollment refused HTTP412 without required at-rest attestation; no sync dataset/envelope created. Shipped app lacks durable clear receiver/mirror binding; Manual Sync reports local apply store unavailable, exact source/mode and encrypted pending entry persist after refusal/restart. Same-note real-store cross-app clear-sync/native server editor qualification remains open pending human scope disposition. No production/test/security/transport change. Corrected final accepted evidence links and exact C export AST/source equivalence; original535/historical108 copies remain unchanged. Full additive report/evidence: programme task-15-report.md fix1 appendix and task-15-fix1-evidence/MANIFEST.json; tracked release-evidence fix1 appendix. Owned Notes zero and processes/ports stopped. Existing physical IME/screenreader requirements, static debt and coverage limits remain explicit; all AC/DoD unchecked, Root owns review/closure.
Human scope disposition on 2026-10-12: missing durable server-readable Notes sync is excluded from the linter/formatter implementation and tracked as a separate Notes integration follow-up. See the Scope disposition section in Chatbook Docs/superpowers/reviews/2026-09-27-structured-document-release-evidence.md. Actual unavailable mirror/receiver and security-enrollment refusal remain the observed outcomes, without a completed sync claim. No existing native IME/screen-reader test setup is available; OS composition and spoken announcements remain unverified and no qualification waiver was granted. Whole-branch source review may proceed on the implemented candidate while this task and native-history qualification remain In Progress; no release, merge or Done claim follows.
Single whole-branch final fix wave: actual import modal exposes exact-source versus default note-container reading, re-previews retained file bytes under a current-generation guard, blocks submit during preparation and admits YAML/YML/JSONL; current refused/incomplete checks disable Format with inline feedback while stale results cannot veto newer drafts. Chatbook peer bounds recovery and fixes logical CR/CRLF Problems labels. Actual manager/input and rendered editor controls, affected source/history regressions, advertised typecheck and immutable-base lint/static evidence are in programme final-fix-report.md/final-fix-evidence. Existing ADR065-069/Chatbook194 govern; guide updated. Keep In Progress/criteria unchecked: no native physical IME/screen-reader setup or waiver, durable sync separately deferred by human. Root owns scoped re-review; no push/merge/task closure.
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
