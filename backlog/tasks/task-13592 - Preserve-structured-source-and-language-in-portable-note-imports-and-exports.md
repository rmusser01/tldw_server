---
id: TASK-13592
title: Preserve structured source and language in portable note imports and exports
status: Done
created_date: 2026-10-07 03:08
dependencies:
- TASK-13588
- TASK-13590
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-07 11:26
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users move source files and portable notes without losing document text or language.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Raw source exports retain exact text with a matching mode extension; portable notes retain source as a string plus language and metadata.
- [x] #2 Versioned Chatbook/portable envelopes preserve Canvas and structured Notes together, with older supported readers retained.
- [x] #3 Unknown archive/envelope versions fail before storage mutation; recognized source extensions carry mode without sniffing document text.
- [x] #4 Targeted source/envelope/import/transaction tests verify exact-source round trips, atomic rejection and scoped static evidence.
- [x] #5 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md (existing programme decision)
Reason: implement approved portable source/language archive and import service contracts; amend the existing decision before a new contract detail, with no new storage or transport owner.
1. Read task6 brief, owner preparation/Ruling29, current F5 reviewed exports, applicable instructions/lessons and governing ADR. Recheck archive versions and actual import/API/fingerprint owners.
2. Add meaningful failing exact raw source, portable envelope/language, Canvas-plus-Notes archive, old-reader/unknown-version and approval-fingerprint/retry controls.
3. Extend versioned archive/note records and actual source-aware import planning/fingerprint/execution; retain supported v1-v3 readers and preserve Canvas when selecting the verified new version.
4. Extend existing server Notes import schema/normalizer/endpoint and U types/hooks/export builders with explicit raw-source versus portable/legacy contracts, no source-language sniffing or synthetic empty content. Qualify whole versioned-batch refusal before mutation and truthful older import endpoint refusal while preserving ordinary legacy behavior and omitted/null semantics.
5. Run targeted C source/archive/Canvas/transaction/receipt tests and real S API serialization/materialization plus affected U import/export tests; whole W typecheck after shared edits and scoped immutable-base static/security/formatter checks. No full sweep, new service or editor adoption.
6. Update relevant docs/ADR and concise task evidence, self-review, append exact RED/GREEN/final commands/output and boundaries to task-6-report.md, commit explicit C/S unit paths separately; keep In Progress for controller independent review.
Rulings35/36 (task-6-contract-rulings.md and progress.md): exact portable format_version=tldw.note.v1 with portable_note/source import discriminators; explicit raw-source context defaults false for old JSON/YAML containers, JSONL may be source-only. Missing portable language is invalid; filename-less raw text uses a safe fallback title and omits inferred mode. All new Notes archives select verified v4 including null and unique stable-identity JSON carriers, retaining Canvas; conversation-only version selection remains. Amend governing ADR194/065 before these contract edits; preserve old readers, legacy omitted-field identity fingerprint and actual import path/authority validation.
Fix round1 (task-6-review.md / task-6-fix1-findings.md): retain exactly the declared portable carrier/record set qualified by preflight through execution, preserving legacy container behavior; correct opaque-language source extension lookup to recognize own known modes only. Add real API mixed-carrier/earlier-valid-record and prototype-name source controls, then rerun the covering S/U tests and whole-W typecheck with actual exit status. Compare scoped static/security/format checks to previous reviewed head bebc8889866f99dcf003b10b36d3aa8cb7dee32e, update exact evidence prose, append fix report and commit explicit unit files. C remains unchanged; tasks stay In Progress until scoped re-review. Existing ADR070 governs these contract corrections; no new boundary or dependency.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented exact raw-source downloads and tldw.note.v1 portable imports/exports under Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md. Existing API/import/Sync owners validate the complete selected portable carrier set before mutation and execute the retained preflight rows; actual browser preview shares root precedence. Own-property mode lookup preserves unknown modes while choosing .txt. Full-row paginated exports retain metadata and refuse preview-only baselines. Final targeted S33/U38 pass and whole-W captured typecheck exit0; immutable reviewed-base static/security and changed-region formatting introduce no debt. Independent task review and fix1 scoped review approved. Lessons recorded in backlog/docs/lessons-structured-notes.md. Full evidence is companion Chatbook .superpowers/sdd/2026-09-27-structured-document-editing/task-6-report.md. Existing lint/format debt and test warnings/telemetry remain disclosed; Stage35 pre-action failures were reproduced with the base hook only, without complete-base-suite or live browser/protected-peer claims. Production controls and cross-app release qualification remain later tasks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Portable source and mode preservation complete for this unit; Sbebc888 original plus a223068 fix, S33/U38 targeted tests and whole-W typecheck0, scoped review approved. ADR070 applies. No full sweep or live deployment claim.
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

## Renumbering provenance

Originally TASK-13517 in the reviewed structured-document candidate. The publication task is TASK-13592 because current dev assigns the former ID to unrelated work. This changes tracking identity only; acceptance criteria, status, original creation date and implementation evidence are retained. Original reviewed commits and local evidence preserve the historical ID.
