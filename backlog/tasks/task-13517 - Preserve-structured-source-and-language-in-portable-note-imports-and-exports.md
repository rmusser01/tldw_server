---
id: TASK-13517
title: Preserve structured source and language in portable note imports and exports
status: In Progress
created_date: 2026-10-07 03:08
dependencies:
- TASK-13513
- TASK-13515
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-07 11:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users move source files and portable notes without losing document text or language.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Raw source exports retain exact text with a matching mode extension; portable notes retain source as a string plus language and metadata.
- [ ] #2 Versioned Chatbook/portable envelopes preserve Canvas and structured Notes together, with older supported readers retained.
- [ ] #3 Unknown archive/envelope versions fail before storage mutation; recognized source extensions carry mode without sniffing document text.
- [ ] #4 Targeted source/envelope/import/transaction tests verify exact-source round trips, atomic rejection and scoped static evidence.
- [ ] #5 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md (existing programme decision)
Reason: implement approved portable source/language archive and import service contracts; amend the existing decision before a new contract detail, with no new storage or transport owner.
1. Read task6 brief, owner preparation/Ruling29, current F5 reviewed exports, applicable instructions/lessons and governing ADR. Recheck archive versions and actual import/API/fingerprint owners.
2. Add meaningful failing exact raw source, portable envelope/language, Canvas-plus-Notes archive, old-reader/unknown-version and approval-fingerprint/retry controls.
3. Extend versioned archive/note records and actual source-aware import planning/fingerprint/execution; retain supported v1-v3 readers and preserve Canvas when selecting the verified new version.
4. Extend existing server Notes import schema/normalizer/endpoint and U types/hooks/export builders with explicit raw-source versus portable/legacy contracts, no source-language sniffing or synthetic empty content. Qualify whole versioned-batch refusal before mutation and truthful older import endpoint refusal while preserving ordinary legacy behavior and omitted/null semantics.
5. Run targeted C source/archive/Canvas/transaction/receipt tests and real S API serialization/materialization plus affected U import/export tests; whole W typecheck after shared edits and scoped immutable-base static/security/formatter checks. No full sweep, new service or editor adoption.
6. Update relevant docs/ADR and concise task evidence, self-review, append exact RED/GREEN/final commands/output and boundaries to task-6-report.md, commit explicit C/S unit paths separately; keep In Progress for controller independent review.
Rulings35/36 (task-6-contract-rulings.md and progress.md): exact portable format_version=tldw.note.v1 with portable_note/source import discriminators; explicit raw-source context defaults false for old JSON/YAML containers, JSONL may be source-only. Missing portable language is invalid; filename-less raw text uses a safe fallback title and omits inferred mode. All new Notes archives select verified v4 including null and unique stable-identity JSON carriers, retaining Canvas; conversation-only version selection remains. Amend governing ADR194/065 before these contract edits; preserve old readers, legacy omitted-field identity fingerprint and actual import path/authority validation.
Fix round1 (task-6-review.md / task-6-fix1-findings.md): retain exactly the declared portable carrier/record set qualified by preflight through execution, preserving legacy container behavior; correct opaque-language source extension lookup to recognize own known modes only. Add real API mixed-carrier/earlier-valid-record and prototype-name source controls, then rerun the covering S/U tests and whole-W typecheck with actual exit status. Compare scoped static/security/format checks to previous reviewed head bebc8889866f99dcf003b10b36d3aa8cb7dee32e, update exact evidence prose, append fix report and commit explicit unit files. C remains unchanged; tasks stay In Progress until scoped re-review. Existing ADR065 governs these contract corrections; no new boundary or dependency.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented exact source downloads and tldw.note.v1 exports/imports under ADR065 and Rulings35/36. Existing Notes import schema/endpoint validates declared portable batches before mutation and carries omitted/null/opaque language through compound Sync capture, fingerprints, replay and optimistic overwrite. Bulk export requests complete canonical limit/offset pages with keywords/timestamps and refuses preview-only rows or unknown mode baselines. Targeted API/SQLite/Sync and U tests plus whole-W typecheck pass; immutable-base compatibility and scoped lint/security/formatter evidence is in .superpowers/sdd/2026-09-27-structured-document-editing/task-6-report.md in the companion Chatbook checkout. In Progress pending independent review; no controls or new service/dependency.
Fix round1: root portable envelopes now take precedence over wrapper-shaped extra fields in the existing JSON extractor, and import execution reuses the exact preflight row set. Browser preview follows the same root precedence; source extension lookup accepts only own known modes while preserving opaque mode strings in portable output. Real API mixed-carrier controls passed across four wrappers and both discriminators with a preceding valid item; prototype-name and actual import-preview controls passed. Final targeted covering: S33 passed, U38 passed, whole-W typecheck exit0; reviewed-base scoped static/security and changed-region formatting show no new debt. ADR065 remains governing; tasks remain In Progress for independent re-review. Full fix evidence: task-6-report.md.
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
