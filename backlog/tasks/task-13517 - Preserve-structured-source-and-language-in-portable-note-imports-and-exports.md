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
updated_date: 2026-10-07 10:44
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
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented exact source downloads and tldw.note.v1 exports/imports under ADR065 and Rulings35/36. Existing Notes import schema/endpoint validates declared portable batches before mutation and carries omitted/null/opaque language through compound Sync capture, fingerprints, replay and optimistic overwrite. Bulk export requests complete canonical limit/offset pages with keywords/timestamps and refuses preview-only rows or unknown mode baselines. Targeted API/SQLite/Sync and U tests plus whole-W typecheck pass; immutable-base compatibility and scoped lint/security/formatter evidence is in .superpowers/sdd/2026-09-27-structured-document-editing/task-6-report.md in the companion Chatbook checkout. In Progress pending independent review; no controls or new service/dependency.
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
