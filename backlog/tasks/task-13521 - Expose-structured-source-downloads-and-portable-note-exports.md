---
id: TASK-13521
title: Expose structured source downloads and portable note exports
status: Done
created_date: 2026-10-11 22:07
dependencies:
- TASK-13517
- TASK-13520
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-12 09:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make source downloads and portable notes distinct and preserve language through their real user actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Download source returns exact source with a matching known-mode extension; Export note returns a versioned portable envelope with source/language/metadata.
- [x] #2 Real action wiring uses the reviewed import/export builders and scoped authority without passing structured source through Markdown conversion.
- [x] #3 User guidance explains check/save separation, supported/refused constructs, limits and undo behavior; targeted action/static and source-byte controls are recorded.
- [x] #4 Relevant ADR assessment, scoped static/security evidence and completion documentation are recorded without hiding unfinished release gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md; 066-local-monaco-asset-owner.md; 067-notes-temporary-language-mode.md; 068-exact-source-secondary-note-capture.md; 069-native-monaco-bom-source-adapter.md (existing).
Reason: direct action wiring and guidance consume accepted portable/source/currentness/runtime boundaries; no new storage/schema/provider/transport contract.
1. Read Task13521, task14 brief/context, final reviewed F6/I2/I3 contracts/reports and applicable AGENTS/ADR/lessons. Capture exact canonical source/mode/presence through the existing owner; effective temporary views must never be implicitly portable.
2. Add meaningful failing action tests for exact JSON source download bytes and known-mode extension versus the existing tldw.note.v1 envelope; test current source, unknown/absent/null mode and authority/refusal behavior through actual helpers.
3. Wire distinct Source download and Export note labels/extensions through existing Notes header/export owners, preserving explicit Markdown conversion and download/export authority. Reuse existing archive/import validators with unsupported-version preflight before effects.
4. Extend Docs/User_Guides/WebUI_Extension/Notes_Structured_Source.md (controller Ruling54), rather than create a duplicate guide under nonexistent Docs/User_Guide. Explain refusal/limits/undo, temporary old-peer choices, unchanged save behavior, empty recovery and valid YAML versus intended structure; no schema/block-lint claims.
5. Run named action and focused F6 round-trip controls, actual owned browser/file-dialog byte checks, scoped types/lint/actual-config changed-region formatter checks bound to immutable bases. Reuse corrected owned-note/profile/output fixtures and normal authenticated hosts; no full sweep, private setup assumption or hidden platform gaps.
6. Record exact commands/exits/raw source-bound outputs and resource cleanup, assess Bandit applicability, self-review and explicitly commit S unit paths separately from C. Leave In Progress/unchecked until controller review; no nested agents.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented distinct Download source and Export note (.note.json) header actions through the existing canonical editor capture and F6 export helpers. Current source/canonical mode are read synchronously, old-peer temporary intent stays absent, and explicit null/future modes retain presence. Existing download guards and explicit conversions remain; the mobile overflow menu is bounded. Extended the existing Notes_Structured_Source guide per Ruling54, with no duplicate guide/schema/dependency/provider. Existing ADR065-069 govern; no new ADR is required.
Verification: action/F6/owner 55 passed; real authenticated Chromium2 passed with ordinary/fenced source bytes, desktop/mobile portable downloads and shipped CSP; SQLite/API import pre-effect27 passed (203 unsuppressed warnings). Advertised8GiB typecheck passes after recorded default-heap OOM; strict new source and edited-region formatter checks pass; BASE146->current146 legacy lint, zero introduced findings. No Python production changes, so server Bandit is inapplicable. Owned Notes zero/processes absent. Full report/index: Chatbook programme task-14-report.md/task-14-evidence. Physical/other-browser/installed-extension/history-peer profiles remain unqualified. Self-review complete; keep In Progress and all AC/DoD unchecked for Root review.
Independent Task14 review: Spec compliant / Approved, no Critical or Important findings. Root verified all543 durable artifact hashes and18 unit source hashes; additive formatter-label clarification preserves original records and verifies six actual base/head source pairs. Scope reconciliation: task-14-review-scope-resolution.md. Existing ADR194/065-069 apply; no new ADR. Non-pristine warnings, inherited native mypy barrier and stale writer description remain I5/final triage. Six inherited save scenarios, cross-application and physical/deployment release gates remain unfinished; this closes only the verified source/portable actions and guidance.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Distinct exact-source and portable-note actions use current canonical captures, preserve mode presence and pass authenticated desktop/mobile byte controls. Independent review approved; original warnings and unfinished cross-application/physical/deployment gates remain explicit.
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
