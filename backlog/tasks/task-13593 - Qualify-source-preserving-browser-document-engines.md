---
id: TASK-13593
title: Qualify source-preserving browser document engines
status: Done
created_date: 2026-10-07 03:09
dependencies:
- TASK-13586
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-07 13:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide local strict YAML, JSON and JSONL checks and reversible formatting without numerical or lexical corruption.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Supported shared fixtures agree with Python validity/categories and scalar-offset diagnostics; tolerant parse output alone never proves validity.
- [x] #2 JSON and JSONL formatting preserves original numbers/strings, object order, decoded duplicate-key diagnostics, physical records/newlines and idempotence.
- [x] #3 Qualified YAML formatting preserves comments, directives, markers, anchors, aliases, merges, scalar tokens/styles and semantic graph for nontrivial streams; unsupported constructs are explicitly refused.
- [x] #4 UTF-16 positions convert correctly to Unicode scalar offsets and limits/source-safe failure categories are tested on real parser output.
- [x] #5 Candidate versions/licenses and fixture evidence precede adoption; targeted tests, scoped type/static checks and web/extension loading controls pass.
- [x] #6 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md (existing programme decision)
Reason: qualify browser-local source-preserving engines and new runtime dependency/tooling choices under the approved cross-runtime language contract; amend the existing ADR before accepted dependency adoption, without a UI/save/worker authority change.
1. Read task9 brief/binding constraints/context, applicable S/U/W/extension guidance and structured-notes lessons, ADR070, actual E1 TS constructors/shared fixtures and the reviewed Python public contract/qualification outcomes. Immutable S base bf0965412cb65b6fbfee3b31e5e211dbd2872841; C reviewed final engine ba744f9059d plus closure3ffaf527f304.
2. Trace installed/native parser facilities before candidates, then add meaningful RED strict precision/lexeme/decoded-duplicate/JSONL/Unicode-position and YAML source/graph/version controls, including the independently confirmed Core decimal-only sign boundary.
3. Qualify strict source scanner/visitor/text-edit JSON and physical-record JSONL with no resolved JavaScript object values as formatting source. Enforce depth during parsing, exact token equality, record/newline preservation and scalar-offset diagnostics.
4. Qualify source-aware YAML APIs for nontrivial comments/multi-document/anchors/merges/block-scalars plus explicit unsupported refusal; compare shared validity/categories with the reviewed Python table without modifying fixture expectations. Record real package version/license/source invariants before accepting/pinning dependencies and lock updates; amend ADR070 before adoption.
5. Run explicit contract/JSON/YAML/position selections and scoped lint/format/type checks, whole-W typecheck after shared edits, and actual Next/extension module-loading controls. No full sweep, runtime Prettier, editor/worker adoption, CDN, shared dependency rebind or user data/config.
6. Write durable browser qualification report and full task-9-report.md with exact RED/GREEN/final commands/output/process exits, source/fixture hash, supported/refused constructs, resource/import/bundler observations and precise limits. Self-review and commit explicit unit paths; remain In Progress for fresh controller review.
Fix round1/5 after independent E3 review: ImportantI1 repeated YAML merge declarations bypass the per-map key set; ImportantI2 explicit merge-tag scalar spellings bypass qualification. ADR070 already governs duplicate/qualified-value refusal; no new runtime/authority decision. Exact fix base9d50710a37cab6bd759b0b55b6762a4c5f64c7d2; C unchanged3ffaf527f304. Before production changes, reproduce both holes with meaningful RED in the existing whole-YAML test owner, retain valid single/sequence merges, explicit overrides and valid explicit merge tags, correct each shared TS admission branch once. Run whole amended YAML selection and relevant scoped strict/whole-W types, lint/formatter/diff, refresh the captured final-engine artifact and focused reproduced cases/positive controls. Preserve unchanged JSON/position/contract/build provenance without reruns solely to regenerate it; any new loading concern gets a named targeted build. Append exact RED/GREEN/commands/output/exits, amended artifact provenance and historical-vs-current qualification to the bounded report/durable record; self-review, explicit commits, remain In Progress for fresh scoped re-review. Deferred warning M1 is not an unrelated tooling fix.
Additional fix1 scope under existingAC3/ADR070: actual9d captured-engine probe confirms valid four-space explicit-tagged mapping key checks complete/format-allowed but fails/no replacement during reindent because leading() sees the tag before key offset and leaves that line unchanged while siblings move. Controller Ruling38 authorizes only a small CST property-prefix indentation correction for block mapping keys, preserving inline-key/value/flow/block-scalar boundaries; no general formatter rewrite. Keep the three meaningful actual-reindent positives, add adjacent tag/anchor and inline/no-op preservation controls as needed, retain whole-YAML and final captured-bundle/strict/static/whole-W evidence. Record the actual base probe and new branch risks before scoped re-review.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented pure request-aware JSON/JSONL and source-aware YAML checks and reversible formatting in four shared UI modules, with three focused test files. Adopted exact jsonc-parser 3.3.1 (MIT) and yaml 2.8.1 (ISC) only after isolated fixture/license/ADR/bundler qualification; late self-review version-inheritance and mantissa-less YAML 1.1 refusals were qualified and added to ADR-070 before final acceptance. No UI/save/worker authority changed.

Evidence: final 142 targeted contract/JSON/YAML/position tests passed; strict scoped types, whole-W typecheck with the prescribed 8 GiB heap, scoped ESLint (zero warnings), Prettier and frozen Bun install passed. Actual default Next 16.1.4 Turbopack and WXT 0.20.13/Vite 5.4.21 imported-engine builds executed in fresh Chromium contexts; WXT bundle used extension-like CSP, not installed-extension lifecycle. All 37 unchanged shared fixture validity/category sets and admission agree with reviewed Python ba744f9059d; parser-specific span widths/multiplicity are disclosed. Fixture SHA-256 fcf516e4c0bb8ff7a19748a4d5f7deb8a30ad0078ea0ce2af0e1e81af50fb0d7.

Limits: pure synchronous engine has cooperative two-second checks, 1 MiB UTF-8 source, depth 100 and diagnostic cap 100, but no hard termination or heap ceiling (large synthetic YAML reached about 318 MB whole-process RSS). Unsupported YAML profiles, including candidate 1.1 mantissa-less floats, are explicit; Python accepts bare 1.1 dot as a string. Later worker/lifecycle qualification remains required. No full suite; Bandit not applicable because no Python production source changed. Immutable base bf0965412cb65b6fbfee3b31e5e211dbd2872841.

Paths: apps/packages/ui/src/services/document-editing/{engine,json-document,yaml-document,source-positions}.ts and __tests__/{json-document,yaml-document,source-positions}.test.ts; apps/packages/ui/package.json; apps/bun.lock; Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md; backlog/docs/lessons-structured-notes.md. Durable qualification: Docs/superpowers/reviews/2026-09-27-document-engine-browser.md. Complete commands/output: Chatbook plan workspace task-9-report.md, raw captures /private/tmp/tldw-task9-evidence. Self-review completed; remains In Progress for fresh controller review.
Review fix round 1/5 closes Important I1/I2 from base 9d50710a37cab6bd759b0b55b6762a4c5f64c7d2: merge declarations now enter the same map-local duplicate set; explicit merge tags must match qualified scalar source before target handling. Single/sequence merges, explicit imported-member overrides and sibling scopes remain supported. Ruling38 additionally authorized one shared CST property-prefix whitespace correction for tagged/anchored block mapping keys after real positive reindent controls exposed the pre-existing preservation failure. ADR070 governs these corrections; no dependency/runtime/ownership boundary changed.

Evidence: I1 RED 2 failed/74 passed -> 76 passed; I2 RED 7 failed/77 passed (including three layout failures), then 3 failed/81 passed; expanded layout RED 5 failed/86 passed -> final whole YAML 91 passed. Strict scoped types, whole-W 8 GiB typecheck, changed-source ESLint/Prettier and diff checks pass. Final browser-target engine rebuilt (85 modules/no externals) and 15 exact-hole/positive/layout bundle cases pass; 260791-byte artifact SHA256 2ebc823bdccb659abe3b00d003733a9539ff594d18c83d0a1f49254ae546c1c2. Original bundle/captures retained; current engine-final.mjs points to amended artifact. Frozen fixture SHA unchanged. No unchanged JSON/positions/contracts, benchmarks, Next/WXT builds or old browser controls rerun; their existing 9d50710a37 provenance is historical, not new amended-source qualification.

Paths: yaml-document.ts and its existing YAML test owner; durable browser qualification report; structured-notes lessons; this task. Bounded fix evidence appended to Chatbook plan task-9-report.md; exact commands/output/exits/source/artifact hashes under /private/tmp/tldw-task9-evidence/fix1. Self-review complete; remaining cooperative timing/heap/lifecycle limits and deferred inherited warning M1 unchanged. Bandit N/A, no Python production edits. Keep In Progress for fresh scoped controller review.
Controller closure: fresh independent scoped review over 9d50710a37..4f1f7356c6e3bb69efae92e9361cf8044c1e2a0c is clean: Important I1/I2 addressed, authorized Ruling38 key-property indentation scope verified, zero new Critical/Important/Minor findings. Full review/report and retained command/output/exit/hash records were read. All five initial Cannot Verify groups are explicitly dispositioned in the plan workspace scope-resolution record. Current evidence is whole YAML 91 plus 15 amended browser-target bundle controls in Node and scoped/whole-W types/static checks; original 142 and real Next/WXT Chromium loading are historical 9d evidence, not an unexecuted combined 164 or a new real-browser run. ADR070, durable browser qualification and incident lesson updated. No Python touched: Bandit N/A. E4 owns actual termination/lease/debounce/cleanup; I2/I3/I5 own mounted editor/history/release gates. Synthetic ~318 MB is current whole-Node-process RSS, neither peak nor heap cap. Deferred original M1 warning noise is retained for final review; no remaining E3 blocking findings. All six AC and six repo DoD requirements satisfied within this unit's bounded scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pure browser JSON/JSONL/YAML engines qualified; merge duplicate/tag-value and scoped tagged-key indentation fixes independently reviewed clean. Current 91 YAML/15 bundle controls, original 142/Next/WXT evidence and exact static/type exits recorded. All AC/DoD complete; hard worker and mounted editor/release gates remain with E4/I2/I3/I5; M1 warnings deferred.
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

Originally TASK-13518 in the reviewed structured-document candidate. The publication task is TASK-13593 because current dev assigns the former ID to unrelated work. This changes tracking identity only; acceptance criteria, status, original creation date and implementation evidence are retained. Original reviewed commits and local evidence preserve the historical ID.
