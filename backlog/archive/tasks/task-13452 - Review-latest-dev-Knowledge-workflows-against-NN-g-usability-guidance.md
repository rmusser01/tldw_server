---
id: TASK-13452
title: Review latest dev Knowledge workflows against NN/g usability guidance
status: Done
labels:
- ux
- knowledge
- review
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Perform an independent design and browser evidence review of /knowledge in the latest dev WebUI and extension. Cover first-time and power-user capture, single and batch ingestion, single and multi-item review, content creation/review, and research. Deliver an evidenced, prioritized issue/solution list and walkthrough without changing product behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pin fetched origin/dev revision and isolate the review runtime and sample data.
- [x] #2 Complete independent design and browser assessments with NN/g heuristic mapping and explicit evidence limitations.
- [x] #3 Deliver current workflow instructions, prioritized issues, concrete solutions, acceptance criteria and a validation plan.
- [x] #4 Persist the review and critique snapshot; record verification, cleanup and final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review base: 75ab224081bf140ef52017c1a9b0a04f6878d488 (origin/dev fetched 2026-10-04). User authorized two independent reviewers. Plan: 1. establish isolated latest-dev runtime; 2. independently inspect workflows and states; 3. synthesize an evidenced NN/g audit and walkthrough; 4. persist, verify and stop owned services. Backlog MCP searches timed out; using repository backlog-py canonical CLI. ADR assessment: review proposes no adopted architecture change; no new ADR required.
Latest-dev runtime ready: WebUI localhost:18922, backend 127.0.0.1:18921, local deterministic inference 127.0.0.1:18923. Reused installed dependency directories without installing new packages. Disposable public-API fixture seed created Media IDs 1/2 and one scoped Note; no user library content touched. Both independent reviewers completed empty-state inspection and are inspecting populated states. ADR-007 governs the canonical research workspace; this review does not adopt a new architecture decision. Runtime cleanup will signal only the recorded supervisor process tree.
Artifacts: Docs/Reviews/KNOWLEDGE_NNG_UX_REVIEW_2026_10_04.md; Docs/Reviews/assets/2026-10-04-knowledge-nng-ux-review/; .impeccable/critique/2026-10-05T00-00-40Z__apps-packages-ui-src-routes-option-knowledge-tsx.md. Primary issue ledger contains 15 findings; five priorities include four P1 groups and one P2 group. Both reviewers independently reproduced the preset scope reset and note-to-research source loss. One real disposable TXT upload completed storage, chunking, retrieval and source preview; a derived Knowledge export was saved and reopened in Notes. Native extension built successfully and setup/options/Cmd+K/narrow layout/companion-chat entry were checked.

Stage 1: Latest-dev isolation — Complete. Goal: pin current source and isolate data. Success: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488 rechecked; synthetic public-API fixtures and isolated runtime ready. Verification: server health and WebUI rendered.
Stage 2: Independent inspection — Complete. Goal: prevent reviewer anchoring. Success: A finished before B findings entered synthesis. Verification: independent browser observations, detector exit 0 with no findings across 66 files, native extension build exit 0.
Stage 3: Synthesis and walkthrough — Complete. Goal: actionable NN/g review. Success: current flows, 10 heuristic scores, 15 evidenced issues, fixes and acceptance checks, persona walkthroughs, opportunities and validation plan. Verification: report contains 15 ledger rows, 10 heuristic scores totaling 22/40, and 36 valid local source/evidence links.
Stage 4: Persistence and cleanup — Complete. Goal: durable report and clean isolated checkout. Success: report/evidence and critique snapshot saved; first-run trend read; supervisor exited and ports 18921/18922/18923 closed; all native Chromium contexts and reviewer IAB tab closed; four owned dependency links and owned Next cache removed. Verification: runtime descriptor stopped and TCP probes closed.

Verification: JSON evidence parses; report source links exist with valid line numbers; report whitespace/newline checks pass; canonical Backlog normalization and Git whitespace checks are final gates. Bandit was attempted on explicit Markdown files and encountered unsupported Markdown parse errors; directory-scoped artifact scan then completed with 0 errors/findings and 0 Python lines. No product Python was changed and no application security coverage is claimed. Product tests were not rerun for this documentation-only audit; the UI journeys and native build are recorded above.
Known limits: one item processed, two-item queue/review only; failed-batch retries and content-review commit not executed; live web/model quality, native website capture/permissions, transcription, real HTTP-error recovery and large-library runtime remain unverified. Cold embedding startup and development comparison/migration notices are excluded as product defects. ADR assessment: no new architecture rule; preserve ADR-007 research workspace. No product behavior changed.
Final gates passed: pre-commit on all eight review/task artifacts, including formatting, private-key/conflict guards, canonical Backlog format and both always-run repository guards. Snapshot whitespace normalized; trend remains first baseline 22/40. Stopped disposable runtime/Chromium profiles, generated extension output, collected uploads and temporary snapshot/report scripts were removed after evidence preservation. Retained only durable report, five evidence files, critique snapshot and task record. No application security/test coverage claim is made for the non-code audit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed latest-dev dual-review Knowledge walkthrough and NN/g audit at 75ab224. Saved full report, evidence and critique baseline (22/40), with 15 issue/solution/acceptance entries and five priorities. Scope continuity, research provenance, accessible source selection, onboarding and recovery lead the recommendations. Disposable runtime stopped and owned setup artifacts cleaned; documentation-only verification recorded.
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
