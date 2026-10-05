---
id: TASK-13453
title: Remediate chat UX across WebUI and extension in five stages
status: In Progress
labels:
- chat
- ux
- webui
- extension
- nng
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Coordinate the complete latest-dev NN/G chat audit remediation for first-time, returning, power, roleplay, and assistive-technology users equally. Cover all ten finding groups and named enhancements through five reviewable stages without duplicating existing lifecycle work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Written design maps every audit finding and enhancement to a stage, concrete behavior, and validation evidence; user review is recorded before implementation.
- [ ] #2 Stages TASK-13454 through TASK-13458 satisfy their acceptance criteria on the then-current dev baseline; no finding is silently dropped because of an existing PR or backend dependency.
- [ ] #3 Shared WebUI, full-page extension, and compact sidepanel flows are verified with acknowledged persistence, owner-bound continuation, actionable recovery, and accessible responsive interaction.
- [ ] #4 Final review records remaining capability limits, unrun human-study checks, verification results, and NN/G before/after findings accurately.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Written-design approval and the requested critical review are recorded on 2026-10-04; application implementation has not begun. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. Evidence: prior audit commit 942efa7da9 on codex/chat-nng-ux-audit-20261004, report in the attached chat-nng-ux-audit worktree. Overlapping PRs: #3147 honest saving labels, #3152 sidepanel tab integrity, #3178 history content search; #3165 D7 is proposed and is not assumed approved. Highest task ID checked across dev and all open PRs: 13452; this program reserves 13453-13458.
Design written and self-reviewed: all ten findings plus enhancements mapped to five stages, no placeholders, local links verified, and stage dependencies avoid blocking accessibility on the conditional backend roleplay contract. Scope and written-design approval are recorded; the requested critical review corrections govern subsequent planning and application edits. Also governed by backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md. No new durable architecture rule is proposed. Current changes are documentation/task records only; Bandit is not applicable to this non-code scope.
Verification 2026-10-04: backlog-py normalize --check passed for all six new records; design scan confirmed five stages, all ten finding groups, six task links, existing local references, and no placeholders; git diff --cached --check passed after normalizing Markdown line breaks. Immediately refreshed origin/dev and confirmed the baseline remains 75ab224081bf140ef52017c1a9b0a04f6878d488. No application changes or application test/build claims in this design-only commit.
Written design approved by the user on 2026-10-04, followed by the requested pre-implementation critical review. Two independent reviewers and source checks identified unknown-create recovery, greeting-only creation, full preflight, settings acknowledgement/scope, outage-safe history deduplication, exact handoff destinations/fields, and focus/source-draft preservation. Corrections are incorporated in Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. All ten findings and enhancements remain in the five stages; user approval does not close application acceptance criteria. Fresh origin/dev remains 75ab224081bf140ef52017c1a9b0a04f6878d488. This checkpoint changes design/task records only; no runtime or application test/build claims, and Bandit is not applicable to non-code changes. Detailed execution planning must use these corrected contracts.
Review verification: backlog-py normalize --check passed for TASK-13453 through TASK-13458; the corrected design retains five stages and all ten finding groups, local links exist, approval is recorded, pending-approval text is removed, and git diff --check passed. Source-derived failure scenarios remain explicitly unverified at runtime. No application files were changed or application tests run for this review checkpoint.
Latest-dev refresh before the draft retention unit: origin/dev advanced from 75ab224081 to ba553fdc51a16ba04330754671f0c2ee3398e17f via Claims metrics PR 3093. No apps/packages/ui changes in the incoming range. Integrating dev into the isolated review branch to preserve the requested baseline; no PR or push. Character preflight unit committed in d2763dfe92 with 165 passing affected tests, typecheck and no added lint warnings; its broader Stage 1 gates remain open.
Stage1 application work is in progress with reviewed incremental draft/owner-transition deliveries. Current latest dev c2260945ee integrated10d6abff30. PR3165 now merges accepted D7 owner decisions: native server identity for new saved chats, explicit legacy import, read-only offline, local Temporary retention through reload until closed/cleared, Sync v2 in scope. This supersedes prior D7-pending notes and Temporary transcript exclusion target; associated backend contracts and full-surface retention remain open. Program design updated; no broader stage completion claimed. Retained-draft recovery checkpoint422tests/typecheck/no added touched lint, two independent source reviews clear, Bandit N/A TS/docs.
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
