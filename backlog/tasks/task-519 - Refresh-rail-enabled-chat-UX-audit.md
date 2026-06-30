---
id: TASK-519
title: Refresh rail-enabled chat UX audit
status: Done
labels:
- UX
- chat
- audit
references:
- Docs/superpowers/plans/2026-05-27-chat-rails-ux-rebaseline-implementation-plan.md
documentation:
- Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md
- Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png
modified_files:
- Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md
- Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json
- Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refresh prior /chat findings against the captured rail-enabled origin/dev-based page and classify each as fixed, still active, not reproduced, or not evaluated with evidence.
- [x] #2 Capture and document sidepanel chat handoff evidence, including the full-screen /chat route contract and any live-debug-route limitations.
- [x] #3 Add severity-ranked findings, quick wins, larger improvements, ideal workflows, assumptions, non-goals, and verification notes scoped to /chat and direct sidepanel chat handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 5 refreshed the rail-enabled /chat audit after rail restoration. Added executive summary, evidence notes, first-time and power-user walkthroughs, prior finding reclassification, severity-ranked refreshed findings, quick wins, larger improvements, ideal workflows, and open questions/non-goals. The audit is time-scoped to the captured origin/dev-based branch state; local origin/dev advanced after capture and must be rebased/refreshed in the final verification task.

Captured sidepanel chat debug-route evidence from a 390x844 viewport after auth/config seeding and recorded the raw unseeded capture limitation. Expanded the refreshed findings table with observed behavior and prior-finding classification columns. Task 5 is evidence-limited: first-send/streaming/retry/save-title, prompt picker, history/sidebar, long sessions, and compare/export/share remain explicitly not revalidated because the full live cockpit suite still has four captured non-rail baseline failures.

Verification: git diff --check passed; evidence.json parses successfully; placeholder scan found no unresolved placeholder markers or empty table cells. Bandit skipped because this slice only changed docs, JSON evidence, and a screenshot artifact.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refreshed the rail-enabled /chat UX audit and direct sidepanel handoff evidence. The audit now records why the earlier no-siderails finding was provenance-related, confirms rails on the captured origin/dev-based page, captures sidepanel starter evidence, and ranks the remaining /chat issues led by character/persona continuity, plain-chat creation 422, Web search feedback, model scope discoverability, mobile first-run density, and sidepanel dashboard ambiguity.
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
