---
id: TASK-2278
title: Plan Knowledge QA follow-on trust remediation
status: In Progress
labels:
- knowledge
- ux
- planning
- webui
- extension
priority: high
documentation:
- Docs/superpowers/specs/2026-06-07-knowledge-qa-follow-on-trust-remediation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a follow-on staged remediation design for /knowledge after TASK-528, based on live WebUI and extension QA findings. Scope remains Knowledge QA only: no flashcards, decks, spaced repetition, or study-set behavior. The plan must build from latest origin/dev and focus on citation trust, evidence materialization, extension reliability, scoped-search reliability, export/history recovery, and live UAT gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design treats TASK-528 and the latest origin/dev state as baseline instead of duplicating completed readiness/empty-state work.
- [x] #2 Design preserves /knowledge as a personal-library QA workflow and explicitly keeps flashcards out of scope.
- [x] #3 Design proposes staged follow-on remediation plans for citation trust, evidence source previews, extension setup/sync reliability, scoped search, export/history recovery, live UAT, and longer-term evidence workflow improvements.
- [x] #4 Design identifies dependencies, affected surfaces, acceptance criteria, and verification strategy for WebUI, extension, backend, and docs.
- [ ] #5 Written spec is reviewed and approved before implementation planning begins.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-07: Drafted follow-on Knowledge QA trust remediation design against rebased origin/dev baseline 922f61459c. Review pass tightened canonical trust states with unknown_trust, bounded stages 1-6 citation relevance, export behavior for degraded/unknown answers, and Stage 7 non-blocking scope. Verification so far: git diff --check passed; ASCII scan returned no matches. Awaiting user approval before implementation planning.
2026-06-07 review pass 2: Found and addressed spec handoff risks before implementation planning: web fallback origin was not first-class, trust-state normalization could diverge across surfaces, Stage 2 citation-jump ownership was ambiguous, live-vs-mocked UAT fixture boundaries needed clarification, and the TASK-528 WXT extension runtime E2E blocker needed explicit release-gate treatment. Verification after patch: git diff --check passed; ASCII scan returned no matches.
2026-06-07: Rebased reviewed spec commit onto latest origin/dev 0efb6778b8 after origin/dev advanced again; updated spec provenance accordingly before continuing.
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
