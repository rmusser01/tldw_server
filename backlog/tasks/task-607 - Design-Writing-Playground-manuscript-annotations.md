---
id: TASK-607
title: Design Writing Playground manuscript annotations
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-23 07:02'
labels:
- design
- webui
- extension
- writing-playground
- manuscripts
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-24-writing-playground-manuscript-annotations-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an approved design spec for durable manuscript-owned Writing Playground annotations and comments. Scope covers scene range comments, chapter/project notes, manual and AI-authored annotations, backend-owned persistence/API, selected-text synchronous AI critique, Jobs-backed scene review, WebUI/extension UI behavior, anchoring/drift handling, testing, and rollout boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec captures durable manuscript-owned Writing Playground annotations for scene range comments and chapter/project notes.
- [x] #2 Spec covers manual and AI-authored annotation flows, including synchronous selected-text critique and Jobs-backed scene review.
- [x] #3 Spec defines backend API/data model, anchoring/drift handling, saved-scene gating, permissions, error handling, and testing strategy.
- [x] #4 Spec review loop is completed and results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Draft spec created at Docs/superpowers/specs/2026-05-24-writing-playground-manuscript-annotations-design.md. Pre-review checks passed: git diff --check passed; ASCII scan returned no matches; placeholder scan returned no matches.

Spec review loop: reviewer returned Approved with no issues. Advisory notes for implementation planning: resolve cap/default/provider/anchor-persistence decisions; keep anchor_status enum explicit as attached/reattached/needs_review/scene_level; tie annotation list pagination to the existing manuscript list response pattern.

Bandit skipped: documentation-only design task; no Python source was touched.

Post-review hardening: origin/dev introduced TASK-496/TASK-497 collisions while this branch was in flight, so this design task was initially renumbered to TASK-505. User-requested review findings were patched into the spec: added saved-scene editor binding prerequisite, clarified ChaChaNotes schema/migration ownership, aligned optimistic locking with existing expected_version header conventions, made provider/model request fields explicit for AI review endpoints, resolved V1 caps/default scene-review max, and changed anchor reattachment to derived-by-default.

Second spec review found one blocking ambiguity: open questions contradicted the hardened anchor semantics and left derived anchor_status filtering/pagination unclear. Patched the spec to state that V1 derives anchor_status on list/read without mutating rows, applies anchor_status filtering before total/pagination, omits that filter if accurate derived totals are not implemented, and removed the resolved anchor persistence questions.

Final post-hardening spec review returned Approved with no issues. Implementation-planning advisories to carry forward: make clear that the stored anchor_status column is not the V1 source of truth for list/read responses, and decide whether direct annotation access needs GET /annotations/{annotation_id} or list-only reads are sufficient.

Folded final review advisories into the spec: clarified that stored anchor_status is not the V1 list/read source of truth, and added direct annotation access as an implementation-planning question.

User review requested a UX adjustment: make Google Docs-style margin comments the primary desktop annotation surface, while keeping inspector/list behavior for management and narrow/extension fallback.

Margin-comment UX adjustment applied to the spec. Desktop scene range comments are now specified as right-side margin cards aligned to inline highlights, with the inspector retained as the management surface and responsive/narrow extension fallback. Local review found no contradiction with V1 scope: full Google Docs parity remains out of scope.

Earlier post-rebase verification found TASK-497 collisions already present on origin/dev, so this design task was renumbered to TASK-505 at that time.

User approved the review findings for follow-up spec hardening. The spec now splits frontend manual annotation work into foundation and margin-rail slices, defines margin-card positioning and collision behavior, constrains broad derived anchor_status filtering, resolves direct annotation access into V1, and adds keyboard/accessibility acceptance criteria for margin comments.

Latest rebase onto origin/dev found new TASK-505 collisions already present upstream. Renumbered this design task to TASK-607 and updated the spec Backlog reference.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed durable manuscript-owned Writing Playground annotations for saved manuscript scenes, chapters, and projects. The spec covers scene range anchors with drift handling, chapter/project notes, manual annotations, backend-owned selected-text AI critique, Jobs-backed scene review, suggested-fix handoff into the existing revision queue, WebUI/extension shared UI expectations, permissions, safety boundaries, rollout stages, and verification strategy. Spec review approved with no blocking findings.

Post-review hardening was reviewed again and approved with no blocking issues.

User-requested UX adjustment added desktop margin comments as the primary scene range comment surface, with inspector/drawer fallback for narrow and extension layouts.

Follow-up review hardening split the frontend rollout, added a deterministic margin-rail layout contract, constrained expensive anchor-status filtering, resolved direct annotation lookup into V1, and added accessibility verification.
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
