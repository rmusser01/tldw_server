---
id: TASK-12019
title: UAT Writing Playground manuscript annotations
status: Done
assignee: []
created_date: '2026-06-26 06:53'
updated_date: '2026-06-26 18:03'
labels:
  - uat
  - webui
  - extension
  - writing-playground
  - manuscripts
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-24-writing-playground-manuscript-annotations-design.md
  - >-
    Docs/superpowers/plans/2026-06-23-writing-playground-manuscript-annotations-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run a post-merge UAT and polish pass for the Writing Playground manuscript annotations workflow, using rendered WebUI evidence and explicitly recording the blocked extension harness status. Capture browser/rendered evidence for saved-scene binding, manual annotations, margin rail behavior, dirty-scene gating, selected-text/scene review affordances, and suggested-fix revision handoff; fix narrow defects found during the pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused annotation backend/frontend regression suites still pass on origin/dev baseline or documented known failures.
- [x] #2 Rendered WebUI or extension UAT exercises the manuscript annotation flow with screenshots/console evidence.
- [x] #3 Any found UAT defects are either fixed with focused coverage or recorded as follow-up work.
- [x] #4 Final task notes include verification commands, browser evidence, residual risks, and dirty/unrelated file notes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-06-26 UAT/polish notes:
- Found rendered WebUI crash in Rich mode: TipTap raised SSR hydration error before the annotation margin rail could render. Added WritingTipTapEditor SSR option coverage and set immediatelyRender: false.
- Found Plain/Rich mode switch marking saved scenes dirty and disabling range comments. Root cause was TipTap plain-text serialization and adapter offsets using single-newline paragraph boundaries while manuscript content_plain uses blank-line paragraph delimiters. Updated serializer + TipTap offset mapping and added focused coverage.
- Rendered Playwright UAT seeds a session/project/chapter/scene plus two annotations, verifies Rich margin rail cards are visible/non-overlapping, switches to Plain, opens the inspector, verifies the annotation list, and asserts no false Scene unsaved/save-before-range-comments state. Evidence: /tmp/writing-annotations-uat/writing-annotations-rich-rail.png and /tmp/writing-annotations-uat/writing-annotations-plain-inspector.png.
- Residual environment noise during UAT: local API /openapi.json returns 500 independently of writing endpoints; Ant Design Drawer logs a width deprecation warning. Extension E2E harness was blocked earlier by service-worker/blank extension launch in this environment, so WebUI shared-component UAT was used for rendered evidence.
- Verification: focused backend annotation suite passed earlier (75 passed); focused frontend annotation suite now passes (10 files, 68 tests); git diff --check passes; package tsc needed NODE_OPTIONS=--max-old-space-size=8192 and then failed on unrelated baseline type errors outside WritingPlayground. Bandit is not applicable because this pass changed frontend TypeScript/tests and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Writing Playground manuscript annotations WebUI UAT/polish and recorded the extension E2E harness as blocked in this environment. Fixed Rich mode TipTap SSR crash, aligned TipTap paragraph serialization/selection offsets with manuscript blank-line delimiters, and made scene dirty checks respect Plain vs Rich editor authority. Added focused frontend coverage and captured rendered WebUI UAT screenshots for Rich margin rail and Plain inspector fallback. Residual unrelated issues documented: local /openapi.json 500, Ant Design Drawer width deprecation warning, and extension E2E harness service-worker/blank-launch blocker.
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
