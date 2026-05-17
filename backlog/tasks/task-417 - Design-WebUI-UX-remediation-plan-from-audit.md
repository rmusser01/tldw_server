---
id: TASK-417
title: Design WebUI UX remediation plan from audit
status: Done
labels:
- ux
- design
- webui
- extension
- remediation
priority: high
modified_files:
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- backlog/tasks/task-417 - Design-WebUI-UX-remediation-plan-from-audit.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a text-only remediation design/spec addressing every issue identified in the 2026-05-17 WebUI/extension UX/HCI audit. The deliverable is a Markdown design document with work packages plus finding and route coverage matrices. No product code changes are in scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Markdown remediation design spec exists under `Docs/superpowers/specs`.
- [x] #2 Spec addresses every UX/HCI audit finding from F1 through F19.
- [x] #3 Spec covers all 74 audited root/top-level routes.
- [x] #4 Spec decomposes remediation into staged work packages with route and finding coverage.
- [x] #5 Spec preserves the task boundary: text-only remediation design, no product frontend or backend code changes.
- [x] #6 Verification results are recorded for placeholder scan, ASCII/trailing whitespace scan, `git diff --check`, route coverage, and finding coverage.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Follow-up consistency review aligned the spec's WP11 route and finding coverage
with the child implementation split: WP11A owns audio routes, WP11B owns
study/safety/specialized routes, and F9/F18/F19 support rows now point at the
split execution slices instead of the umbrella label.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the WebUI/extension UX remediation program design spec at
Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md.
The spec covers 12 work packages, finding coverage for F1-F19, route coverage
for all 74 audited root/top-level routes, sequencing, verification gates, and
implementation planning rules. Follow-up review tightened the spec by adding an
Interaction Before Explanation principle, clarifying WP11 primary ownership for
`/audiobook-studio`, aligning the later WP11A/WP11B implementation split, and
adding guardrails for child implementation plans, finding/route closure
tracking, overloaded package splitting, and structural UX fixes over
explanation-only copy. Mechanical checks passed for placeholders,
ASCII/trailing whitespace, `git diff --check`, route matrix coverage, and
finding matrix coverage. No product code changes were made.
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
