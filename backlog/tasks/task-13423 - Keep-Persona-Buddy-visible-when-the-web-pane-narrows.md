---
id: TASK-13423
title: Keep Persona Buddy visible when the web pane narrows
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The floating Persona Buddy disappears in the desktop browser when its pane becomes narrower than 1024 pixels, even though the live voice session still works. Keep the active Buddy available in compact panes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An enabled active Persona Buddy renders in a web viewport below 1024 pixels and remains within the viewport.
- [x] #2 Resizing across 1024 pixels preserves the mounted artwork, unsent control draft and live session without another automatic visual-pack or session load.
- [x] #3 Existing disabled, inactive-surface and independent-Buddy ownership gates continue to suppress the legacy shell.
- [x] #4 Targeted regressions, scoped static checks and an actual compact-browser verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
## Implementation Plan

ADR required: no
ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md (existing guidance)
Reason: This bounded responsive correction keeps the existing render ownership, session, persistence and loading boundaries. It removes the legacy Phase 2 desktop-width limit for an active browser pane and uses existing viewport clamping.

1. Add failing regressions for rendering below 1024 pixels and retaining the same shell, artwork and unsent draft across the breakpoint.
2. Remove the web width suppression while retaining enable, surface and independent-ownership gates.
3. Run targeted tests and scoped static checks; verify the actual 858-pixel browser pane and publish through PR 3091.
Implemented the compact-pane correction by removing the legacy web <1024 suppression from BuddyShellHost. Enablement, active-surface and independent-Buddy ownership gates are unchanged; the existing viewport clamp keeps artwork and controls accessible. The legacy Track B design now records that TASK-13423 supersedes its desktop-width restriction. Existing ADR005 applies; no new ownership, persistence, session or loader boundary.

Validation: both narrow-render and resize-retention regressions failed before the fix. The final seven-file targeted run passed 128 tests, including three narrow-pane ownership gates. Scoped ESLint has no errors and the existing line-361 dependency warning; scoped TypeScript has the same four dependency declaration errors as baseline and no owned diagnostics. The added test region passes formatting using the existing shared-file style; whole-file Prettier warnings also occur on HEAD. Bandit is not applicable to this TypeScript/documentation-only change. Self review and independent review found no actionable code defects.

Actual browser verification at 1024, 1023, 390 and 320 pixels preserved the unsent draft and the exact connected session. Loaded sprite frames remained visible; the natural 858×981 pane was restored. API access counts added zero visual-pack list/detail or session reads during resizing. The disposable test draft was cleared without sending, controls closed, and position reset through the UI. Browser console warnings/errors were empty. Runtime source is byte-identical to the repaired source. Private receipts: /private/tmp/buddy-replay-fix-uat-20261003/responsive-browser-receipts.json, responsive-request-comparison.json, responsive-source-verification.json and buddy-visible-natural-pane.png. Tests/static receipts: /private/tmp/buddy-remaining-fixes-20261003/buddy-responsive-final-128.log, buddy-responsive-red.log, responsive-type-baseline-comparison.json and responsive-format-verification.json.

Tooling note: the hosted Backlog MCP was unresponsive. Its local canonical Python implementation supports safe checklist/note mutation but does not expose a separate plan field; the Implementation Plan heading was recorded through that API before code changes, preserving the supported task markers.
October 3 follow-up plan before edits: CI run 37100918985 at e74d3df255 fails two docs refresh parity tests because the published Persona Buddy guide lacks the source draft-retention update. The source guide also still describes the superseded <1024 suppression. Update that user guidance for compact panes, regenerate published docs through Helper_Scripts/refresh_docs_published.sh, run the targeted docs parity tests and retain the failure/pass evidence. Existing responsive architecture and ADR assessment apply; no new ADR or Python behavior change.
Published-guide follow-up completed: source Persona_Buddy_Guide.md now describes compact-pane visibility/session/draft retention and removes the obsolete desktop-width troubleshooting prerequisite. Helper_Scripts/refresh_docs_published.sh regenerated the sole changed published mirror, including the earlier closed-draft guidance. Both CI parity failures reproduced locally before refresh; all 33 test_docs_published_refresh.py tests now pass. Source and published guide are byte-identical; git diff --check passes. No Python, dependency, workflow or assertion changes. Bandit is inapplicable to this documentation-only follow-up. Private failure/pass receipts: /private/tmp/buddy-remaining-fixes-20261003/buddy-published-docs-red.log and buddy-published-docs-green.log. Existing ADR assessment remains applicable.
Independent documentation review follow-up plan: verify the Buddy click handler, correct single-click versus double-click control instructions, record the completed October 3 voice qualification with the requester and direct-observation evidence distinguished, regenerate the published guide, and rerun the targeted docs parity checks. No new ADR is required for these corrections to existing guidance.
Final guide review completed: verified Host/Dock handlers and corrected click-to-react, double-click-to-open, ×/Escape-to-close and image/handle drag instructions. The guide now records completed October 3 voice qualification, distinguishing requester listening/audibility confirmation from direct thinking/speaking/idle observation. Regenerated the published mirror; all 33 targeted docs checks passed again in 17.67 seconds (buddy-published-docs-reviewed-green.log). Independent follow-up review found no actionable findings; source and published guide are byte-identical and git diff --check passes. No production behavior or ADR change in this follow-up.
2026-10-04 publication integration: merged dev502da5bf0ccd into codex/buddy-preserve-closed-draft at a6501c1fc699 with no conflicts. All eight Buddy/VN product and test files remain byte-identical to published head5de9f73564; their prior targeted and human acceptance evidence retains original attribution. Whitespace verification and independent read-only integration review found no actionable issue. Normal publication updates existing PR3091; fresh hosted checks apply to the updated head and are not inferred from prior FullSuite success. No human UAT, provider call or physical audio repeated. No new ADR: existing ADR005/046 apply. PR remains draft pending the human-written Change summary required by Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md before merge.
2026-10-04 review follow-up plan before edits: replace the temporary branch URL for the existing human acceptance receipt with its immutable 5de9f73564a7c36a7afd3e71a8610489692189df commit permalink, regenerate Docs/Published through the canonical refresh script, and run the 33 targeted documentation refresh checks. Verify the receipt bytes remain unchanged. ADR required: no; existing ADR005/046 apply, because this repairs documentation durability without changing behavior or ownership. The requester has now supplied the required human-written Change summary and PR3091 is ready.
Permalink review follow-up completed: the source guide now links the human acceptance receipt at immutable published commit5de9f73564a7c36a7afd3e71a8610489692189df; the canonical docs refresh regenerated its identical Published mirror. The receipt bytes are unchanged, all33 targeted docs refresh checks passed in21.60s, whitespace verification passes, and independent review found no actionable findings. Log: /private/tmp/buddy-pr3091-permalink-docs-20261004.log. No product, Python, test assertions, provider calls or human UAT changed; Bandit is inapplicable. The requester-written Change summary is preserved verbatim in PR3091 and satisfies the human merge gate.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Buddy remains rendered in compact web panes and retains its artwork, unsent draft and connected session across resizing. 128 targeted frontend tests and actual compact-browser checks passed with no extra pack/session reads. The current user guide and generated mirror include draft retention, accurate interaction controls and completed human voice acceptance; 33 targeted docs checks passed. Existing static-tool baseline findings are documented.
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
