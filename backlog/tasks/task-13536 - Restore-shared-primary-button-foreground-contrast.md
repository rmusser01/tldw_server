---
id: TASK-13536
title: Restore shared primary button foreground contrast
status: Done
labels:
- chat-ux
- accessibility
- theme
- webui
- extension
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native /chat character greeting review found text-primary-foreground is used by shared primary buttons but absent from the shared Tailwind palette and CSS defaults. Define it once for WebUI and extension, derive a contrasting foreground for existing custom palettes, and clear it with other theme overrides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both client Tailwind configurations generate the existing primary foreground utility.
- [x] #2 Default light and dark primary buttons meet 4.5:1 text contrast at full opacity.
- [x] #3 Applied built-in and custom palettes select the higher contrast black or white foreground using the existing contrast helper.
- [x] #4 Clearing a custom theme removes its derived foreground override and restores stylesheet defaults.
- [x] #5 Native greeting review, relevant theme and greeting tests, both client types, and scoped static checks are recorded with evidence limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. No durable architecture rule, persisted ThemeDefinition field, public API, dependency, or theme import/export schema changes. Reuse the existing shared theme application and contrast helper; ADR-059 governs task editing. Design and staged plan: Docs/Design/IMPLEMENTATION_PLAN_chat_primary_foreground_13536.md. Parent all-ten NN/G program and stages remain In Progress. Baseline: origin/dev 2c5f19d032 integrated at 68c828b295; owned branch HEAD 06b703c439. Read-only duplicate search found no matching open task; highest ID across dev, local tasks and open PRs was 13535. Native before image: /private/tmp/chat-native-character-mobile-light-before-13536-20261007.png. Scope is primary-background text at full opacity; separate opacity/hover, physical touch, zoom and assistive-technology checks remain in the parent program.
Causal red: all six regressions failed on unchanged production code for absent utility/default/application foreground. /private/tmp/chat-primary-foreground-13536-red.log. Actual native light button computed foreground rgb(31,35,40) on rgb(47,111,237). Existing contrast helper reports black/primary 4.617528 in light and 6.709013 in dark, so both default fallback foregrounds are black; forced white would fail dark at 3.130118. Production diff is three files: shared Tailwind mapping, stylesheet defaults, existing theme apply/clear. No ThemeDefinition schema or caller changes.
Green: six regressions pass using both actual compiled Tailwind configs, declared default CSS colors and actual applyThemeTokens/clearThemeTokens over built-in and custom palettes. /private/tmp/chat-primary-foreground-13536-green.log. Regression expectation checks the generated CSS variable binding without fixing Tailwind alpha serialization. Proceeding to native/compiled CSS, theme and greeting matrix, both type gates, covered lint and formatting. Two existing authorized independent reviewers reassigned read-only; no new windows or agents.
TASK-13536 complete: repaired the missing shared primary foreground utility and default/custom apply/clear path with three production-file insertions and no theme schema/dependency/caller changes. Six causal red then green regressions; 50 theme/greeting tests, both client types, effective covered lint (0/0/0), configured new-test format and diff checks pass. Per-client Tailwind CSS artifacts match shared tokens via the common comparator; these are CSS artifacts, not packaged WXT or a production Next build. Native 390px computed colors verify 4.617528 light/6.709013 dark; screenshots and receipt /private/tmp/chat-native-primary-foreground-13536-20261007.json. Viewport reset/dark restored. Independent reviewers clear. Bandit N/A JS/TS/CSS/docs-only. Parent design updated. Plan retired on completion. Opaque primary pair only; ThemePreview sample/hoveropacity/physicaltouch/zoom/AT/packaged/client adoption remain parent work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored the existing primary foreground class once in shared theme infrastructure. Default and custom opaque primary buttons select readable text, and clearing a theme restores defaults. Native greeting light/dark colors, six causal regressions, 50 relevant tests, both types, covered static checks and per-client CSS token comparisons verify the bounded fix. All-ten remediation remains In Progress; no PR/push/merge.
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
