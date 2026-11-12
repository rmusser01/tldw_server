---
id: TASK-13539
title: Pair shared primary button foreground with its active backgrounds
status: Done
labels:
- chat-ux
- a11y
- theme
- webui
- extension
dependencies:
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native Start chatting uses the shared Button primary variant with fixed white text:3.13:1 on the default dark primary. Its hover/pressed primaryStrong background needs its own paired foreground. Reuse existing semantic CSS variables and contrast computation in the shared root; do not patch each consumer or change theme file schema.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The actual shared Button primary variant uses readable paired normal/hover/pressed foregrounds; other variants and action semantics remain covered.
- [x] #2 Both clients generate the strong foreground utility; default and custom themes derive and clear both foregrounds without a ThemeDefinition schema or import/export change.
- [x] #3 Causal regressions, relevant neighbor suites, both client types, scoped static checks and two authorized independent source reviews pass.
- [x] #4 Native first-time chat action colors are recorded in both default modes with state/packaged/accessibility limits explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only caller trace: Common/Button is consumed by EmptyState and full/compact chat controls and recovery panels, plus existing shared UI consumers. Native actual computed Start chatting:color255/255/255 on92/141/255, class tldw-btn primary. ThemePreview is referenced only by unused ThemeEditorModal; current ThemePicker uses Quick/Advanced editors, so the legacy preview is not a demonstrated live defect and is excluded. Prior13536 closed only the generated opaque token pair. Duplicate primary-foreground search finds Done13536 and parent stages, with no shared-primary-variant/strong-pair task. Highest local/dev/openPR IDs13538/13530/13526. ADR assessment:no new durable decision; extend existing CSS-only derivation, retain version1/17RGB theme tokens; ADR-059 tracking. Plan:Docs/Design/IMPLEMENTATION_PLAN_chat_shared_button_states_13539.md.
All four owned plan stages are Complete; only this completed plan is retired. The shared Common/Button primary variant now consumes the existing normal foreground and matching hover/pressed strong foreground. Existing primary/strong backgrounds stay unchanged. CSS defaults and the real shared Tailwind map add the strong foreground; the existing applier derives both independently with contrastRatio and clears both. Primary pairs switch together via transition-none, avoiding a blended gray label; other variants retain their transitions. No ThemeDefinition version/schema, import/export, dependency or action/focus/loading change. Parent NN/G consistency/readability gates remain separate from this bounded root.
Valid failure-first color/utility/default/custom/reset evidence:9failures/7positive controls (/private/tmp/chat-shared-button-states-13539-causal-red.log). The earlier setup/syntax-error run is excluded. Both independent reviewers requested effective transition coverage; removing only transition-none gives2causal failures/5passing color/interaction controls (/private/tmp/chat-shared-button-states-13539-transition-red.log). The actual rendered-Button/generated-utility helper skips reduced-motion media only for its ordinary-motion check, so media cannot falsely make the baseline pass. Final formatted matrix:136passed in13suites, no skips (/private/tmp/chat-shared-button-states-13539-final-matrix.log), including real EmptyState/full send/compact character/recovery/prompt-assist consumers. Both client type gates pass. Effective unchanged-rule ESLint covers all5authored JS/TS/TSX files including the new test:0baseline/0current/0added. Three owned formatting checks and git diff --check pass; other insertions follow local style. Bandit N/A: JS/TS/TSX/CSS/docs only, no Python repository change. Both independent source reviews clear production; final transition follow-up is clear.
Actual current native first-time Start chatting normal/hover colors are measured in both default modes, with hover:true. Dark normal6.709013:1, dark hover4.835586:1, light normal4.617528:1, light hover6.499871:1. Actual transition-property:none, composer focus and General chat starter selected announcement remain. Receipts:/private/tmp/chat-native-shared-button-13539-20261007.json; before:/private/tmp/chat-native-shared-button-dark-before-13539-20261007.png; after normal/hover photos with dark/light names. Dark restored, no viewport override/new tab/Chrome window. Fresh origin/dev2c5f19d032 is included.
Limits: emitted utility declaration traversal verifies per-client endpoint/transition contracts, not a general browser cascade engine; actual default native normal/hover supplies its live boundary. Held native pressed is generated-CSS-only, and custom native theme editing, packaged extension, manually styled primary controls, transparency/opacity, physical touch/200% zoom, VoiceOver and uncoached participants remain open. ThemePreview has no live caller: only unused ThemeEditorModal refers to it; actual ThemePicker uses Quick/Advanced editors. Its legacy hard-coded sample is not a demonstrated settings-flow issue and is excluded. Entire all-ten program and TASK-13457/13458 remain In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the actual shared primary Button root: normal and stronger hover/pressed backgrounds use their own readable derived foregrounds, and the pair switches together.136checks, both client types, zero scoped lint and two independent reviews pass. Native default light/dark normal plus hover and preserved first-time action are recorded; native held pressed and broader extension/accessibility/manual-control gates remain explicit. No schema/dependency change; parent program In Progress.
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
