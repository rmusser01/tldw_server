---
id: TASK-13682
title: Keep settled chat persistence badges readable in both themes
status: Done
labels:
- chat
- ux
- accessibility
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final independent PR3229 review confirmed the small Saved on server, Temporary and failed badge labels use semantic colors with insufficient light-theme contrast. Reuse the established readable foreground while preserving semantic border/background and truthful status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Saved, Temporary and failed labels use the readable existing foreground without changing copy or semantics.
- [x] #2 Mounted causal tests, both-profile verification and actual production light/dark computed contrast qualify the changed states.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Associated with PR #3229 and TASK-13458. ADR assessment: existing design tokens and shared status semantics, no new durable rule; ADR049/050 govern persistence ownership. Two independent final reviewers requested; reviewer B confirmed contrast defect. No dependency or theme schema change.
Actual final-production contrast qualification found a second underlying cause: ChatHeader always declares data-ischat-route=true, yet also applies a Temporary purple900 background override. Tailwind variant ordering wins over the intended surface background; the actual light-theme Temporary badge measures1.57:1 against purple (#65257a) with text #1f2328. First added badge test had an incorrect nested header locator and was corrected without assertion relaxation. Remove the contradictory background override from this always-chat-route header, retain semantic Temporary badge/label and shared surface, then requalify real Saved/Temporary contrast in both themes.
Saved, Temporary and failed persistence labels reuse the established readable text foreground, retaining semantic border/background and truthful status. The conflicting Temporary purple background override is removed from the always-chat-route header. Mounted25 header cases per profile plus real Saved/Temporary light/dark axe/computed styles qualify contrast. Failed shares the warning projection; no forced runtime failure claim. Final reviewed production build DUqqxSjslGXmYUQ7qf1PA: all22 real-server cockpit cases pass with no retries/backend route interception, including JSON/PNG imported card greeting, two protected turns, exact Scene staging/Apply/cold reopen, empty-history Refresh, responsive12-cell matrix and actual Saved/Temporary badge contrast in both themes. Connected Scene/Drawer/Form/Mirror source verification covers450 distinct cases in each client profile:338 cases pass in four complete suites, and the full112 shared editor cases pass separately with fixture cleanup and unchanged behavioral assertions/individual waits and scoped real-modal15s integration budgets. Native and WebUI production connected dependency roots pass. Changed35 TypeScript files:0 errors/0 new findings (final lint report records baseline warning counts). Source-matched58 production deltas;969 emitted JSON manifests,1131 static references,0 missing; shared token sync and590.6/600KB shared /869.4/900KB max budgets pass. Two independent bounded source reviewers report no remaining actionable issue; review is selective, not every line of the whole PR. Original9 conversations/28 rows and compact4-row source remain deep-equal. Fresh whole-PR67 production Python Bandit files:14 identical merge-base findings,0 introduced,0 scan errors. Physical installed-extension docking/promotion, native-toolbar zoom, touch/assistive technology and recruited participant studies remain unrun; publicH2, activeSync and tracked persona remain unavailable. Remote-dev integration and exact-head CI remain separate required gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Saved, Temporary and failed persistence labels reuse the established readable text foreground, retaining semantic border/background and truthful status. The conflicting Temporary purple background override is removed from the always-chat-route header. Mounted25 header cases per profile plus real Saved/Temporary light/dark axe/computed styles qualify contrast. Failed shares the warning projection; no forced runtime failure claim.
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
