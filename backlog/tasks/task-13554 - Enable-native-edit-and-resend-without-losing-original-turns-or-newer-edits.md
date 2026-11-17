---
id: TASK-13554
title: Enable native edit-and-resend without losing original turns or newer edits
status: Done
labels:
- frontend
- chat
- ux
- bug
dependencies:
- TASK-13553
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement accepted D7 CM01 Edit and resend against the native owner and existing P1 explicit branch contract. Capture a detached before-message ancestry under the original mounted owner/source fence; preserve original canonical nodes and exact editor/newer-input intent through pending, refusal, interruption and late outcomes. Follow only verified settlement when the originating edit is still current. Native Regenerate remains a separate required server entry point.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A loaded native user edit admits a new sibling user at its original parent with branch true and settles a new reply; original nodes remain byte-for-byte unchanged.
- [x] #2 Pending, refused, interrupted, stopped or retired actions do not truncate/install a branch or discard exact/newer editor input; no automatic keep/follow on partial action recovery.
- [x] #3 Full page, full extension and deployed compact callers share the intent and current editor lease; native ordinary and character paths revalidate target identity/revision and owner before dispatch.
- [x] #4 Causal tests, actual HTTP graph controls, both types, authored static/format checks and two independent reviews are recorded without claiming unrun native/package/physical gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required:no for this unit. Existing ADR049/050/059 and accepted D7 section6.2 govern owner-bound capture, independent node identities and confirmed UI publication. Public protocol remains the shipped P1 tldw_history_branch field; new intent/editor lease is private client state. Regenerate public API change will receive its own assessment. Search resend found no duplicate delivery task. HighestID13553 verified across freshly fetched dev97ea9cd, all open PR paths and both local backlogs.
Stage 1 reproduced four character edit-resend failures against the actual mounted full hook. Detached action dispatch now uses the shipped before_message selection and branch:true, but the existing common request sanitizer trims authored nonblank user text. Extend this same unit to preserve nonblank user text verbatim in the shared serializer while retaining blank-text and image URL validation; qualify pure preparation and both service transports. Ordinary and character action result follow remains gated by the live editor lease. Existing Regenerate it.fails remains an open defect and is not a positive result.
Independent state and transport reviews found source-lease composition, same-ID H1 epoch retirement, noncaptured follow acknowledgement, Compare fallback, payload loss, and automatic recovery defects. Shared fixes preserve displayed payload checks, separate source/editor from original H1 leases, require captured final installation, block Compare, and retain explicit edited-action results outside auto-keep. Recovery causal red: 13 genuine failures in three suites (142 filtered skips) before implementation. Truthful completed-reply copy and explicit guarded Open saved reply use a fresh scoped receipt reread; no replay or partial auto-follow. Normal same-ID control fixture corrected to actual strict native capture node shape (preview, not unsupported content). Regenerate it.fails remains excluded from positives.
Final 25-suite matrix: 1223 runner passes, of which 1222 are positive and one remains the Regenerate it.fails counterexample. Both client compilers and authored lint/diff pass. Same-browser native ordinary Edit and resend preserves four exact original records and adds the expected sibling/reply; exact whitespace and pending editor retention verified. Real character walkthrough is refused before dispatch because native captured user metadata includes the backend structural sender_role=user. Extend this unit at the shared payload/helper root: allow only matching structural role metadata; require original selected canonical content and matching metadata references before permitting an edited metadata-bearing target, and retain the original capture during detached fresh validation. Do not strip or permit substantive metadata. Native capture read is evidence only, not a forged action admission. The retained character draft and original records remain unchanged.
Final structural-role causal red: 7 genuine failures; repaired three-suite subset312 positive. Final25-suite matrix1229 runner passes =1228 positive +one existing Regenerate it.fails, excluded from acceptance. Current web/extension types exit0; effective authored lint43 files0 errors/introduced findings43 inherited warnings; authored formatting/mirrors/diff pass. Both independent reviews clear including original canonical role metadata validation. Actual backend controls32 +authenticated HTTP graph controls39 pass. Bandit N/A: authored scope TS/TSX/JSON/Markdown only. Earlier native ordinary exact four-record preservation/new sibling proof is bounded to its earlier backend runtime. Fresh current backend HEADfd85d58 includes fetched dev97ea9cd. Fresh IAB roleplay acceptance is unverified: connection Save reveals shell and lazy modal chunks fail alongside API reads; host GET/chunk/CORS probes pass, supported reload/IPv4 retarget did not fix it. Source review finds no justified common account abort/redirect repair; keep owner gates. Current preview evidence current-preview-block.jpg. Native/packaged/physical/AT and all-ten gates stay open in TASK13458. Completed owned plan recorded at /private/tmp/chat-native-edit-resend-13554-completed-plan.md and master record; retire only that completed plan. Next reviewable unit fixes old-assistant generating state during detached action.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented native Edit & resend using detached explicit before-message branch admission with original graph preservation, exact current-editor/source/owner fences and guarded verified result follow. Character structural sender_role metadata is validated against original canonical capture; substantive metadata remains refused. Completed edited replies stay in scoped recovery for explicit Open when the originating editor retires, without partial automatic keep/follow. Exact nonblank text and atomic history/row publication are shared across deployed callers. 1228 positive frontend regressions,32 backend controls and39 actual HTTP controls pass; both client types/static gates and two reviews clear. One existing Regenerate expected-failure is excluded. Fresh current native roleplay is blocked by preview lazy-module/API load failures and remains an explicit program gate; earlier native ordinary evidence is bounded. No Python authored; Bandit N/A. All-ten program and separate graph/settings/acceptance work remain In Progress.
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
