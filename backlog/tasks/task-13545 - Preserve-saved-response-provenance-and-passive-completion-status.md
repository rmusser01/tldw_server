---
id: TASK-13545
title: Preserve saved response provenance and passive completion status
status: Done
labels:
- chat-ux
- webui
- extension
- response-metadata
dependencies:
- TASK-13458
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The actual saved Mira reply retains generation_status, model_id, provider and finish_reason on the server, while its reopened message renders no model provenance. Read the already-carried allowlisted metadata in the shared renderer without creating replay authority. Qualify private local_history hints to positively local-owned captures in every traced consumer so native metadata cannot manufacture legacy recovery flags.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record native server and rendered evidence, trace all metadata and private-local-wrapper consumers, and document the minimal design and ADR assessment before production edits.
- [x] #2 Actual native capture and both server mapper paths retain model/validated usage and passive complete/interrupted/stopped/length presentation, preserving canonical roles/content/images and real local behavior; saved metadata cannot grant Retry/Continue or private local hints to native owners.
- [x] #3 Causal and relevant existing behavior suites, both client types, covered effective lint, formatting, security scope assessment and both independent reviewers pass.
- [x] #4 Verify the completed saved reply model badge after native reload, record precise unrun partial/packaged/physical/AT/D7 limits in the parent audit/task, and commit the bounded repair.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Highest task IDs checked: local13544, fresh origin/dev13530, all open PRs13526. Existing searches generation metadata/response provenance/settlement found broader native ownership/CM01 tasks but no matching shared presentation repair. Native proof /private/tmp/chat-saved-generation-13545-before-receipt.json and -before-dom.json: owned completed response has custom-openai-api/gpt-4o-mini/complete/stop; rendered reply has zero model badges. The initial wrong role-field assertion and sandbox network denial are excluded. Both source traces confirm metadataExtra reaches shared Message; prefer one allowlisted presentation reader and consumer integration, preserving image/local behavior. Source-found native local_history recovery flags must be fenced by actual local owner; all sibling wrapper consumers are traced. ADR049/050/059 govern; no new public protocol, persistence or durable architecture rule. Accepted design Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md; plan Docs/Design/IMPLEMENTATION_PLAN_chat_saved_response_13545.md.
TASK13545 bounded saved-response repair complete: shared allowlisted recorded model/provider and exact partial usage; passive complete/partial status with no synthesized recovery; global busy cannot restore stale flags. Both private local_history consumers positively qualify local ownership, preserving canonical native rows and local/image controls. Causal review findings were fixed before final682/24 frontend checks, both client types0,8 effective lint62/62/0,13 owned format regions,8 exact Chrome flat mirror entries and diff pass. Existing backend metadata22/22 and scoped Bandit0 pass; backend logic unchanged, only false mapping guidance corrected. Both final source reviewers clear. Native full reload and390 compact wrapper show actual saved gpt-4o-mini badge/custom-openai-api title with3 intact rows/blank input; matched default-query all5 prior row hashes and separate metadata-enabled receipt match. Final browser returns to full chat/1280x720 with original sticky false, no new tab/window. Cold dev config startup retired; Retry alone did not restore bootstrap, warm reload succeeds; native partial/lost-response/provider quality/packaged/physical/AT/participants/D7 complete journeys are not claimed. Master design records exact artifacts/ADR049/050/059 and limits; all4 owned plan stages complete, only own plan retired. Current remote dev2c5f19d0328360d45bce99b8bae75ce252cc312f freshly fetched and included. Parent Stage5/all-ten and TASK13540 native AC4 stay In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored truthful saved-response model/provider and exact usage in shared full/compact presentation, with passive recorded partial status and positive local-owner guards in both private-hint consumers. Verified682 frontend checks,22 backend metadata checks, both client compilers, zero added effective lint, scoped Bandit0 and two source reviews; actual full/compact native reloads preserve the saved reply and matched earlier server rows. Broader native interruption, packaged/physical/AT/participant/D7 acceptance remains open.
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
