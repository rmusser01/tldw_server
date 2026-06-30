---
id: TASK-233
title: Design OpenWebUI chat JSON import
status: Done
assignee:
  - Codex
created_date: '2026-05-10 16:15'
updated_date: '2026-05-10 16:29'
labels:
  - chatbooks
  - chat
  - import
  - design
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved design spec for importing OpenWebUI chat JSON exports into tldw_server through the existing Chatbooks import workflow. V1 scope is OpenWebUI exported JSON only; direct OpenWebUI database/admin import is planned follow-up, not part of this design.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents OpenWebUI JSON import through the existing Chatbooks import surface
- [x] #2 Spec requires preserving OpenWebUI message trees via parent_message_id
- [x] #3 Spec captures duplicate handling using source=openwebui and external_ref
- [x] #4 Spec records v1 exclusions including direct database import and attachment hydration
- [x] #5 Spec is committed with this Backlog task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write the approved OpenWebUI chat JSON import design spec under Docs/superpowers/specs/2026-05-10-openwebui-chat-import-design.md.
2. Capture architecture, data mapping, user flow, error handling, testing, and v1 exclusions exactly as approved in brainstorming.
3. Run a spec review loop with a focused reviewer prompt and patch the spec if actionable issues are found.
4. Perform the user-requested design self-review against current Chatbooks/ChaCha code paths and patch concrete implementation-planning gaps.
5. Verify the non-code change with content hygiene and git diff checks, record Bandit as not applicable for design-only work, and commit only the spec plus TASK-233 tracking file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote design spec at Docs/superpowers/specs/2026-05-10-openwebui-chat-import-design.md and started the spec review loop.

Spec review iteration 1 found message ID collision risk for rename imports and an underspecified OpenWebUI preview/import response contract. Patched the spec to add import-copy namespacing, explicit optional OpenWebUI preview/result response fields, canonical derived external refs, and stricter v1 role handling.

Spec review iteration 2 approved. Applied advisory cleanup to specify source_format as a multipart form field and align response wording with existing success/message/job_id fields.

Reopened for user-requested design self-review before implementation planning. Found additional planning gaps to patch: OpenWebUI JSON must branch before ZIP validation, JSON filename handling cannot reuse the ZIP-only filename validator unchanged, source metadata should use explicit namespaced storage, and OpenWebUI Unix timestamps need explicit normalization.

Design self-review patch added explicit endpoint/source-format branching before ZIP validation, JSON filename validation constraints, frontend helper changes for source_format, duplicate lookup helper guidance, namespaced conversation/message metadata, UUID-shaped deterministic message IDs, and Unix-second timestamp normalization.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added and then self-reviewed the user-approved OpenWebUI chat JSON import design spec. The spec now defines OpenWebUI JSON import through the existing Chatbooks import surface, preserves message trees through parent_message_id, pins duplicate handling to source=openwebui plus external_ref, excludes direct database import and attachment hydration from v1, and hardens implementation planning around source_format branching, JSON-safe validation, frontend helper updates, duplicate lookup helpers, namespaced metadata, UUID-shaped deterministic message IDs, and Unix timestamp normalization. Verification included content-hygiene scans, full spec review, two spec-review iterations, and a follow-up self-review against current Chatbooks/ChaCha code paths; Bandit is not applicable because this task changed only documentation and Backlog tracking metadata.
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
