---
id: TASK-233.4
title: Design OpenWebUI webui.db chat import
status: Done
assignee: []
created_date: '2026-05-10 21:46'
updated_date: '2026-05-10 21:52'
labels:
  - chatbooks
  - openwebui
  - design
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-10-openwebui-db-chat-import-design.md
  - https://docs.openwebui.com/reference/database-schema/
parent_task_id: TASK-233
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a repo-grounded design spec for v2 OpenWebUI import from an uploaded OpenWebUI webui.db SQLite file through the existing Chatbooks import workflow. The approved scope is upload-only, selected OpenWebUI user only, visible folder mirroring under an OpenWebUI/user namespace using existing tldw folder support, and attachment/file/artifact references preserved as metadata only without binary hydration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec defines the upload-only webui.db preview/import flow through the existing Chatbooks surface without server-local path support.
- [x] #2 Spec requires preview-time OpenWebUI user selection and imports only the selected user.
- [x] #3 Spec maps OpenWebUI folders/projects into existing tldw visible folder support under an OpenWebUI/user namespace and preserves original source metadata.
- [x] #4 Spec explicitly preserves attachment/file/artifact references as metadata only and excludes binary hydration from v2.
- [x] #5 Spec covers security validation for uploaded SQLite files including safe filenames, read-only opening, schema validation, sanitized errors/logs, cleanup, and focused test expectations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved v2 source-format scope for uploaded OpenWebUI webui.db imports. 2. Ground the design in current Chatbooks import behavior, OpenWebUI schema docs, and existing tldw folder support. 3. Record backend, frontend, security, testing, and documentation expectations for the later implementation plan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created a design spec at `Docs/superpowers/specs/2026-05-10-openwebui-db-chat-import-design.md`.

The spec recommends extending Chatbooks with `source_format=openwebui_db` instead of creating a separate endpoint. It requires uploaded SQLite files only, explicit selected OpenWebUI user import, folder mirroring through existing visible tldw folder support, metadata-only attachment/file/artifact preservation, duplicate compatibility with the JSON importer, and security controls for uploaded private SQLite databases.

Design review follow-up tightened namespace safety, duplicate selected-user labels, use of `chat.folder_id` as authoritative folder membership, `folder.items` as secondary evidence, and project-like source metadata handling when no natural tldw folder destination exists.

Verification: `rg -n "source_format=openwebui_db|selected_openwebui_user_id|OpenWebUI / <selected user label>|Attachment|read-only SQLite|safe filename|project|chat\\.folder_id|folder\\.items|normalize it to a safe folder segment" Docs/superpowers/specs/2026-05-10-openwebui-db-chat-import-design.md` confirmed the required design points.

Verification: `git diff --cached --check` completed with no whitespace errors.

Bandit skip: docs/backlog-only design slice; no Python source changed.

Known skips/blockers: implementation plan and runtime code remain future work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the v2 OpenWebUI `webui.db` import design spec. The spec keeps the feature in Chatbooks, requires explicit uploaded-DB source selection and selected OpenWebUI user import, maps folders into existing tldw visible folder support under an `OpenWebUI / <user>` namespace, preserves unsupported attachments/files/artifacts as metadata only, and records security/test expectations for the implementation stage.
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
