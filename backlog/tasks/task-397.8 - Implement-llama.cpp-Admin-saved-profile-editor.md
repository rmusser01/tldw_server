---
id: TASK-397.8
title: Implement llama.cpp Admin saved profile editor
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-16 22:29'
labels:
  - llamacpp
  - webui
  - admin
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1804'
  - 'https://github.com/rmusser01/tldw_server/pull/1804#discussion_r3253657615'
  - 'https://github.com/rmusser01/tldw_server/pull/1804#discussion_r3253657616'
  - 'https://github.com/rmusser01/tldw_server/pull/1804#discussion_r3253657617'
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
parent_task_id: TASK-397
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a focused Admin WebUI profile editor slice for saved llama.cpp launch profiles. The slice should let admins create, edit, duplicate, and delete durable profiles using the existing backend profile APIs, while preserving explicit start/use-in-chat behavior and deferring remote downloads/catalog workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Profiles panel lets admins create, edit, duplicate, and delete saved llama.cpp profiles using existing backend API payloads.
- [x] #2 Admin page wires profile create, update, and delete actions to tldwClient and refreshes profile/runtime state after successful mutations.
- [x] #3 Profile saves do not auto-start llama.cpp instances or auto-wire Chat; those remain explicit runtime actions.
- [x] #4 Focused Admin llama.cpp Vitest coverage covers the profile editor and still passes.
- [x] #5 Verification recorded; Bandit is not applicable because this slice touched only TypeScript/Markdown/Backlog files.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from fresh origin/dev worktree codex/llamacpp-admin-profile-editor. Current inspection shows backend profile CRUD client methods already exist in apps/packages/ui/src/services/tldw/domains/models-audio.ts, so this slice can focus on a WebUI profile editor panel plus Admin page wiring and focused Vitest coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1804 Qodo inline review feedback. Verified all three findings against current code: safeJsonStringify silently hid serialization failures, model-less profile saves were possible when no GGUF asset/model path was present, and mmproj_model_id could conflict with server_args.mmproj. Added focused regressions and patched LlamacppProfilesPanel to surface unserializable saved server args, block model-less saves, and block known mmproj path conflicts before save. Focused Admin llama.cpp Vitest suite now passes: LlamacppProfilesPanel, LlamacppRuntimePanel, LlamacppAssetsPanel, and LlamacppAdminPage: 33 tests. Full UI TypeScript check still fails on existing repo-wide baseline errors outside touched llama.cpp profile editor files. Diff whitespace checks passed. Bandit is not applicable because no Python code was touched.
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
