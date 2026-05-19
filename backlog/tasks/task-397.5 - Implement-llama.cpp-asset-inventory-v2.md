---
id: TASK-397.5
title: Implement llama.cpp asset inventory v2
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-16 14:40'
labels:
  - llamacpp
  - backend
  - webui
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-llamacpp-asset-inventory-v2-implementation-plan.md
parent_task_id: TASK-397
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the merged Asset Inventory V2 plan: local asset schemas, imported folder config parsing, GGUF/mmproj/folder asset discovery, stale-path warnings, candidate mmproj pairing, asset register/import endpoints, legacy inventory compatibility, frontend API/types, and a minimal Admin assets panel. Remote downloads and model-family routing remain deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend exposes asset schemas, imported folder config parsing, and local asset scanning for GGUF, mmproj, folder, and unknown assets.
- [x] #2 Backend supports admin-only asset list/register-path/import-folder endpoints while preserving legacy inventory and start-by-model compatibility.
- [x] #3 Asset discovery reports stale-path, allowlist, unknown-capability, and inferred mmproj pairing warnings without remote download behavior.
- [x] #4 WebUI shared client/types and Admin page expose a minimal assets panel with register/import actions and warnings.
- [x] #5 Focused backend, frontend, Bandit, and diff verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Executed Docs/superpowers/plans/2026-05-16-llamacpp-asset-inventory-v2-implementation-plan.md inline with TDD in worktree .worktrees/llamacpp-asset-inventory-v2.

Implementation commits:
- dc0ab2d7c Add llama.cpp asset inventory schema contract
- 9ed473e40 Add llama.cpp local asset discovery
- 2819c1d92 Infer llama.cpp mmproj asset candidates
- eba5b9a32 Add llama.cpp asset inventory APIs
- 9658889e9 Preserve llama.cpp legacy inventory compatibility
- 0a21954c7 Add llama.cpp asset API client types
- 334cb100b Add llama.cpp assets panel
- 1334a5062 Wire llama.cpp assets into admin page

Known verification skip: bunx tsc --noEmit --pretty false could not run because Bun could not write to its tempdir inside the sandbox; the required escalated rerun was rejected by the approval reviewer. Frontend behavior was validated through focused Vitest coverage instead.

Review-fix pass for PR #1764: verified live review comments and fixed only still-valid findings. Fixed: asset endpoint blocking I/O by offloading asset/config scans and mutations to the threadpool, registered mmproj/projector leakage in legacy inventory/resolve paths, missing Ant List rowKey on grouped assets, and silent frontend catch handling. Skipped as already satisfied in current code: Gemini backend symbol/import warnings for datetime/UTC, _QUANT_RE, _canonical_path, and _unresolved_path_key. Skipped as incompatible with the installed Ant Design version: changing Space orientation to direction; the local Vitest run warned direction is deprecated and orientation is the current compatible prop.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented llama.cpp Asset Inventory V2 across backend and WebUI. The backend now exposes asset schemas, imported folder config parsing, GGUF/mmproj/folder/unknown scanning, inferred mmproj candidate metadata, asset register/import endpoints, and legacy inventory/start-by-model compatibility. The WebUI now has shared asset types/client methods plus an Admin assets panel with register/import actions, grouped assets, warnings, row keys, and inferred candidate labels.

Review fixes for PR #1764 added threadpool offloading for blocking asset endpoint work, stricter exclusion of registered mmproj/projector assets from legacy GGUF inventory and model-id resolution, and frontend callback handling that leaves retry inputs intact without swallowing errors.

Verification recorded:
- Backend: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py -v -> 34 passed, 5 warnings.
- Frontend: ./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppInventoryPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx from apps/packages/ui -> 3 files passed, 21 tests passed.
- Bandit: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r touched llama.cpp Python files -f json -o /tmp/bandit_llamacpp_asset_inventory_v2.json -> 0 results.
- Whitespace: git diff --check -> clean.

Deferred by design: remote downloads, model-family routing, automatic profile mutation, and automatic mmproj pairing.

PR: https://github.com/rmusser01/tldw_server/pull/1764
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
