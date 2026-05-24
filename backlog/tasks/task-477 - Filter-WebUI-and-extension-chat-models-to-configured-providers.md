---
id: TASK-477
title: Filter WebUI and extension chat models to configured providers
status: Done
labels:
- webui
- extension
- chat
- models
- frontend
priority: medium
modified_files:
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/domains/models-audio.ts
- apps/packages/ui/src/services/tldw/TldwModels.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts
- apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure shared WebUI/extension chat model selector data only includes models from providers that are currently configured and enabled, so users do not see impractical catalog-only models in normal selectors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared chat model discovery excludes models from providers explicitly reported as unconfigured or disabled.
- [x] #2 Configured provider models remain visible, including catalog/pricing models for a configured provider.
- [x] #3 Older server responses without provider availability metadata remain backward compatible.
- [x] #4 Focused frontend tests cover configured, unconfigured, disabled, and legacy metadata cases.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing frontend service tests around TldwModelsService/TldwApiClient model availability metadata.
2. Preserve provider/model availability metadata from /api/v1/llm/providers or /api/v1/llm/models/metadata in the shared model service.
3. Filter getChatModels/fetchChatModels results to configured/enabled providers while keeping legacy unknown metadata visible.
4. Run focused Vitest tests and record verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented shared WebUI/extension chat model filtering in the tldw model service. The API client now preserves provider availability metadata and best-effort enriches models from /api/v1/llm/providers when /api/v1/llm/models/metadata omits provider status. Chat model selectors now exclude models from providers explicitly marked is_configured=false, provider_is_configured=false, provider_enabled=false, or failed/disabled/unavailable/not-configured availability, while keeping legacy unknown metadata visible for backward compatibility. Bumped the persisted model cache schema to force stale unfiltered lists to refresh.

Verification: bunx vitest run src/services/__tests__/tldw-api-client.models-normalization.test.ts src/services/tldw/__tests__/TldwModels.test.ts --maxWorkers=1 --no-file-parallelism passed 23 tests in the clean PR worktree. git diff --check exited 0. bunx tsc --noEmit --pretty false hit the default Node heap limit, then NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false completed and failed on existing package-wide baseline type errors; /tmp/tldw_ui_tsc_model_selector_pr_final.log has no entries for touched files. Bandit skipped because only TypeScript/Backlog files were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Shared WebUI/extension chat model discovery is now provider-availability aware. Normal chat model selectors only receive models from providers that are explicitly configured/enabled/available, while legacy servers without status metadata remain compatible. Focused service regression tests cover metadata preservation, provider-list enrichment, image catalog edge cases, unconfigured providers, disabled providers, failed availability, cached unconfigured models, and legacy unknown status.
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
