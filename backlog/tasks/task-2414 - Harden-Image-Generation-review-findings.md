---
id: TASK-2414
title: Harden Image Generation review findings
status: Done
assignee: []
created_date: '2026-06-23 18:17'
updated_date: '2026-06-23 21:57'
labels:
  - backend
  - image-generation
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address Image Generation module review findings: SwarmUI URL/token leakage, blocking workflow adapter calls, missing workflow validation, unbounded image output handling, sensitive Stable Diffusion logging, and expensive reference-image listing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SwarmUI image URL fetches do not send swarm_token to off-origin URLs and block unsafe remote hosts
- [x] #2 Workflow image generation validates request bounds and extra params consistently with file artifacts
- [x] #3 Workflow image generation does not block the event loop while adapters run
- [x] #4 Image extraction enforces output byte limits and image format validation before returning results
- [x] #5 Stable Diffusion adapter logs and errors avoid prompt/path/secret leakage
- [x] #6 Reference-image candidate listing avoids full image decode when metadata is available
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan: IMPLEMENTATION_PLAN_image_generation_hardening_2414.md

Implemented shared Image Generation request validation for workflows and file artifacts, shared image output byte/magic validation, SwarmUI same-origin URL enforcement before authenticated image fetches, workflow off-event-loop backend generation, stable-diffusion log/error redaction, default inline byte cap enforcement, and reference image listing file-size/header optimizations.

Verification:
- Red regressions initially failed for workflow threading/validation, byte caps, stable-diffusion redaction, and oversized reference rows.
- source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Image_Generation/test_openrouter_image_adapter.py tldw_Server_API/tests/Image_Generation/test_together_image_adapter.py tldw_Server_API/tests/Image_Generation/test_novita_image_adapter.py tldw_Server_API/tests/Image_Generation/test_modelstudio_image_adapter.py tldw_Server_API/tests/Image_Generation/test_swarmui_adapter.py tldw_Server_API/tests/Image_Generation/test_image_format_utils.py tldw_Server_API/tests/Image_Generation/test_stable_diffusion_cpp_adapter.py tldw_Server_API/tests/Image_Generation/test_reference_images.py -q => 34 passed.
- source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py::TestImageGenAdapter::test_image_gen_adapter_uses_thread_for_backend_generation tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py::TestImageGenAdapter::test_image_gen_adapter_rejects_invalid_generation_params -q => 2 passed.
- source .venv/bin/activate && python -m pytest tldw_Server_API/tests/FileArtifacts/test_image_adapter_allowlist.py tldw_Server_API/tests/Files/test_files_image_endpoint.py tldw_Server_API/tests/Image_Generation/test_image_generation_config_defaults.py tldw_Server_API/tests/Image_Generation/test_image_reference_capabilities.py -q => 39 passed.
- git diff --check => passed.
- source .venv/bin/activate && python -m bandit -r touched source scope -f json -o /tmp/bandit_image_generation_2414.json => 0 findings.

Worktree PR prep verification on branch codex/image-generation-hardening-2414 at .worktrees/image-generation-hardening-2414:
- Image Generation provider/reference suite: 34 passed.
- Workflow/FileArtifacts/files endpoint/config/reference capability suite: 38 passed.
- git diff --check: passed.
- Bandit touched source scope: 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Image Generation review findings end to end: unsafe SwarmUI authenticated image fetches are blocked off-origin; workflows now validate image params and run backend generation in a worker thread; provider output extraction enforces byte caps and actual PNG/JPEG/WebP bytes; stable-diffusion logging/errors no longer expose prompts, paths, or secret CLI values; reference image candidate listing skips oversized rows before storage access and avoids full image decode for dimensions. Focused tests and Bandit passed.
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
