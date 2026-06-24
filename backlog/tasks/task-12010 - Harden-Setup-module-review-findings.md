---
id: TASK-12010
title: Harden Setup module review findings
status: Done
assignee: []
created_date: '2026-06-23 21:16'
updated_date: '2026-06-23 21:39'
labels:
  - setup
  - security
  - review
dependencies: []
references:
  - tldw_Server_API/app/core/Setup
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the validated Setup module review findings: redact installer command secrets, make setup config/readiness/status persistence atomic and locked where needed, require explicit custom embedding trust acknowledgement at the install schema boundary, bound installer subprocesses, and prevent automatic unpinned VCS dependency installs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Targeted regression tests reproduce the validated setup review findings before production changes.
- [x] #2 Installer subprocess logging redacts credentials and subprocess execution is timeout bounded.
- [x] #3 Setup config, readiness, audio readiness, and install status persistence avoid truncation/lost-update hazards.
- [x] #4 Custom Hugging Face embedding trust requires an explicit acknowledgement at the install plan boundary.
- [x] #5 Unpinned VCS dependency installs are blocked unless explicitly overridden.
- [x] #6 Focused setup tests, Bandit on touched setup code, and diff checks are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Manual task record created because the Backlog MCP resources were unavailable and the official Backlog CLI hung on search/list/create in this checkout. The user approved a manual Backlog task-file exception before repository edits.

Plan: Docs/superpowers/plans/2026-06-23-setup-module-review-hardening.md

Implemented:
- Added secret-aware redaction and bounded subprocess output handling for setup installer commands.
- Added a configurable setup subprocess timeout via `TLDW_SETUP_SUBPROCESS_TIMEOUT_SECONDS`.
- Blocked unpinned VCS requirements by default, with `TLDW_SETUP_ALLOW_UNPINNED_VCS=1` as the explicit override.
- Added atomic writes and same-directory lock files for setup config, setup readiness, audio readiness, and installer status persistence.
- Added `trusted_custom_model_acknowledged` to the embeddings install plan and enforced it for direct custom embedding installs.
- Added focused regression coverage for the reviewed failure modes.

Verification:
- `python -m compileall -q ...Setup production files...` passed.
- `python -m pytest ...focused new regression nodeids... -q --timeout=180` passed: 8 passed.
- `python -m pytest tldw_Server_API/tests/Setup/test_install_manager_dependencies.py tldw_Server_API/tests/Setup/test_setup_manager_masking.py tldw_Server_API/tests/Setup/test_audio_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_preview.py -q --timeout=180` passed: 52 passed.
- `python -m pytest tldw_Server_API/tests/Setup/test_setup_readiness_api.py tldw_Server_API/tests/Setup/test_setup_audio_installer_lifecycle_api.py tldw_Server_API/tests/Setup/test_audio_bundle_provisioning.py -q --timeout=180` passed: 47 passed.
- Bandit on touched Setup production files completed with 0 findings in `/tmp/bandit_setup_module_review_hardening.json`.
- Task-scoped `git diff --check -- ...touched files...` passed.
- Repo-wide `git diff --check` was not clean because of an unrelated pre-existing whitespace issue in `tldw_Server_API/tests/FileArtifacts/test_file_artifacts_service_exports.py:317`.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed.
- [x] #2 Tests or verification recorded.
- [x] #3 Documentation updated when relevant.
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip.
- [x] #5 Final summary added.
- [x] #6 Known skips or blockers documented.
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Setup module against the approved review findings. Installer subprocesses now redact secrets, use bounded output capture and timeouts, and block unpinned VCS installs by default. Setup config/readiness/status persistence now uses atomic writes and lock files to avoid truncation and lost updates. Custom embedding installs now require an explicit trust acknowledgement at the install-plan boundary. Focused Setup tests and Bandit passed; the only diff-check caveat is an unrelated whitespace issue outside the touched Setup scope.
<!-- SECTION:FINAL_SUMMARY:END -->
