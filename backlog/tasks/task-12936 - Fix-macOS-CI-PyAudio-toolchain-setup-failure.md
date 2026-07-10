---
id: TASK-12936
title: Fix macOS CI PyAudio toolchain setup failure
status: Done
labels:
- ci
- macos
- dependencies
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run 28994210037 job 86048012649 failed before tests in the macOS product-workflows-engine shard while building pyaudio. PortAudio was installed, but Xcode 26.5 clang lookup aborted via xcodebuild. Prepare a minimal CI setup hardening patch and keep it unpushed until the monitored main CI run completes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Patched .github/actions/setup-ffmpeg/action.yml to harden macOS PortAudio/PyAudio native build setup. Verification: YAML parsed with PyYAML; git diff --check passed. Bandit not applicable because this task only touches GitHub Actions YAML. Branch remains unpushed per user instruction until the monitored main CI run completes.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
