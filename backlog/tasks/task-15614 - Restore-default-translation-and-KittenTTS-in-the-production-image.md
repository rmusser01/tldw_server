---
id: TASK-15614
title: Restore default translation and KittenTTS in the production image
status: Done
labels:
- backend
- translation
- tts
- docker
references:
- Docs/ADR/011-audio-api-semantics.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document Workspace requests translation without naming a provider and selects KittenTTS for Listen by default. Use the effective credential lookup for translation and include the existing KittenTTS extra and a writable cache in the production API image.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider selection covers config values, server overrides, and blank higher-priority keys while preserving explicit providers.
- [x] #2 The production API image includes the existing KittenTTS extra and provides a model cache writable by its runtime user.
- [x] #3 Setup guidance explains how to enable the provider when the deployment config lives in a separate directory.
- [x] #4 Focused translation, KittenTTS and container packaging tests pass, with no new Bandit findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. The changes reuse the existing provider credential lookup and KittenTTS extra. Docs/ADR/011-audio-api-semantics.md still governs model-first speech routing. All five added cases reproduced on Dev. The focused translation, KittenTTS, packaging and release suite passed 53 tests. Ruff passed and Bandit reported no new findings.
Independent review found no blockers. Dockerfiles/Dockerfile.prod built successfully for Linux arm64. The normal appuser used the cache and the speech service decoded 102188 bytes of 24 kHz MP3 from a synthetic passage. Local verification covered the focused suites and the arm64 image.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Translation uses the effective provider lookup shared with the analyzer. The production API image installs the existing KittenTTS extra and makes the model cache writable for appuser. The setup guide explains where to install provider YAML for a custom config.txt. All 53 focused tests and the Linux arm64 synthesis smoke passed. Ruff passed and Bandit found no new issues.
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
