---
id: TASK-12162
title: Prepare 0.1.36 corrective release
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-05 14:44'
labels:
  - release
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Prepare a 0.1.36 corrective release from current dev to main after the accidental 0.1.35 release merge. Scope: bump version metadata, add changelog/README release summary, validate release contracts, commit, push, and open a main-bound PR.
Prepared 0.1.36 corrective release metadata: bumped pyproject.toml, FastAPI app metadata, README release line, and Docs/mkdocs.yml from 0.1.35 to 0.1.36; added CHANGELOG.md 0.1.36 corrective entry covering PR #2653 and dev/main sync repair; added README 0.1.36 corrective release summary and marked 0.1.35 as superseded. Validation: git diff --check passed; release docs + PyPI workflow contract tests passed 17/17; py_compile tldw_Server_API/app/main.py passed; Bandit on tldw_Server_API/app/main.py wrote /tmp/bandit_release_0_1_36.json with zero results.
Opened PR #2655 against main: https://github.com/rmusser01/tldw_server/pull/2655. Branch codex/release-0.1.36-corrective contains the 0.1.36 corrective release metadata and validation evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Corrected release flow: closed incorrect intermediary-branch PR #2655, fast-forwarded 0.1.36 release commit onto origin/dev, and opened direct dev -> main PR #2656. Review-fix pass for PR #2656: verify and address still-valid Qodo/Gemini comments on MCP external resource redaction, websocket transport exceptions/deep-copying, and defensive manager guards.

PR #2656 review-fix validation: addressed still-valid Gemini/Qodo comments with minimal changes. Fixed websocket resource methods to raise NetworkError for upstream resources/list/read JSON-RPC failures; deep-copied websocket resource metadata/read payloads; hardened wait_for_servers and _replace_server_resources against malformed/None runtime data; redacted related file:// URIs in external resource read text. Validation: git diff --check passed; 4 focused regressions passed; affected MCP external runtime/websocket test files passed 50/50; py_compile passed for touched implementation files; Bandit on touched implementation files wrote /tmp/bandit_pr2656_review_fixes.json with zero results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared 0.1.36 as a direct dev -> main corrective release in PR #2656 after closing the incorrect intermediary-branch PR #2655. The release metadata and changelog/README/docs now target 0.1.36. PR review follow-ups were addressed in 91432af8e0: websocket resource errors use NetworkError, websocket resource payloads are deep-copied, external runtime resource/server guards handle malformed data, and file:// resource read text is redacted. Validation recorded: git diff --check, focused regressions 4/4, affected MCP external runtime/websocket files 50/50, py_compile, and Bandit with zero results. Known skips: informational CodeRabbit/Qodo summary/rate-limit comments required no code changes.
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
