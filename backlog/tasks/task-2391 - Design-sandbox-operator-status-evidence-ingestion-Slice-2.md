---
id: TASK-2391
title: Design sandbox operator status evidence ingestion Slice 2
status: Done
labels:
- sandbox
- operator-ux
- vz_linux
- design
modified_files:
- Docs/superpowers/specs/2026-06-19-sandbox-operator-evidence-ingestion-design.md
- Docs/superpowers/specs/2026-06-18-sandbox-operator-status-consolidation-design.md
- backlog/tasks/task-2391 - Design-sandbox-operator-status-evidence-ingestion-Slice-2.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design Slice 2 for bounded host-gated VZ smoke evidence bundle ingestion into the consolidated sandbox operator-status endpoint. Scope is a read-only spec/update covering env-configured evidence directory input, safe host-smoke-evidence.json parsing, advisory status classification, privacy boundaries, tests, and non-goals. No implementation code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec/update defines evidence bundle input shape and explicitly avoids Markdown scraping.
- [x] #2 Spec/update keeps path input env/server-config only for this slice and rejects request-supplied paths.
- [x] #3 Spec/update documents safety constraints for missing, unsafe, symlink, oversized, malformed, stale, expected-skip, and blocking-failure evidence.
- [x] #4 Spec/update defines operator-status evidence section/status/action behavior without mutating helper/image-store/evidence files.
- [x] #5 Spec/update is committed and linked from the task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added focused Slice 2 design spec for env-configured host-gated evidence bundle ingestion into sandbox operator status.
- Clarified that server code reads the evidence bundle JSON (`host-smoke-evidence.json`) and must not parse the Markdown summary or import the CLI summarizer.
- Locked down env/server-config-only path input for this slice; no request-supplied paths.
- Documented fail-closed handling for unsafe paths, symlinks, oversized JSON, malformed JSON, unsupported schema, stale evidence, build/sign skips, and non-zero final exits.
- Updated the parent operator-status design to link to the Slice 2 spec and use evidence-bundle terminology.
- Review hardening: made expected files explicit, required fail-closed safe traversal, allowlisted runtime path pointers, excluded `helper_path`, bounded phase/string output, and added schema/timestamp edge-case tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed Slice 2 for advisory evidence ingestion in `Docs/superpowers/specs/2026-06-19-sandbox-operator-evidence-ingestion-design.md`, with a parent-spec link in `Docs/superpowers/specs/2026-06-18-sandbox-operator-status-consolidation-design.md`. Follow-up spec review tightened the parser contract around expected files, schema versions, timestamp handling, path-pointer privacy, bounded collections, and fail-closed traversal. Verification: `git diff --check` passed. This is documentation/spec-only, so Bandit was not applicable.
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
