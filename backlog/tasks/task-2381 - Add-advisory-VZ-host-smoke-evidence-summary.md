---
id: TASK-2381
title: Add advisory VZ host smoke evidence summary
status: Done
labels:
- sandbox
- vz_linux
- host-gated
- ci
- diagnostics
priority: Medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an advisory-first host-gated evidence summary path for VZ Linux smoke runs. The summary should read structured smoke evidence when available, write operator-friendly GitHub step-summary output, and never mask the primary smoke result in this first slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design spec captures advisory-only summary behavior, malformed/missing evidence handling, and non-goals before implementation.
- [x] #2 The host-gated workflow can run an always-run advisory evidence summary step after smoke/evidence generation.
- [x] #3 The summary reports evidence present/missing, required file presence, phase outcomes, final exit code, cleanup status, and artifact/log pointers when available.
- [x] #4 Malformed or missing evidence produces warnings and exit 0 in this first advisory slice.
- [x] #5 Focused tests cover complete evidence, missing evidence, malformed JSON, and workflow wiring without requiring a real VZ host.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write and commit the approved design spec. 2. Review the spec for issues before implementation planning. 3. After approval, create an implementation plan and implement TDD-first in a separate commit/PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Design spec drafted at `Docs/superpowers/specs/2026-06-17-vz-host-gated-evidence-summary-advisory-design.md`.
- Local design review tightened the read-only contract: append-only GitHub step summary output, no evidence mutation, direct-child-only probes, symlink/non-regular-file skips, and bounded JSON reads.
- Follow-up spec review found and addressed two masking risks: `if: always()` can run after checkout/setup failure when the script or interpreter is missing, and `$GITHUB_STEP_SUMMARY` write failures could otherwise make the advisory step fail. The spec now requires shell guards, stdout/stderr fallback, advisory exit `0`, Markdown sanitization, non-directory evidence path handling, and tests for these cases.
- Subagent spec review was not spawned because the available multi-agent tool requires explicit user authorization for delegation.
- Planning verification: `git diff --check`; Bandit not run for the design-only docs/backlog commit.
- Implementation plan drafted at `Docs/superpowers/plans/2026-06-17-vz-host-gated-evidence-summary-advisory.md`.
- Local plan review tightened the handoff by replacing an implicit evidence-directory probe with exact `lstat` handling for missing, symlinked, non-directory, and unreadable evidence roots. Plan reviewer subagent was not spawned because delegation requires explicit user authorization in this environment.
- Implemented `tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py` and `tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py`. Review loops hardened descriptor-safe evidence reads, intermediate symlink handling, malformed nested metadata filtering, and scalar-only JSON rendering.
- Wired `.github/workflows/vz-linux-host-gated.yml` to run `Summarize smoke evidence` with `if: always()` after managed smoke and before artifact uploads. Updated `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py` to cover ordering, guards, advisory behavior, and docs policy terms.
- Updated `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md` to document the advisory GitHub step summary as the first inline diagnostic surface without replacing smoke results or artifacts.
- Verification: `python -m pytest tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q` passed with 40 tests and 6 existing warnings.
- Verification: `bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` passed.
- Verification: `python tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py --evidence-dir /private/tmp/tldw-vz-evidence-missing >/tmp/tldw-vz-evidence-summary-smoke.md` exited 0 and wrote the advisory missing-evidence summary.
- Verification: `python -m bandit -r tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py -f json -o /tmp/bandit_vz_evidence_summary_final_docs.json` completed with 0 findings.
- Verification: `git diff --check` passed.
- Discarded check: `bash -n .github/workflows/vz-linux-host-gated.yml` is not a valid YAML validation command and failed on workflow syntax; workflow YAML is parsed by the contract tests.
- Real VZ VM smoke was not run for this slice because the change is host-independent advisory reporting and workflow contract wiring.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an advisory VZ host smoke evidence summarizer, wired it into the host-gated workflow, and documented the operator contract. The summarizer is read-only, exits 0 for missing/malformed evidence, appends to GitHub step summary when available, and uses descriptor-safe path handling plus allowlisted JSON rendering to avoid symlink traversal or raw log leakage.
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
