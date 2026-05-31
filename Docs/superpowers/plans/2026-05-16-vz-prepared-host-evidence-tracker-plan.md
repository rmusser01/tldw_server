# VZ Prepared-Host Evidence Tracker Plan

## Stage 1: Evidence Contract
**Goal**: Add a durable operator-facing tracker that defines the prepared-host evidence packet for real `vz_linux` runs.
**Success Criteria**: The tracker names required host, git, helper, bundle, command, result, artifact, expected-skip, and residual-gap fields without claiming evidence that has not been recorded.
**Tests**: Focused infrastructure doc tests assert the tracker exists and contains the required acceptance language.
**Status**: Complete

## Stage 2: Cross-References And Boundaries
**Goal**: Link the tracker from the host-gated CI policy, macOS operator notes, and sandbox roadmap.
**Success Criteria**: Contributors can find the tracker from the existing workflow policy and roadmap; the docs keep real VM execution manual or host-gated and keep destructive drills opt-in.
**Tests**: Doc tests assert the policy links the tracker and preserves manual/nightly boundaries.
**Status**: Complete

## Stage 3: Verification And Task Closeout
**Goal**: Validate docs/tests and close the Backlog task with the verification record.
**Success Criteria**: Focused pytest and diff hygiene pass; Bandit is explicitly skipped as docs/test-only if no production Python changes are made.
**Tests**: `python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q`, `git diff --check`.
**Status**: Complete

## Design Review Notes

- Avoid turning evidence tracking into automatic PR execution. Real VZ runs stay limited to manual dispatch, opted-in nightly host-gated runs on trusted refs, and local prepared-host operator commands.
- Avoid stale evidence looking current. Every evidence packet must include date, git SHA, runner identity, bundle/helper versions, workflow or command source, and a residual-gaps section.
- Avoid leaking host/user data. Evidence should preserve artifact names, sizes, paths relative to the runner temp directory, and redacted log excerpts or checksums instead of raw secrets or user workspace contents.
- Keep failure drills and `launchd-drill` separate from default smoke. They remain manually requested evidence fields with expected-skip reasons when not requested.
- Keep this slice docs/test-only. Runtime behavior, repair semantics, and workflow triggers should not change in this PR.
