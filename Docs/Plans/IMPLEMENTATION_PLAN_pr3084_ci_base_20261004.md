# PR3084 latest-dev CI base integration

**Task:** TASK-13260.278.18.83.45
**Goal:** Bind final PR3084 checks to the latest reviewed upstream CI diff policy while preserving the complete owned implementation.
**Architecture:** Integrate upstream pre-commit merge-base selection and hook-stage corrections; no owned production/test change.
**Stack:** Existing Python3.12 environment, supported backlog-py task editor and existing validation tools; no installation.
**ADR required:** No new decision; ADR059 task-tooling policy remains governing.

## Stage 1: Inspect and integrate
**Goal**: Review five incoming CI/task paths and preserve six published commits through safe rebase.
**Success Criteria**: Full owned binary patch and all six commits unchanged; only five incoming paths in committed tree delta; pre-edit task notes preserved.
**Tests**: Complete source review, range-diff/binary-patch comparison and exact live base/remote checks.
**Status**: Complete

## Stage 2: Verify and review
**Goal**: Verify affected CI contracts and independently review final source/tracking.
**Success Criteria**: Natural affected test success; unchanged-source checks retained without replay; syntax/lint/security comparison has no new findings; review clear.
**Tests**: License-first workflow, local-CI runner, required-workflow/e2e budget and changed-task-format contracts; existing pre-commit validate-config when available; incoming-test syntax/Ruff/baseline-current Bandit.
**Status**: Complete

## Stage 3: Publish
**Goal**: Normally publish the reviewed integration after immediate latest-dev and exact remote-head checks.
**Success Criteria**: Exact-lease push and authored body/human summary readback; original native/UAT/API approval holds retained; remove only owned completed plans after publication.
**Tests**: Reviewed commit patch equality, diff check/clean checkout, PR head/body readback; fresh final-head CI required.
**Status**: In Progress

## Limits
Affected test operation600s cap/no retry, no install/environment reconstruction/shared-cache changes, source/assertion/CI workaround or stopped native control. Local7a/210498/8e tracking remains excluded. No unchanged app/PG/canonical/frontend tests or build replay, API replacement or browser action. Human summary and original task status/criteria/DoD remain preserved; native/first-import/full UAT remain unaccepted.

## Integration result
Rebased c68858353664712d643c0a5eaad5021cfb486737 onto dev e70e0abb129ea8010f8096d1080b8ea23570f442. All six published commits are identical by range-diff and the complete owned binary patch is unchanged. No owned-path overlap; committed old/new tree delta is exactly five incoming paths. Pre-edit supported task notes were restored through autostash. Independent incoming-source review is clear. Existing pre-commit4.6.2/Ruff0.16.10 are available; no installation.

## Qualification result
Affected four-file CI contract group:96passed30warnings13.95s pytest15.877s outer/natural0/no skips reported, within600s/no retry. Existing pre-commit4.6.2 validate-config exits0. Old73e/current incoming test syntax passes; Ruff I001 and one format flag are identical baseline limitations. Scoped baseline/current Bandit211identical LOW finding signatures/0errors/no new findings. No owned production/test changes and no unchanged app/PG/canonical/frontend/native checks or controls replayed. Fresh final-head CI, first-import/bound-workflow/native natural-exit and live UAT remain open. Distinct API replacement approval remains unanswered.
