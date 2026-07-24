# Trusted-License-First PR CI Sequencing Design

- **Status:** Approved by the requester on 2026-07-23
- **Backlog task:** TASK-12986
- **Target branch:** `dev`

## Problem

Pull requests currently start the trusted frontend-license audit and every
ordinary PR workflow independently. GitHub Actions does not provide a reliable
workflow-priority setting, so large test matrices, Docker builds, browser
suites, package builds, and security scans can claim runners before the
five-minute trusted audit publishes its branch-qualified status.

The repository's active rulesets require only:

- `frontend-license-policy/trusted/main` for `main`; and
- `frontend-license-policy/trusted/dev` for `dev`.

All other CI is currently informational for merge enforcement. The desired
change is therefore resource sequencing, not a new merge-security boundary:
lightweight diagnostics should remain immediate, while costly PR jobs should
wait for the trusted status on the exact pull-request head.

## Goals

1. Prevent expensive PR jobs from starting before the matching trusted
   frontend-license status succeeds.
2. Keep actionlint, pre-commit, and docs-only validation immediate.
3. Preserve each workflow's existing PR association, check identity, path
   classification, concurrency, and downstream job graph.
4. Preserve manual `workflow_dispatch` behavior when no pull-request metadata
   exists.
5. Fail closed when the trusted status fails, errors, cannot be read, or does
   not become terminal before a bounded timeout.
6. Leave the trusted publisher and both live repository rulesets unchanged.

## Non-Goals

- Do not add runner priority; GitHub does not expose such a control.
- Do not make ordinary PR-controlled workflow code part of the license trust
  boundary.
- Do not add a ruleset bypass or remove a required status.
- Do not use `workflow_run`, a PAT, a new GitHub App, labels, or manually
  published success statuses.
- Do not merge all PR workflows into one monolithic workflow.
- Do not wait for informational CI before allowing a merge.

## Considered Approaches

### Shared wait job in each expensive workflow — selected

Add one reusable, read-only status-wait contract and call it as the first job
in every expensive PR workflow. Make every existing root job depend on that
call. Existing downstream `needs` chains then inherit the sequencing without
changing their work.

This preserves pull-request event context and existing check identities. It
does queue one small waiter per selected workflow, but it prevents the costly
job fan-out that causes the actual contention.

### Single PR CI orchestrator — rejected

One orchestrator could wait once and then call every expensive workflow.
However, converting the current workflows to reusable callees would be a broad
refactor, could rename check contexts, and would couple otherwise independent
workflow contracts.

### `workflow_run` chaining — rejected

Cross-workflow chaining would avoid polling, but a `workflow_run` receives a
privileged default-branch context and does not naturally retain the desired PR
head check association. Running untrusted pull-request code from that context
would create an unnecessary security hazard.

## Architecture

### Reusable status-wait workflow

Add one workflow callable only through `workflow_call`. It accepts or derives:

- event name;
- repository;
- pull-request head SHA; and
- pull-request base ref.

For `pull_request` events targeting `main` or `dev` it:

1. validates the repository value, the 40-character lowercase hexadecimal head
   SHA, and base ref `main` or `dev`;
2. builds the exact context
   `frontend-license-policy/trusted/<base-ref>`;
3. reads commit statuses for the exact head SHA using only a token with
   `contents: read` and `statuses: read`;
4. selects the newest status with that exact context;
5. exits successfully only for `success`;
6. exits immediately with failure for `failure` or `error`; and
7. continues polling for a missing or `pending` status until the bounded
   timeout, then fails.

Each caller grants the gate-call job `contents: read` and `statuses: read`
explicitly. The reusable workflow may narrow those permissions but must not
assume it can elevate a caller's token.

The waiter polls every 30 seconds for at most 15 minutes, and its job timeout is
17 minutes. This bounds runner use and keeps the worst-case status-read volume
below one request per selected workflow per 30-second interval.

For every non-`pull_request` event, including `workflow_dispatch`, `push`,
`schedule`, and release events, the waiter succeeds immediately. It also
succeeds immediately for a pull request targeting any branch other than
`main` or `dev`, because the trusted publisher intentionally does not create a
status for those bases. Existing triggers and supported base branches therefore
remain unchanged.

The waiter uses no checkout, artifacts, cache, secrets, status writes, or
pull-request source execution.

### Expensive workflow wiring

Gate PR workflows that launch any of the following:

- broad or matrix pytest suites;
- Docker image builds;
- Playwright or extension browser suites;
- frontend build, Vitest, or coverage suites;
- CodeQL, Bandit, SBOM, or dependency scans;
- package builds; or
- targeted product suites that install dependencies and run tests.

The initial gated set is:

- `backend-required.yml`
- `ci.yml`
- `codeql.yml`
- `container-build-check.yml`
- `coverage-required.yml`
- `e2e-required.yml`
- `e2e-smoke.yml`
- `frontend-e2e-tiers.yml`
- `frontend-required.yml`
- `frontend-ux-gates.yml`
- `jobs-suite.yml`
- `mcp-unified-rc.yml`
- `notes-remediation-targeted.yml`
- `pypi-package.yml`
- `sbom.yml`
- `security-required.yml`
- `ui-characters-harness-tests.yml`
- `ui-dictionaries-tests.yml`
- `ui-playground-quality-gates.yml`
- `ui-research-workspace-parity.yml`
- `ui-watchlists-a11y-gates.yml`
- `ui-watchlists-extension-e2e.yml`
- `ui-watchlists-help-tests.yml`
- `ui-watchlists-scale-gates.yml`
- `ui-worldbooks-tests.yml`

Each selected workflow receives one gate-call job with explicit
`contents: read` and `statuses: read` permissions. Every existing job that has
no current dependency becomes dependent on that gate. Existing non-root jobs
retain their current dependency chain. Rollup jobs using `always()` must also
depend directly or transitively on the gate and include an explicit successful
gate-result condition; they must skip or fail when the gate fails rather than
turning gate-induced skips into a misleading successful aggregate.

The immediate set remains unchanged:

- `actionlint.yml`
- `pre-commit.yml`
- `onboarding-docs-gate.yml`

Release, scheduled, and manually gated workflows without a PR trigger remain
out of scope.

### Trust boundary

The existing `pull_request_target` publisher remains the only trusted license
decision. Its source-bound status remains the only required merge signal.

The new waiter is deliberately only a resource-control mechanism. A pull
request can propose changes to ordinary PR workflows, including the waiter
wiring, but doing so cannot forge the source-bound required status or make an
unauthorized pull request mergeable.

## Data Flow

1. A pull-request activity targets `main` or `dev`.
2. The existing trusted audit starts independently and publishes `pending` for
   the exact head SHA and base-qualified context.
3. Lightweight PR workflows start normally.
4. Each expensive workflow starts only its reusable wait job.
5. For a `main`/`dev` PR, the wait job observes the exact trusted status at
   30-second intervals for no more than 15 minutes. Other existing event/base
   combinations pass through immediately.
6. On success, the existing expensive workflow graph is released.
7. On failure, error, API failure, or timeout, expensive jobs remain skipped.

No result from ordinary CI is fed back into the trusted license decision.

## Failure Handling

- **Trusted failure/error:** fail the wait job immediately; do not start costly
  jobs.
- **Missing/pending status:** poll until the deadline.
- **API or response-shape error:** fail closed with a concise diagnostic that
  names only repository, head SHA, base ref, and status context.
- **Timeout:** fail closed and instruct the operator to rerun the informational
  workflow after the trusted status becomes successful.
- **Non-PR or unsupported-base event:** preserve current behavior by bypassing
  status waiting based on GitHub event/base metadata, not a user-provided
  override.
- **Retarget or synchronize:** the new event supplies a different base context
  or head SHA, so an earlier decision cannot release work for the new identity.

## Testing

Add focused tests that:

- prove the status-polling path accepts only `main`/`dev` and an exact head
  SHA;
- prove it selects the exact branch-qualified context;
- cover missing, pending, success, failure, error, API failure, and timeout;
- prove manual dispatch succeeds without PR metadata;
- enumerate the gated workflow set;
- statically verify every runner-owning job in a gated workflow has a
  transitive dependency on the wait job;
- prove each gate-call job explicitly receives `contents: read` and
  `statuses: read`;
- prove non-PR events and PRs targeting bases other than `main`/`dev` pass
  through without status polling;
- prove each `always()` rollup skips or fails when the gate does not succeed;
- verify lightweight workflows do not call the waiter;
- verify the trusted publisher workflow is unchanged; and
- validate all touched YAML with the repository's actionlint contract.

Run the focused CI contract suites, workflow YAML parsing, actionlint, Bandit
for any touched Python helper, and `git diff --check`.

## Rollout and Rollback

Land the reusable waiter and workflow wiring in one reviewable PR to `dev`.
The PR that introduces the change is expected to exercise the new wiring from
its own head while the existing source-bound trusted status remains the actual
merge gate.

Rollback is a repository-only revert of the wait workflow and the added
dependencies. No ruleset, branch-protection, secret, token, or external
service rollback is required.
