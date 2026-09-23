# CI Required Gates

This document defines the required pull-request gate contract for `dev`.

## Required Check Names

The active `dev-core-required-gates` repository ruleset requires these checks:

1. `backend-required`
2. `security-required`
3. `coverage-required`
4. `frontend-required`
5. `e2e-required`
6. `container-build-check`

These check names are stable and must remain unchanged. Each status is bound to
GitHub Actions integration `15368`.

## Live `dev` Enforcement

GitHub aggregates two active, `dev`-only repository rulesets:

- `dev-core-required-gates` (ID `21824526`) requires all six checks above,
  uses strict/current-base enforcement, and has no bypass actors.
- `frontend-license-gate-dev` (ID `19362594`) retains the pull-request rule
  and requires `frontend-license-policy/trusted/dev` from integration `15368`.

The effective pull-request rule sets the required approving-review count to
zero, dismisses stale reviews after a push, and requires extra approval for
unattributed changes.
Neither ruleset grants an administrative bypass; GitHub reports
`current_user_can_bypass: never` for both.

If the additive core ruleset malfunctions, disable ruleset `21824526` without
deleting it or modifying ruleset `19362594`, then record the before/after API
responses in TASK-13013.2.

### Container Build Check Details

`container-build-check` validates that the `app`, `webui`, and `admin-ui` Dockerfiles build successfully on PRs to `main` and `dev`. The workflow uses a matrix strategy with `fail-fast: false`, so all three images are tested even if one fails. A summary job rolls up the matrix results into a single `container-build-check` status for branch protection.

See [Container Image Lifecycle](Container_Image_Lifecycle.md) for the full build and publish pipeline.

## Conditional Execution and No-op Behavior

Each required gate is *designed* to always report a status, so branch protection behaves
deterministically. Whether it does depends on repository configuration -- see
[Gate Reporting and the `workflow_run` Admission Path](#gate-reporting-and-the-workflow_run-admission-path),
and check current behaviour rather than assuming.

- If relevant paths changed, the gate executes its full checks.
- If relevant paths did not change, the gate exits with an explicit no-op success message.

Examples:

- UI-only PRs no-op `backend-required` and `coverage-required`.
- Backend-only PRs no-op `frontend-required`.
- `e2e-required` runs on frontend changes and selected backend API/schema/auth paths.

## Gate Reporting and the `workflow_run` Admission Path

Tracked as TASK-13355. Every claim below is dated, because the behaviour depends on
repository configuration that changes.

### What was observed, 2026-09-22

With `LICENSE_FIRST_CI_ENABLED` **unset**, required gates did not report at all. Across
four PRs (#2981-#2984) and six pushes, every check reached `CANCELLED` within roughly 40
seconds of each push -- 45 to 50 checks per PR, all six required gates included, with
zero `FAILURE` -- and each PR stayed `BLOCKED` until the gates were re-run by hand. All
four were landed with the procedure below.

### Configuration change, 2026-09-23

`LICENSE_FIRST_CI_ENABLED=true` was set at 05:14:57Z, which is the first of the two
resolutions TASK-13355 offers. It was then tested on PR #2988 and **it does not unblock
pull requests.** What it changes and what it does not:

- `admission` now succeeds instead of skipping. Confirmed: run `35821927623`
  (`event=workflow_run`, created 05:19:24Z) shows `admission / admission:
  completed/success`, where the same job was `SKIPPED` before.
- The admitted run then does the real work and passes. Confirmed: the same run reached
  `completed/success` on all three jobs including `backend-required` itself.
- **But its status never reaches the pull request.** That run's `head_sha` is
  `c2bab8a5`, the tip of `main`, not the PR head `116c9314`. `gh pr checks 2988` still
  lists exactly one check named `backend-required`, the `CANCELLED` one from the
  `pull_request` run.

The cause is a permissions asymmetry. `frontend-license-gate.yml` holds
`statuses: write` and posts its status explicitly against the PR head, which is why
`frontend-license-policy/trusted/dev` does appear on the PR. The six required workflows
hold only `contents: read` and rely on GitHub's implicit check run, which attaches to
their own `head_sha` -- and for a `workflow_run` event that is the default branch.

So only the `pull_request` run can report to a pull request, and that is the run the
admitted one cancels. Enabling the variable makes each PR run the gates twice, neither
occurrence of which unblocks it. **This is a design gap, not a configuration gap**, which
is why resolution A alone is insufficient.

### Mechanism

Steps 1, 2, 4 and 5 were observed directly. Step 3 is an inference; the evidence for it
is given.

1. A pull-request event fires `Frontend License Gate Audit`, which triggers on
   `pull_request_target` with types `[opened, reopened, synchronize, ready_for_review,
   edited]`. **`edited` is included, so editing a PR description counts.**
2. When that audit completes it fires every workflow declaring
   `workflow_run: [Frontend License Gate Audit]`. Those runs are attributed to the
   default branch, so they do not appear when listing runs for the PR branch -- which is
   what made this hard to see.
3. *Inferred:* a `workflow_run` run resolves its concurrency group through
   `github.event.workflow_run.pull_requests[0].number`, the same PR number the
   `pull_request` run used, so both share one group and `cancel-in-progress: true` kills
   the `pull_request` run. GitHub does not expose which run cancelled which, so this is
   not directly observed. The evidence is that cancellations landed about six seconds
   after the audit succeeded, and that `license-first-admission.yml` requires
   `workflow_run.pull_requests` to contain exactly one valid number, so the field that
   collapses the two groups is populated.
4. That `workflow_run` run's `admission` job requires
   `vars.LICENSE_FIRST_CI_ENABLED == 'true'`, which was unset before 2026-09-23, so it
   was `SKIPPED`.
5. Each gate job requires `needs.admission.result == 'success'` for `workflow_run`
   events, so it skipped too. Observed as `event=workflow_run` runs of
   `backend-required`, `coverage-required`, `frontend-required` and `pre-commit` all at
   `completed/skipped`.

The run that would have reported a status was cancelled by a run that then reported
nothing.

### Checking the current state

Do not trust the prose above; check. The question that matters is whether the six
required check *names* reach a conclusion on the pull request, so ask the pull request:

```bash
gh pr checks <pr> --json name,state \
  --template '{{range .}}{{.state}} {{.name}}{{"\n"}}{{end}}' \
  | grep -E "required|container-build-check"
```

- `SUCCESS` for all six: gates are reporting, nothing to work around.
- `CANCELLED`: the run that would have reported was superseded. Use the landing
  procedure below.
- A name missing entirely: that lane never started for this head.

To see whether the admitted path is doing the work, and against which commit:

```bash
gh api "repos/rmusser01/tldw_server/actions/runs?event=workflow_run&per_page=20" \
  --template '{{range .workflow_runs}}{{.name}} {{.conclusion}} {{.head_sha}}{{"\n"}}{{end}}' \
  | grep -E "required|container-build-check"
```

`skipped` means admission is declining. A conclusion paired with a `head_sha` that is the
default-branch tip rather than the PR head means the work ran but the status landed
somewhere the pull request cannot see -- which is the state recorded above for
2026-09-23.

### Landing a pull request when gates do not report

`Helper_Scripts/ci/land_required_gates.sh <branch> <pr-number>` automates this. Run it
from a checkout whose branch is already rebased onto the current `dev` and pushed. Pass
`--no-merge` to stop after the gates go green.

The steps it performs, and why each is shaped that way:

1. **Rebase onto the current `dev` and push.** `dev-core-required-gates` uses
   strict/current-base enforcement, so a branch even one commit behind cannot merge, and
   GitHub reports the required checks as "expected" rather than counting those that
   already passed on the older base.
2. **Wait for the audit on that exact head SHA to complete.** Not merely "the latest
   audit" -- a rebase-push spawns a new one, and re-running gates while it is pending
   gets them cancelled when it finishes.
3. **Re-run the six gates individually, paced roughly 40 seconds apart.** A burst makes
   them cancel each other, and `gh run rerun` fails on a run that is already queued or
   in progress, which silently skips that gate.
4. **Merge once all six are green**, before anything else lands on `dev`.

**Landing several PRs is serial, not parallel.** Strict-base enforcement means each merge
moves `dev` and puts every remaining PR behind, so each one needs its own
rebase-push-gates-merge cycle, and an unrelated merge landing midway restarts the cycle
for the rest. Four PRs took roughly four hours of wall-clock on 2026-09-22 for this
reason.

Three traps worth stating outright:

- **Do not edit the PR description after opening it** if you want its gates to survive.
  `pull_request_target` fires on `edited`, which spawns an audit and cancels every gate
  run in flight.
- **`--admin` does not bypass either requirement**, and they fail differently. Neither
  ruleset grants an administrative bypass (`current_user_can_bypass: never`), so with
  checks missing `gh pr merge --admin` is refused with *"Repository rule violations found
  / 6 of 6 required status checks are expected"*. Separately, when the branch is behind,
  a plain `gh pr merge` is refused with *"the head branch is not up to date with the base
  branch"*. Merging through the GitHub UI as a user whose own admin bypass applies does
  work.
- **`--squash` is rejected repository-wide**; merges must use `--merge`.

## Security Threshold Policy

`security-required` enforces blocking findings at `HIGH`/`CRITICAL` severity with an allowlist.

- Allowlist file: `.github/security/ci-allowlist.yml`
- Every allowlist entry must include:
  - vulnerability id
  - owner
  - expiry date (ISO format)

Expired allowlist entries are ignored by the gate.

## Rollout Status

1. Introduce required lanes and deterministic no-op semantics.
2. Tighten blocking behavior across required lanes.
3. Refine path coupling and flake handling in `e2e-required`.
4. Enforce the required lane names on `dev` with strict/current-base checks.
5. Enforce `container-build-check` with the other required statuses.

The required-status enforcement described in phases 4 and 5 is complete for
`dev` as of 2026-08-29. The controlled TASK-13013.2 proof PR recorded a
failing `coverage-required` check from integration `15368` and GitHub reported
the ready pull request as `BLOCKED`.

## Legacy CI Workflow

The large legacy `.github/workflows/ci.yml` workflow remains available during rollout for broad visibility and historical comparison.
Required merge protection is provided by the six lanes listed above.
