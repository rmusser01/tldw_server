# PR-Only License-Status Bypass Design

- **Status:** Independent spec review approved on 2026-07-23; written
  specification pending final requester review
- **Backlog task:** TASK-12986
- **Target repository:** `rmusser01/tldw_server`

## Problem

The repository requires one source-bound frontend-license status on pull
requests targeting `main` or `dev`. Those required statuses currently share
rulesets with structural branch protections. When GitHub Actions is delayed,
an administrator cannot merge an audited pull request without also bypassing
the structural protections in the same ruleset.

An earlier design proposed making expensive workflows poll the trusted status
before starting. Review rejected that approach: its wait jobs could consume the
GitHub Free runner pool before the trusted publisher starts, create substantial
API load, and accept a stale same-SHA result for a new pull-request event.

The desired result is narrower: keep the trusted status required for normal
merges, but let a repository administrator deliberately bypass only that
status while merging a pull request.

## Live Baseline

The design is based on these live rulesets, read on 2026-07-23:

- The repository default branch is `main`.
- `5653432` (`safee`) targets the default branch and contains deletion,
  non-fast-forward, pull-request, and
  `frontend-license-policy/trusted/main` status rules.
- `19362594` (`frontend-license-gate-dev`) targets `refs/heads/dev` and
  contains pull-request and
  `frontend-license-policy/trusted/dev` status rules.
- Both rulesets are active and have no bypass actors.
- Both required statuses are source-bound to GitHub Actions integration
  `15368`.

Every live assumption must be re-read immediately before mutation. A mismatch
stops the rollout.

## Goals

1. Keep all existing structural protections active and non-bypassable.
2. Keep the branch-qualified trusted license statuses required for ordinary
   merges.
3. Allow the repository administrator role to bypass only a license-status
   ruleset, only while merging a pull request.
4. Avoid any interval without the existing structural or status protection.
5. Make interruption, retry, verification, and rollback fail closed.
6. Leave GitHub Actions workflows and the trusted status publisher unchanged.

The rollout is staged and interruption-safe, but it is not a transaction
across multiple rulesets. GitHub's ruleset update endpoint provides no
documented compare-and-swap condition.

## Non-Goals

- Do not sequence, reprioritize, or rewrite CI workflows.
- Do not bypass deletion, force-push, or pull-request requirements.
- Do not allow direct-push or always-on bypass.
- Do not remove or weaken the source binding on either trusted status.
- Do not make informational CI checks required.
- Do not add tokens, secrets, collaborators, teams, or GitHub Apps.
- Do not automatically merge a pull request.

## Considered Approaches

### Split structural and status rulesets — selected

Remove the required-status rule from each existing combined ruleset while
leaving every structural rule and the empty bypass list intact. Put each
status into a dedicated ruleset whose only bypass actor is repository role
`5` (administrator) with `bypass_mode: pull_request`.

This uses GitHub's native policy boundary and adds no runner, workflow, API
polling, or credential mechanism.

### Add bypass actors to the combined rulesets — rejected

A ruleset bypass applies to every rule in that ruleset. Adding it to the live
combined rulesets would also make structural protections bypassable.

### Gate workflows with polling or an admission token — rejected

Polling creates runner and API pressure. A label or dispatch admission system
requires an additional credential and operational state. Neither is necessary
to permit an intentional administrative merge.

## Target Ruleset Architecture

### Existing main ruleset `5653432`

Update in place and preserve:

- name `safee`;
- target `branch`;
- active enforcement;
- condition include `~DEFAULT_BRANCH`;
- empty bypass actor list;
- deletion rule;
- non-fast-forward rule; and
- the existing pull-request rule and all of its parameters.

Remove only its `required_status_checks` rule.

### Existing dev ruleset `19362594`

Update in place and preserve:

- name `frontend-license-gate-dev`;
- target `branch`;
- active enforcement;
- condition include exactly `refs/heads/dev`;
- empty bypass actor list; and
- the existing pull-request rule and all of its parameters.

Remove only its `required_status_checks` rule. The existing name is retained
to avoid an unrelated rename.

### New main status-only ruleset

Create `frontend-license-status-main` with:

- target `branch`;
- condition include exactly `refs/heads/main`;
- no exclusions;
- active enforcement after staged verification;
- exactly one `required_status_checks` rule;
- context `frontend-license-policy/trusted/main`;
- integration ID `15368`;
- `strict_required_status_checks_policy: false`;
- `do_not_enforce_on_create: false`; and
- exactly one bypass actor:
  - `actor_type: RepositoryRole`;
  - `actor_id: 5`;
  - `bypass_mode: pull_request`.

### New dev status-only ruleset

Create `frontend-license-status-dev` with the same shape as the main
status-only ruleset except:

- condition include exactly `refs/heads/dev`; and
- context `frontend-license-policy/trusted/dev`.

No status-only ruleset may contain a pull-request, deletion, or
non-fast-forward rule. Repository role `5` includes current and future
repository administrators; no other repository role is accepted.

## Staged Activation

1. Read and save the complete live payloads for both existing rulesets,
   effective branch rules for `main` and `dev`, repository permissions,
   administrator collaborators, and the current authenticated actor.
2. Establish an explicit ruleset-maintenance window in which no other
   administrator or automation will edit repository rules. Assert that the
   repository default branch is exactly `main`, the authenticated actor is an
   administrator, only the expected administrators exist, and the exact
   ruleset baseline described above still holds. Also assert that no ruleset
   already uses either proposed new name. Stop on any mismatch or if exclusive
   policy editing cannot be established.
3. Derive normalized writable before-state and rollback payloads from the live
   responses. Each contains only the accepted update fields `name`, `target`,
   `enforcement`, `bypass_actors`, `conditions`, and `rules`. Prove that
   normalizing each rollback payload reproduces the corresponding live policy.
   Do not retype mutable pull-request parameters.
4. Create both status-only rulesets with enforcement `disabled`.
5. Read them back and assert exact target, conditions, status context,
   integration ID, bypass actor, bypass mode, and disabled enforcement.
6. Activate both status-only rulesets and verify them. At this point the
   trusted statuses are deliberately required twice, so interruption remains
   fail closed and no bypass is yet effective.
7. Immediately before updating ruleset `5653432`, re-read the repository and
   ruleset. Require `default_branch == "main"` and require its normalized
   writable payload and `updated_at` to equal the approved stage state. Derive
   the update from that final response, remove only the duplicated status rule,
   and issue the update without an intervening network operation. Read it back
   and prove every structural field and the empty bypass list match the saved
   baseline.
8. Immediately before updating ruleset `19362594`, re-read it. Require its
   normalized writable payload and `updated_at` to equal the approved stage
   state. Derive the update from that final response, remove only the
   duplicated status rule, and issue the update without an intervening network
   operation. Read it back and prove its pull-request rule, conditions, and
   empty bypass list match the saved baseline.
9. Read the effective rules for `main` and `dev`. Assert that each branch still
   has its original structural protections and exactly one required
   branch-qualified trusted status.

If execution is interrupted, the next attempt first discovers rulesets by
recorded ID and exact name. It may resume only when every observed field
matches an already verified stage; otherwise it stops for manual review.

Changing the repository default branch later requires a coordinated migration
of ruleset targets, trusted workflow triggers, and status contexts. It must not
be treated as an unrelated repository setting change.

## Operational Behavior

For an ordinary merge, the trusted branch-qualified license status remains
required and source-bound to integration `15368`.

When that status is unavailable, a repository administrator may use GitHub's
explicit ruleset-bypass path while merging a pull request. The bypass does not
authorize a direct push, branch deletion, force-push, or skipping the
pull-request requirement because those rules remain in rulesets with no bypass
actors.

Before bypassing, the administrator must inspect and authorize the exact
pull-request head SHA. Record the pull-request number, base branch, head SHA,
reason, and human-written Change summary. This policy change enables a bypass;
it does not perform a merge or waive the repository's AI-generated PR policy.

## Failure Handling and Rollback

- **Preflight mismatch:** make no mutations.
- **Creation/read-back failure:** leave any new ruleset disabled and stop.
- **Activation failure:** disable any newly activated status-only ruleset in
  reverse order and verify the original combined rulesets are unchanged.
- **Existing-ruleset update API failure:** keep the new status-only rulesets
  active and verify the existing combined ruleset remained unchanged. This may
  require the status twice, but it does not weaken protection.
- **Existing-ruleset verification mismatch:** while the matching new
  status-only ruleset remains active, compare the read-back with the exact
  update response. If they identify the same mutation and no concurrent edit,
  immediately restore the affected existing ruleset with its validated
  writable before-state payload. Verify the restored direct ruleset and
  effective branch rules before doing anything else.
- **Concurrent-policy indication:** if the final pre-update comparison or any
  post-update response indicates another editor changed policy, stop. Never
  overwrite the unexpected state with a rollback payload.
- **Unverified restoration:** declare a branch-protection incident, preserve
  all evidence, and halt. Do not continue to the other branch.

Full rollback is also fail closed:

1. Re-establish the exclusive maintenance window and require every live
   normalized payload and `updated_at` value to match a recorded rollout stage.
   Stop rather than overwriting any unexpected state.
2. Restore each existing ruleset from its saved complete before-state so its
   required status is active again.
3. Read back and verify the restored combined ruleset.
4. Disable the matching new status-only ruleset.
5. Re-read effective branch rules and prove the original protection is
   restored.

Do not delete rulesets during rollback, invent a success status, widen the
bypass role, or change `bypass_mode` to `always`.

## Verification and Evidence

Before and after each mutation, retain sanitized JSON that proves:

- ruleset IDs, names, targets, enforcement, and conditions;
- every rule and parameter;
- required status contexts and integration IDs;
- bypass actor type, ID, and mode;
- effective `main` and `dev` branch rules;
- authenticated actor and repository-administrator authority; and
- timestamps and API responses for each staged transition.

Verification must also prove:

- `5653432` still has deletion, non-fast-forward, and its original
  pull-request rule with no bypass actors;
- `19362594` still has its original pull-request rule with no bypass actors;
- each status-only ruleset has exactly one status rule and one PR-only
  administrator bypass actor;
- no workflow file or trusted publisher changed; and
- the working-tree documentation diff passes `git diff --check`.

Bandit and application tests are not applicable because the implementation
changes GitHub repository policy only and adds no executable repository code.
