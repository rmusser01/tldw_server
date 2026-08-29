# TASK-13013.2 Required Dev Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every canonical required CI gate strict and non-bypassable on `dev`, with live evidence that a known failing required check blocks a pull request.

**Architecture:** Preserve the existing `frontend-license-gate-dev` ruleset byte-for-byte and add one independent `dev-core-required-gates` layer. GitHub aggregates the two layers: the existing layer continues to require a pull request and the trusted frontend-license status, while the new layer requires the six canonical GitHub Actions checks with strict/current-base enforcement and no bypass actors.

**Tech Stack:** GitHub repository rulesets REST API through `gh`, GitHub Actions check runs, Backlog.md CLI, Markdown evidence, Git.

**Spec:** `Docs/superpowers/specs/2026-08-21-core-release-readiness-program-design.md`

## Global Constraints

- The authoritative base is merged `origin/dev` commit `41bd5dda336c70259595ebf3ce3fb4a6a5b549db` or a later descendant verified immediately before the live change.
- Never update or delete ruleset `19362594`; its normalized JSON must remain unchanged.
- The additive ruleset targets only `refs/heads/dev`, has no bypass actors, and contains only the required-status-check rule.
- Each required context is bound to GitHub Actions integration `15368`; never fall back to an unbound source.
- Stop without mutation when the audited live state, check source, branch, or same-named ruleset differs from this plan.
- A failed verification disables, but does not delete, only the new additive ruleset.
- The proof branch and pull request are temporary and must never merge. Keep the PR draft until a required check fails; mark it ready only long enough to capture ruleset-derived `BLOCKED`, then close it.
- Do not install software or modify system files.

---

### Task 1: Freeze the live pre-change authority

**Files:**
- Create: `Docs/superpowers/evidence/TASK-13013.2/dev-required-gates-before.json`
- Modify: `backlog/tasks/task-13013.2 - Enforce-the-documented-required-gates-on-dev.md`

**Interfaces:**
- Consumes: live repository rulesets, effective `dev` rules, canonical check names, merged PR #2831 check runs.
- Produces: exact normalized pre-change evidence and a fail-closed authorization decision.

- [x] Capture ruleset `19362594`, the ruleset list, and effective `dev` rules with `gh api`.
- [x] Verify ruleset `19362594` is active, targets only `dev`, has no bypass actors, requires a pull request, and requires only `frontend-license-policy/trusted/dev` from integration `15368`.
- [x] Verify the six canonical checks on PR #2831 head `c336469210016866a9bb877d5ac473916180c47e` all concluded `success` and each came from integration `15368`.
- [x] Verify no ruleset named `dev-core-required-gates` exists; if one exists, require exact normalized policy equality or stop.
- [x] Record the complete pre-change JSON and SHA-256 digest in the evidence file.

### Task 2: Create the additive strict ruleset

**Files:**
- Create: `Docs/superpowers/evidence/TASK-13013.2/dev-core-required-gates-payload.json`
- Create: `Docs/superpowers/evidence/TASK-13013.2/dev-required-gates-after.json`

**Interfaces:**
- Consumes: the frozen Task 1 evidence.
- Produces: one active, dev-only, strict, no-bypass six-check ruleset.

- [x] Materialize this exact normalized payload:

```json
{
  "name": "dev-core-required-gates",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {
    "ref_name": {
      "exclude": [],
      "include": ["refs/heads/dev"]
    }
  },
  "rules": [
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          {"context": "backend-required", "integration_id": 15368},
          {"context": "security-required", "integration_id": 15368},
          {"context": "coverage-required", "integration_id": 15368},
          {"context": "frontend-required", "integration_id": 15368},
          {"context": "e2e-required", "integration_id": 15368},
          {"context": "container-build-check", "integration_id": 15368}
        ]
      }
    }
  ]
}
```

- [x] POST the payload once with `gh api --method POST repos/rmusser01/tldw_server/rulesets --input <payload>`.
- [x] Read the created ruleset back and require normalized equality with the payload.
- [x] Re-read ruleset `19362594` and require byte-for-byte normalized equality with the Task 1 snapshot.
- [x] Read effective `dev` rules and require all seven statuses: the trusted frontend-license status plus the six canonical gates, with the canonical layer strict.
- [x] On any mismatch, PUT only the new ruleset to `enforcement: disabled`, record the response, and stop. No mismatch occurred.

### Task 3: Prove a failing required gate blocks merge

**Files:**
- Create: `Docs/superpowers/evidence/TASK-13013.2/failing-check-proof.json`

**Interfaces:**
- Consumes: the active additive ruleset, known failing pre-fix commit `237ea28e3d01dd79252013df0012ddd3734c8ace`, and its focused correction commit `c336469210016866a9bb877d5ac473916180c47e`.
- Produces: a temporary draft PR whose required failure and blocked merge state are captured without merging.

- [x] Verify commit `237ea28e3d01dd79252013df0012ddd3734c8ace` is the parent of correction commit `c336469210016866a9bb877d5ac473916180c47e` and its hosted `coverage-required` check failed from integration `15368`.
- [x] Create temporary branch `codex/task-13013-2-ruleset-proof` from current `dev` and revert only correction commit `c336469210016866a9bb877d5ac473916180c47e`; this preserves the repaired workflow admission while reproducing the known AuthNZ coverage failure.
- [x] Verify the proof branch differs from `dev` only by that mechanical revert, then push its exact head.
- [x] Open a draft PR to `dev` titled `test: prove required dev gate enforcement` and label its body as non-mergeable temporary proof.
- [x] Wait only for one canonical required check to conclude failure; do not rerun or repair it.
- [x] Verify the exact PR head remains fixed, mark it ready after the failure, then require `mergeStateStatus: BLOCKED`, integration `15368`, and no bypass on either applicable ruleset.
- [x] Capture PR URL, head SHA, failing check, Actions URL, applicable rules, merge state, and bypass state.
- [x] Close the proof PR and delete only `codex/task-13013-2-ruleset-proof` from the remote.
- [x] Verify the PR is closed/unmerged and the proof branch no longer exists.

### Task 4: Publish the canonical policy and close the task

**Files:**
- Modify: `Docs/Development/CI_REQUIRED_GATES.md`
- Create: `Docs/superpowers/evidence/TASK-13013.2/README.md`
- Modify: `backlog/tasks/task-13013.2 - Enforce-the-documented-required-gates-on-dev.md`

**Interfaces:**
- Consumes: Tasks 1-3 evidence.
- Produces: current canonical documentation, auditable ruleset identifiers, rollback instructions, and a complete task record.

- [x] Replace the stale pending-enforcement prose with the exact active ruleset name/ID, strict policy, integration binding, no-bypass policy, and retained frontend-license layer.
- [x] Document that the effective approval policy remains the existing dev pull-request rule: zero required approving reviews, stale reviews dismissed, extra approval required for unattributed changes, and no ruleset bypass actors.
- [x] Summarize the before/after and controlled proof in the evidence README without credentials or transient API tokens.
- [ ] Mark all TASK-13013.2 acceptance criteria and Definition of Done items complete; record Bandit as not applicable because only Markdown/Backlog records changed.

### Task 5: Verify and publish the closeout branch

**Files:**
- Test: all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: committed documentation/evidence/task changes and the active live policy.
- Produces: one reviewable closeout PR based on current `dev`.

- [ ] Validate every evidence JSON file with `python3 -m json.tool` without installing dependencies.
- [ ] Re-run the live normalized ruleset comparisons and effective `dev` rule assertions.
- [ ] Run `git diff --check` and scan the changed files for credentials or private infrastructure details.
- [ ] Commit the exact documentation, evidence, and Backlog scope.
- [ ] Push `codex/task-13013-2-required-gates` and open a PR to `dev` with the ruleset/proof evidence.
- [ ] Verify the PR head, base, changed paths, and hosted required-gate status; do not merge without a separate user decision.
