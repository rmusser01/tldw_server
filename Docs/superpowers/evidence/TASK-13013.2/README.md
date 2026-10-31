# TASK-13013.2 Dev Required-Gate Evidence

TASK-13013.2 activated the additive `dev-core-required-gates` repository
ruleset and proved that a source-bound failing required check blocks a ready
pull request to `dev`.

## Result

- Ruleset `21824526` is active and targets only `refs/heads/dev`.
- It requires `backend-required`, `security-required`, `coverage-required`,
  `frontend-required`, `e2e-required`, and `container-build-check`.
- Every context is bound to GitHub Actions integration `15368`.
- `strict_required_status_checks_policy` is `true`.
- `bypass_actors` is empty and GitHub reports
  `current_user_can_bypass: never`.
- Existing ruleset `19362594` remained normalized-policy identical to its
  pre-change snapshot.

## Controlled Failure Proof

Temporary PR [#2835](https://github.com/rmusser01/tldw_server/pull/2835)
used exact head `6475cef3c03ef7c39bfe84f7f49030d568881e95`, a mechanical revert of
the known AuthNZ fixture correction. Five canonical required gates passed;
`coverage-required` failed from integration `15368` at
[Actions job 99167420781](https://github.com/rmusser01/tldw_server/actions/runs/33277727573/job/99167420781).
After the PR was marked ready, GitHub reported `mergeStateStatus: BLOCKED`.
Both applicable rulesets reported no bypass capability. The PR was then closed
unmerged and the remote proof branch was deleted.

## Evidence Files

- `dev-required-gates-before.json`
  - SHA-256 `cb96daac66afbd0b90e5e5a0703a44f7c6e255098eae3dff3a1e4c38868254fc`
- `dev-core-required-gates-payload.json`
  - SHA-256 `88be53a4d825bf6994b9f9cfceb43a33aacc2e3e313b8f1613e1e57306b0f2c3`
- `dev-required-gates-after.json`
  - SHA-256 `c03acd795035512e4592c39f2b5a4aabb35bb320c9584e5bdb66b9c8733b1775`
- `failing-check-proof.json`
  - SHA-256 `6283f107d2939d685ec4126baa4d063effd13a83da6a492d258927301ad30d4c`

All JSON files passed `python3 -m json.tool`. Live normalized comparisons
verified the new payload, preserved ruleset `19362594`, and the effective
seven-status `dev` policy.

## Rollback

If the canonical gate layer malfunctions, disable ruleset `21824526` without
deleting it. Do not modify or delete ruleset `19362594`. Capture the disabling
API response and the effective `dev` rules in this evidence directory and
reopen TASK-13013.2 before any replacement policy is activated.
