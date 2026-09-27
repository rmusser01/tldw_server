# Task 10 independent review — capability enablement and Track B release gate

## Verdict

**APPROVED. No Critical, Important, or Minor finding is present in
`05a0c8b12a..32bc90ac45`.**

The range performs the approved release action: it changes only
`single_text_recipe_v2.supported` in production, from `false` to `true`, keeps
the advertised limits sourced directly from `SINGLE_TEXT_RECIPE_LIMITS`, and
updates the two exact capability expectations. The fourth changed file is the
Backlog evidence record, which remains In Progress pending this review.

## Four-file scope audit

- `tldw_Server_API/app/api/v1/endpoints/prompts.py` changes one boolean in
  `get_prompt_capabilities`. The route, authentication, persistence
  authorization, rate-limit dependency, schemas, and limits expression are
  otherwise byte-for-byte unchanged.
- `test_prompts_structured_api.py` renames the pre-release capability test and
  changes only its `supported` expectation to `true`; it still exact-compares
  the response limits to `dict(SINGLE_TEXT_RECIPE_LIMITS)` and separately
  requires create/update authorization.
- `test_prompt_improvement_api.py` similarly renames the combined capability
  test and changes only its recipe support expectation. Its exact Track A
  limits and persistence-authorization assertions remain intact.
- `task-12984.2 - Implement-single-text-structured-prompt-recipes.md` records
  the reproducible Task 10 evidence and deliberately retains `status: In
  Progress`, unchecked acceptance criteria, and the implementation-plan link
  until controller finalization.

`git diff --name-only` found no changed frontend, package, dependency,
lockfile, global chat-scope, owner-contract, layout, or Improve-button
placement path. The only production path in the range is `prompts.py`, and its
only production hunk is the one support boolean.

## Contract and compatibility audit

- Centralized limits remain coupled to the authoritative validator constants:
  the response still constructs `limits=dict(SINGLE_TEXT_RECIPE_LIMITS)`.
- Capability support and write authorization remain independent. Fresh tests
  covered admin, `system.configure`, and ordinary-user authorization results,
  plus unauthenticated rejection.
- The support flip does not bypass the existing expected/actual recipe owner,
  credential-revision, direct/background dispatch, uncertainty, or write
  authorization paths. Fresh owner/surface tests exercised those unchanged
  paths after rollout.
- Old-server, unknown-server, offline, and future-schema behavior remains a
  frontend/transport concern and no relevant source changed. Fresh scoped unit
  and real browser journeys still prove local Apply with persistence disabled
  for old/unknown/offline states and v3 quarantine.
- The approved Task 9 Web and packaged-extension payload contracts remain
  byte-for-byte untouched.

## Fresh verification

All positive commands below ran against `32bc90ac45` before this review artifact
was created.

- Focused capability and authorization matrix: **6 passed**, including both
  changed capability expectations, all three persistence authorization cases,
  and real-authentication rejection.
- Independent RED/GREEN equivalent using the endpoint function itself:
  the new `supported is True` contract raises `AssertionError` at detached base
  `05a0c8b12a`; the identical probe exits 0 at `32bc90ac45` and prints the exact
  eight centralized recipe limits.
- Representative backend schema/render/persistence/API boundary matrix:
  **745 passed / 2 unrelated pytest cleanup warnings** in 212.69 seconds.
  This covered the v2 validator and renderer, v1/v2 prompt database behavior,
  structured API, and runtime/future/mistagged persistence boundaries.
- Scoped frontend capability, library, builder, real surface, and owner
  contract matrix using the repository-pinned Vitest 4.0.18 binary:
  **7 files / 251 tests passed**.
- Web `/chat` Playwright gate: **6 passed / 1 expected real-backend-config
  journey skipped**. The six fixture-backed recipe journeys covered the
  released, old, unknown, offline, mobile, accessibility, and v3 paths.
- Packaged-extension Playwright gate: **6/6 passed**, covering the sidepanel,
  `/options.html#/chat`, old/unknown/offline behavior, v3 quarantine, and the
  current Prompt Workspace.
- Extension TypeScript compile: exit 0.
- Web production compile: Next webpack compilation, all **154 pages**, and
  shared-token verification succeeded; the command then reached the inherited
  app-shell budget failure at **683.0 KB gzip versus 600.0 KB**.
- Ruff and Python compileall over the changed and production release scope:
  exit 0.
- Bandit exact Task 10 plan scope wrote
  `/tmp/bandit_task_12984_2_task10_review.json`: exit 0, **5,235 LOC, zero
  findings, zero errors, one unchanged B608 skipped test, no suppression**.
- `git diff --check 05a0c8b12a..32bc90ac45`: exit 0. Worktree status was clean
  before this ignored artifact was added.

An initial unconfigured apps-root `bunx vitest` attempt selected Vitest 5.0.0
without the UI alias config and collected no tests; it is not counted. The
authoritative rerun above used the checked-in frontend config and pinned 4.0.18
binary. It left the worktree unchanged.

## Baseline-failure determination

### Shared Web bundle budget

This failure is not introduced by Task 10 and does not block this rollout
range. Git reports the exact same `apps` tree object at base and head:

```text
05a0c8b12a:apps  7d0e78fe539b98a2debd31a4472f46f0f1aee32d
32bc90ac45:apps  7d0e78fe539b98a2debd31a4472f46f0f1aee32d
```

Therefore every Web source, test, build script, package manifest, and lockfile
consumed by the budget gate is byte-for-byte identical across the range. The
fresh head build independently reproduced the reported 683.0 KB result after a
successful production compile and token-sync check. A detached-base build
attempt used symlinked dependencies and stopped earlier on invalid module
resolution, so it is deliberately not counted as size evidence; the identical
Git tree is the stronger range-introduction proof. The global app-shell budget
remains real repository debt, but a one-line Python capability response cannot
change it.

### Capability catalog rate limit

This failure is also byte-for-byte pre-existing and does not block Track B.
The focused test fails identically at base and head: the second request returns
200 rather than the expected 429. The rate-limit test body has the same SHA-256
at both revisions
(`a4aebb024fec936d427b9844bad0359138aa0949c8591e2881c85765cbc3070c`),
and `auth_deps.py` has the same Git blob at both revisions
(`77710d604520e3de3753172b682963dd23ed9962`). The only changed endpoint token
is the returned support boolean; dependency ordering is unchanged.

The defect should be tracked separately: the route-level rate dependency runs
before `get_auth_principal` populates `request.state.user_id`, so the read-only,
authenticated, low-sensitivity capability endpoint skips catalog enforcement.
It does not bypass recipe create/update authorization or owner enforcement, and
Task 10's approved one-boolean scope should not absorb that unrelated behavior
change.

## Remaining baseline limitations

- The shared Web app-shell budget remains 83.0 KB over its repository threshold.
- `test_prompt_capabilities_catalog_rate_limit_returns_429` remains a genuine
  pre-existing failing test and merits a separate auth/rate-limit task.
- The fixture-backed browser gates do not replace the skipped Web journey that
  requires a configured live backend; the release still has fresh Web and
  packaged-extension recipe E2E coverage.

None of these limitations is introduced by `05a0c8b12a..32bc90ac45`, weakens
the recipe write/owner boundary, or requires widening the audited capability
rollout commit.
