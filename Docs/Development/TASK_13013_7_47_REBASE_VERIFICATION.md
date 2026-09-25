# TASK-13013.7.47 — September 25 rebase verification

PR: https://github.com/rmusser01/tldw_server/pull/2869

Rebased the 137 PR commits from `9a6440fb36a9f5b27de346a0068c3ab1b2ff6d93`
onto dev `a2f5e1b816cfe189db7f553a1ccf8d481dc2edbe`. The recovery ref is
`codex/task-13013-7-before-sep25-rebase`. The preservation audit found 380
branch-only and 2,380 upstream-only paths unchanged, with 125 equivalent
patches and 12 explained integration changes; no patches were dropped.

## Integration repairs

- Run production package and profile-data build checks through the installed,
  isolated virtual-environment interpreter.
- Keep backend package imports in the existing OCI identity-verified runtime
  smoke step; preserve its mandatory outcome and container restrictions.
- Restore owner-scope test coverage in four platform shards.
- Reconcile lock metadata without changing the 649 non-root uv package records;
  consolidate duplicate Bun aliases without adding package records.
- Remove a duplicate frontend guard declaration and combine Next experimental
  settings so build limits coexist with upstream proxy/cache settings.
- Flush three native Characters advanced-section clicks through RTL's
  `fireEvent`; retain subsequent user interactions, payload checks and 403 checks.

## Local verification

| Scope | Result |
| --- | --- |
| Admin UI | 149 files, 828 tests passed |
| Focused frontend | 144 tests passed |
| Frontend guard/configuration | 23 tests passed |
| Characters changed/adjacent cases | 5 passed, 94 intentionally filtered |
| Frontend and Admin type checks/lint | Passed; existing lint warnings remain |
| CI workflow contracts | 157 tests passed |
| Dependency-lock contracts | 19 tests passed, including red/green interpreter regression |
| Supply-chain suite | 1,392 passed, 5 opt-in skips initially; all 12 async-runner failures passed on rerun with pytest-asyncio enabled (14 tests in that module) |
| Actionlint | Four affected release/build workflows passed |
| Shell syntax | Nine workflow scripts passed |
| uv 0.12.7 lock check | Passed offline after metadata refresh |
| Bun 1.3.2 frozen lock check | Passed |
| Scoped Ruff/Black and diff whitespace | Passed |
| Scoped Bandit | No new production-code findings; test assertions only added, existing findings unchanged |

Logs are retained locally under `/private/tmp/task-13013-7-47-*`. The full
99-case Characters run was stopped after eight minutes and is not claimed
passing. An initial unrelated shared Admin subset was not used as evidence for
the required `admin-ui` suite above. Local tests do not prove signed release
attestation verification; that requires actual release evidence.

## Review and merge prerequisites

The five historical Qodo findings have fixes and existing resolved replies.
Their implementations were inspected again during this rebase. Fresh review
and CI on the published rebased head are still required.

The 342 canonical and 310 CI-only policy records were preserved byte-for-byte.
They expired on September 17. An inactive exact-identity renewal proposal
through October 2 is prepared outside the repository, pending requester
approval. This expiry is not evidence of newly exploitable vulnerabilities.
Existing Chroma dispositions remain unchanged.

The historical human Change summary says no exceptions or bypasses were
introduced. An updated human-written summary is requested under
`Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md` before
merge. This checkpoint does not claim current-head CI, fresh Qodo review, or
merge completion.
