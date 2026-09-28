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

## Follow-up after current-head CI

The requester supplied an updated human Change summary, which was posted
verbatim to PR 2869. That merge prerequisite is satisfied.

CI on `2e1fc271f8` completed with source and all five image admission jobs
rejecting `TASK-13013.7.23-OS-01` for `expires_on`. These are the previously
reported approval-expiry failures. Renewal remains pending. Two optional
historical native candidate jobs also failed when their fixed producer
artifacts returned HTTP 404; they are not required merge checks.

The two Characters harness configurations each passed 111 tests and failed
two, covering three unique edit-form cases. Failures occurred while opening
advanced fields or Metadata, before their payload assertions. The isolated
three-case local run passed even before this repair, consistent with an
intermittent state-flush failure. Six native button interactions now use
RTL's act-wrapped `fireEvent.click`, matching the existing successful form
tests. Assertions, timeouts and product code are unchanged.
All three affected cases pass after the repair in both the WebUI and shared
UI configurations (110 other tests intentionally filtered in each run).
The full harness requires fresh CI confirmation.

Dev advanced to `59bd5845038342013a2d84d0130f6164f14b54fd`. The second rebase
completed without conflicts; range-diff reports all 138 patches equivalent.
The recovery ref is `codex/task-13013-7-before-summary-rebase`. The pinned
offline uv check still passes. Scoped ESLint reports zero errors and the
same 121 existing warnings. No Python code changed in this follow-up, so
Bandit is not applicable to its TypeScript test and documentation edits.

## September 27 integration

Rebased all 139 patches onto dev
`3c9d97c56b29abc4c0396274b9560859aee06959`, with recovery ref
`codex/task-13013-7-before-sep27-rebase`. Range-diff maps 132 equivalent
patches and seven conflict adaptations, with no dropped patches. All 398
branch-only and 1,096 upstream-only paths retain their exact tree entries.

Workflow conflicts preserve dev's PR license wait and the release scan gates.
The obsolete SBOM workflow-run trigger was removed to match dev's intended
event routing; direct PR and reusable release calls remain. Duplicate shard
entries were removed without dropping distinct tests. Dev's explicit
virtualizer 3.13.19 pin and schema compatibility assertion are retained.

The current-head Characters harness failures are resolved in GitHub CI.
The newly exposed 15 shared React hook-rule errors are repaired without
disabling rules: render-time clocks move to existing state/timer flows,
operation callbacks retain operation-time clocks, draft timestamps belong to
the write, and the lazy folder picker is declared at module scope. Deadline
timers clean up on unmount; persona expiry handles fractional deadlines and
rechecks after early timer execution. Missing prompt creation dates use the
existing unknown-date convention rather than inventing the current time.

Validation: 132 workflow/dependency contracts pass; pinned uv and frozen Bun
lock checks pass; five integration workflows pass Actionlint; frontend type
check passes. Quiz coverage passes 37 tests and callback/TTS coverage passes
26 tests; ingest/watchlist coverage passes 25 tests plus the persona expiry
regression. Scoped Bandit on the merged Python schema test reports only its
three ordinary assertion findings, with no parse errors. Independent review
found the fractional-expiry case, which was corrected and verified by that
reviewer before publication.
Local logs use `/private/tmp/task-13013-7-sep27-*`, `/tmp/quiz-time-*`, and
`/private/tmp/sep27-hook-*`.

The complete shared-hook gate passes across 5,405 files with zero gate
failures. It separately reports 1,375 other ESLint errors outside its three
compiler-rule scope; this is not a claim that all repository lint is clean.

The human summary remains satisfied. Renewal approval remains pending;
neither canonical nor CI exception records were extended. Fresh CI and Qodo
review are still required before merge.

### Fresh Qodo findings

Qodo's September 28 UTC review identified two release-evidence validation
gaps. Ten negative regression cases reproduced acceptance of fabricated
decisions, missing or inexact approvals, and future scanner metadata.
The image evidence loader now recomputes retained decisions through the
existing exact-match policy evaluator using the current UTC admission date.
The scanner check also bounds scan start by verification time. Both assembly
and verification use these shared checks; valid exact approvals retain their
existing format. Independent review found no remaining issues in this fix.
This change does not renew, add, or broaden any real exception record.
Final verification passes 102 tests, including tampered evidence with
recomputed checksums and clock boundary cases. Ruff, touched-range Black,
direct-script CLI and diff checks pass; production Bandit reports zero
findings. Logs: `/private/tmp/sep27-release-final.log` and
`/private/tmp/bandit_sep27_release.json`. Retained-file consistency does not
replace existing trust in scanner execution or signed provenance.

## September 28 approval renewal and fresh Qodo review

After clarification of the two scopes, the requester said “ok approved” to
renewing the same 342 canonical dispositions and 310 app/audio CI-only
risk-acceptance records through October 2. Both policies now contain new
`-r20260928` records superseding their prior IDs, created September 28 UTC.
Only IDs, supersedes links and dates changed; exact matching fields, owner,
approval PR, rationale and mitigation are preserved. This does not reclassify
the CI-accepted findings as fixed or non-applicable.

The nine renewal cases failed before the change. The affected historical,
renewal, exception-evaluator and CI-bypass suites pass 674 tests. Historical
tests now read the identical canonical-policy bytes already retained in the
TASK-13013.7.45 archive; their original dates and preservation assertions remain.
The new tests check all 652 renewed records against the two archived policy
hashes, reject changed identities and dates, and replay both archived complete
reports: canonical admission retains 155 blockers per app/audio image; combined
CI admission excepts 235 rows with zero unmatched approvals. This is retained
report replay, not a claim of a fresh current-head image scan. Fresh CI remains
a prerequisite for merging and will reject new, changed or stale findings.

Fresh Qodo feedback also led to exact SBOM digest matching (62 release tests),
launcher/acquisition annotations with the dependent launcher checksum updated
(37 launcher/provenance tests), completion-time Watchlists sampling (22 focused
frontend tests), and neutral sampler normalization in the existing shared core
(41 capability and Characters cases). Targeted regressions failed before their
fixes. Source Bandit scans report zero findings; the new renewal test contains
only ordinary pytest assertion findings. No native candidate rebuild was run.

The admin-label translation suggestion is inapplicable to the standalone
English-only admin-ui: its layout uses `lang="en"`, its providers contain no
translation provider, and its package/configuration has no i18n dependency or
catalog. No new translation framework was added.

Automatic approval review rejected posting the detailed approval record to
PR 2869 as an external disclosure without destination-specific authorization.
The requester subsequently explicitly approved publication to PR 2869. The
[approval record](https://github.com/rmusser01/tldw_server/pull/2869#issuecomment-5862294181)
is now posted. The verified renewal is authorized for publication; current-head
CI and final review remain required before merge.

## Final September 28 rebase

Dev advanced to `bd2ae757d274e7eda3edb48eb682733c401efe56` with capability
disclosure authentication changes. Rebased without conflicts; range-diff
confirms all 144 patches equivalent, and all 13 new upstream paths are byte
identical to dev. Recovery ref: `codex/task-13013-7-before-sep28-rebase`.
The nine renewal regressions pass after rebase. Qodo reported zero active
findings on the preceding published head; fresh required CI and final review
are being obtained for the rebased head.

## Final review follow-up

The next Qodo pass raised five items. Restored the documented Dormant badge
for missing/empty/invalid login timestamps while retaining safe parsing and
the strict 90-day boundary (24 users-page tests pass). Added provenance factory
and hook annotations/docstrings (40 tests pass; Bandit zero findings).

Two suggestions were disproved against the actual execution environment:

- Docker runtime identity: the workflow explicitly enables the containerd image
  store. Its image ID is the OCI target manifest/index digest, as confirmed by
  [Moby's implementation](https://github.com/moby/moby/blob/v28.5.2/daemon/containerd/image_inspect.go#L85-L89)
  and read-only Docker 29.2 inspection (ID equals OCI index descriptor digest).
  The proposed config-ID substitution would be incorrect for this store.
  Existing subject-to-platform-to-config hashing remains enforced; 58 runtime
  and CI contracts pass, with Actionlint clean and Bandit zero findings.
- The Expat metadata helper is a standalone bootstrap CLI copied into a bare
  Python image without Loguru. Its output is captured by `run_step` into retained
  logs, and `python -S .../expat_candidate.py --help` succeeds without site
  packages. Adding the suggested application logging import would break that
  dependency-free path. No dependency or logging change was made.

## Current scanner refresh: frontend PCRE2

The current-head Admin UI report rejected three newly reported PCRE2 identities
(CVE-2026-86145, CVE-2026-89157 and CVE-2026-89161), all for
`libpcre2-8-0=10.42-1`. All 56 existing dispositions matched with zero stale
records. This establishes an admission failure, not application exploitability;
CVE-2026-89157 explicitly requires a 32-bit platform, unlike the admitted amd64
images. Debian records `10.42-1+deb12u1` as fixed for all three:
[86145](https://security-tracker.debian.org/tracker/CVE-2026-86145),
[89157](https://security-tracker.debian.org/tracker/CVE-2026-89157),
[89161](https://security-tracker.debian.org/tracker/CVE-2026-89161).

Both Bookworm frontend runtime stages now install only that exact PCRE2 update
through authenticated APT and assert the installed version. The Ubuntu candidate
renderer removes only this verified Debian-specific block when changing distro,
rejecting drift. Existing base pins, builders, runtime users and gates remain.

A minimal linux/amd64 build from the exact pinned Node base succeeded. A complete
dpkg before/after comparison showed only `libpcre2-8-0:amd64` changed; APT reported
one upgrade, zero installs and zero removals. Offline Node 24.20.0 linux/x64,
Node SHA-256, dependency resolution and PCRE2-backed grep checks passed. All 49
focused contracts pass; Ruff, scoped Black, Bandit and independent review are
clear. This package-layer validation is not a full application image scan; fresh
CI admission remains required. No exception policy changed.

CI on `b2738e447d` confirms the required source security gate passes with the
renewed policy. Its full container gate is blocked by the above scanner rows.
