# Real PST validation and dev integration (TASK-13377, 2026-09-26)

Both previously optional PST endpoint tests now execute and pass with the native
`pypff` module from `libpff-python==20231205`. The fixture is a real PST container
containing four public fabricated body-format messages, rather than private mail.
Native conversion now preserves datetime values and fills missing recipients,
sender, date and Message-ID from selected transport headers. Native fields take
precedence; MIME encoding headers are not copied onto the reconstructed body.

## Fixture and parser provenance

The Apache Tika fixture `testPST_variousBodyTypes.pst` is pinned to upstream commit
`e3c6b6b18537100a7016b55cb8d29fa06cf4233a`:
[download the pinned public fixture](https://raw.githubusercontent.com/apache/tika/e3c6b6b18537100a7016b55cb8d29fa06cf4233a/tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-microsoft-module/src/test/resources/test-documents/testPST_variousBodyTypes.pst).
It is 271,360 bytes with SHA-256
`24c5e6bbb8bf26a817c977283e40e7b69d2661fec0845abbe177f97efcb05fb0`.
The parser was installed into `/tmp/tldw-real-pst-13376-python` with `--target` and
`--no-deps`; the shared project virtual environment was unchanged. The optional
module reports libpff version `20231205`. The fixture binary is not committed.

The [machine-readable evidence](evidence/email_core_closeout_13376/real_pst_13377.json)
retains fixture provenance, exact outcome node IDs, the diagnostic guard plugin
source and tested production/probe source hashes. The final guarded run passed
all ten cases with zero non-loopback connection/DNS attempts and zero analysis
calls. Guards fail the session even if production code swallows an attempted call.

## Failures corrected

The enabled native baseline was one pass and one failure: recipients and date
were empty. Eight metadata regressions failed before the production fix, then
passed alongside the two existing native endpoint cases. The native API exposes
transport headers rather than recipient convenience methods, and its delivery
time is a datetime that the prior string-only getter discarded. The full endpoint
and metadata rerun passed 24 cases. The missing-parser test now explicitly makes
`pypff` unavailable so its outcome does not depend on installed optional packages.

Integration with current `dev` at
`f5fa1f3a41855aa02871d8b76d0ec0cebbaf9e07` resolved thirteen conflicts while
preserving dev's literal FTS fallback, cancellation propagation, executor context,
UUID health probe and PostgreSQL startup validation. Review also retained email
identity, scope, archive transaction, attachment and bounded logging behavior.

The first combined integration run passed 141 cases with thirteen shared-fixture
setup errors. Registering the imported fixture locally fixed collection. A focused
rerun then passed 23 cases and exposed one real isolation regression: cached auth
restoration overwrote a validated selected organization, so quota checked the
second organization while persistence reused the first organization's identity.
A request-local selection receipt tied to a deep copy of the original principal
preserves only the validated org/team selectors. Canonical owner, memberships,
admin authority and stale-scope/session-role protections remain intact. Six new
cases cover both cached boundaries, unvalidated state and changed claims: the
baseline was two failures/five passes; the corrected authority and authenticated
integration suite passed all 63 cases. Independent review found no blockers.

## Verification

| Scope | Result |
| --- | --- |
| Final combined ingestion, attachment, logging, cursor, native PST and auth regressions | 201 passed; zero failures/errors/skips |
| Final native PST endpoints plus metadata regressions | 10 passed; zero skips; guards zero |
| Postmerge identity, persistence, cursor, FTS, archive and portable bootstrap regressions | 110 passed |
| Official isolated PostgreSQL identity, search, archive transaction and scope regressions | 54 passed |
| Corrected cached-authority and authenticated upload/access suite | 63 passed |
| Actual PR Python files | 90 compiled |
| Ruff: 47 production/probe paths and four touched test files | Zero findings |
| Bandit: 47 production/probe paths | Zero new findings/errors; eight inherited findings |

Bandit's eight B105/B106 findings flag the unchanged `access`, `api_key` and
`service` token-type labels in the two auth files. Exact filenames, rules, issue
text and flagged source match current dev after removing shifted numeric line
prefixes. No credential literal was introduced. Counts above overlap and should
not be added into a unique-case total. Pytest reported dependency deprecations and
cleanup warnings for pre-existing temporary directories; those unrelated resources
were preserved.

## Reproduction

Activate the project virtual environment, install the optional parser into a
fresh temporary target, download the pinned fixture and verify its hash:

```bash
source .venv/bin/activate
python -m pip install --target /tmp/tldw-real-pst-parser --no-deps 'libpff-python==20231205'
curl -fL 'https://raw.githubusercontent.com/apache/tika/e3c6b6b18537100a7016b55cb8d29fa06cf4233a/tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-microsoft-module/src/test/resources/test-documents/testPST_variousBodyTypes.pst' -o /tmp/tldw-real-pst-fixture.pst
shasum -a 256 /tmp/tldw-real-pst-fixture.pst
PYTHONPATH=/tmp/tldw-real-pst-parser:$PWD \
PST_FIXTURE_PATH=/tmp/tldw-real-pst-fixture.pst \
CHAT_FORCE_MOCK=1 AUTO_DOWNLOAD_MODELS=false \
CONNECTORS_WORKER_ENABLED=false EMAIL_GMAIL_CONNECTOR_ENABLED=false \
python -m pytest \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_pst_metadata_13377.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_process_emails_endpoint.py::test_process_emails_endpoint_pst_with_pypff_extraction \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_process_emails_endpoint.py::test_process_emails_endpoint_pst_recipients_and_date_strict -q
```

To reproduce the additional diagnostic guard receipt, extract `guard_plugin.source`
from the evidence JSON to `/tmp/email_pst_guard_plugin_13377.py`, use the original
fixture path `/tmp/tldw-real-pst-13377-synthetic.pst`, prepend `/tmp` to PYTHONPATH,
and add `-p email_pst_guard_plugin_13377` to pytest. It writes
`/tmp/email_real_pst_guard_receipt_13377.json`. Temporary parser and fixture resources
are removed after verification; credential-free evidence remains in this repository.

## Scope and release evidence

Actual OST files, live Gmail/OAuth provider behavior and staging sync lag remain
unverified. No personal account, private mailbox or downstream model processing
was involved. The earlier 328-pass/two-skip snapshot remains historical; this
follow-up closes its real-PST fixture gap.

The [core closeout report](Email_Core_Closeout_Validation_2026-09-26.md) and its
SQLite/PostgreSQL throughput and million-message JSON certificates keep their
original source revisions. Those benchmarks were not rerun for this merged source.
At this initial validation, the PR targeted `dev` and remained draft. The requester's supplied `Change summary`
is retained verbatim in its description. Creating the PR is not the separate
owner release approval.

Published [draft PR #3023](https://github.com/rmusser01/tldw_server/pull/3023) against `dev`; GitHub confirms it is mergeable.
Tested merge commit: `650fc72a98822233dd1a6fb53bf8d62bdb0e6cd9`. The subsequent
closeout commit changes documentation, evidence and task tracking only. Cleanup
removed the exact temporary parser target, both downloaded PST fixtures, upstream
test source and diagnostic plugins/baseline copies; the shared venv and PostgreSQL
service were preserved. Cleanup receipts are in the evidence JSON.

## PR3023 CI follow-up (2026-09-27)

Current dev `bd2ae757d274e7eda3edb48eb682733c401efe56` is integrated.
The reviewed merge preserves dev's typed database errors, canonical admin
permissions and injected persistence collaborators alongside the email isolation
and bounded logging fixes. Tested source: `0176b267b07e5209fe5f9d9873da55d449f4fbc8`.

The docs gate lacked the generated attachment-policy mirror. Publishing that page
in a regular commit also supplies the creation history omitted by the date
plugin's filtered logs for a merge-only addition. The backend gate failed on
OpenAPI drift; the canonical frontend generator refreshed both types and the
fingerprint. Mypy's NumPy parsing message was advisory, not the failed gate.

| Current-source verification | Result |
| --- | --- |
| Combined email/auth/backend regressions | 236 passed; zero failures/errors/skips |
| Core identity/persistence/search regressions | 110 passed |
| Official PostgreSQL and dev integration targets | 80 passed; one intentional SQLite-only parameter skip |
| Guarded native PST endpoints and metadata | 10 passed; zero skips/outbound/model calls |
| Docs with CI Material 9.7.7/revision-date plugin 1.6.0 | 212 passed; strict build passed with zero counted warnings |
| Public/private, command and endpoint docs boundaries | All passed |
| OpenAPI fingerprint | Stable; 2,104 paths and 3,245 schemas |
| Compile/Ruff/Bandit | 90 Python files compile; 47 production/probe paths lint clean; zero new security findings/errors |

The [follow-up evidence](evidence/email_core_closeout_13376/pr3023_ci_followup_20260927.json)
records source hashes, exact native outcomes and toolchain versions. Suites
overlap. The eight inherited Bandit findings still match current dev exactly.
Dependency and pre-existing temporary-directory cleanup warnings are unchanged.
Earlier benchmark certificates remain historical and were not rerun. The
requester's supplied `Change summary` is preserved verbatim in PR3023.

A later dev update, `de7f453593dbb40f069a4666fd562fc5f3622817`, adds scheduled-task message storage.
Its conflict-free integration preserves all 47 validated email/probe files and
all docs-gate inputs byte for byte. The API fingerprint still matches, and
51 existing tests for the new dev changes pass without failures/errors/skips.
The earlier email/PST/docs results above retain their original tested-source binding.


## Supported OpenAPI dependencies (2026-09-28)

The hosted docs gate passes at `d38f1dae3043bd14e3bab8a92d98bf96017ad2a4`.
The remaining backend drift exposed Pydantic 2.11.7 in the earlier local
environment, below the project's declared minimum 2.13.5. An isolated temporary
target matching CI's schema dependency versions reproduces its exact hash
`99862cf358fdd2540162c2d9e9202d3eb5a4bb33e9ea35c52e03dbc6095124ba`: 2,104 paths
and 3,244 schemas. The canonical generator now emits that fingerprint and
matching frontend types. Two identical OSCE input/output definitions consolidate
into one with three updated references; all endpoint paths and the normalized
full document are otherwise unchanged. Historical local results above retain
their original environment and source binding.

Broader validation also exposed a pre-existing pagination diagnostic false
positive: unmounted legacy Admin webhook models borrowed current components
with the same short names. The diagnostic now uses the existing effective-route
walker to match mounted endpoint identities first. Regression cases verify both
retired-name isolation and rejection of missing pagination on mounted endpoints.
The final combined API/pagination suite passes all 84 cases with zero failures
or skips. Its focused five tests, 76 OSCE schema tests and 219 dev frontend
regressions also pass.
Both real-PST endpoints plus metadata regressions also pass under the supported
dependencies: ten cases, zero skips and zero guarded outbound/model attempts.
All 91 current PR Python files compile, the changed test passes Ruff, and its
four inherited Bandit test-assert findings match HEAD with zero new findings.

Dev `97da6c2dfec239b5739eed4116469f1fb5db7ed5` integrates without conflict; all
47 certified email/probe files remain unchanged. Independent review found no
blockers in either correction. Dependency requirements, production APIs and CI
gates are unchanged; the shared environment is preserved. No ADR is required.
Details and log hashes are appended to the existing follow-up evidence.


## Dev compute authentication refresh (2026-09-28)

Hosted backend, OpenAPI drift and docs gates passed on `d40beae3839d5d72f7aafc24513b7f58e6f085f8`.
Dev then advanced to `a2d5b1c0db789e9db7d4820e05164028ea4734ec` with authentication
fixes for five compute routes. The sole merge conflict was the generated
fingerprint. The unchanged canonical generator and supported CI dependencies
produce `58fa00440a664a47cb9ad6e72350b2515a7e6ac061e81a13aca032fe803a3b47`:
2,104 paths and 3,244 schemas. Only those five path definitions change; all
components remain identical. Incoming Python files match dev byte for byte.

Dev's generated route inventory also omitted two unchanged public audio probes.
Canonical regeneration restores exactly `/api/v1/audio/stream/status` and
`/api/v1/audio/transcriptions/health`, retaining the five compute-auth removals.
Passive status remains public; model warm-up still requires an authenticated admin.
The unchanged helper now matches all 178 anonymous-capable routes exactly.

| Refresh verification | Result |
| --- | --- |
| Existing compute endpoint regressions | 21 passed; zero failures/skips |
| API/pagination contracts | 84 passed; zero failures/skips |
| Public audio, warm-up authorization and route diagnostic tests | 25 passed; zero failures/skips |
| OpenAPI drift and route-auth inventory checks | Both passed |
| Compile and generated fingerprint formatting | 91 PR Python files compile; formatting passes |
| Bandit on eight inherited Python paths | Zero new findings/errors; all 94 findings belong to identical dev source |

All 47 certified email/probe files remain unchanged. Earlier native PST and
benchmark receipts retain their tested-source binding; neither was rerun for this
refresh. Independent review found no blockers. Exact temporary schema dependencies
and wrapper were removed, preserving shared installations and credential-free
receipts. New-head hosted checks will rerun after publication.


## Latest-dev rebase and Qodo corrections (2026-09-29)

PR3023 is ready and rebased onto dev `0da68530e80c713ed3a323a741998e1fed37e3e9`,
with the prior branch history retained in `codex/email-pr3023-pre-rebase-20260929`.
The shared logging conflict preserves dev's URL display helpers and email privacy.
The requester explicitly authorized addressing review comments and merging after
required checks pass; the supplied Change summary remains verbatim.

All nine Qodo comments have concrete corrections or evidence. Preparation, search
and detail errors retain bounded function/line diagnostics in ordinary logs,
without exception values, filenames or locals. Preparation failures return a
fixed safe message with an opaque ID matching the log. The MIME fallback warning,
attachment helper contracts and test classification/formatting are corrected.
Duplicate quota DDL is removed because dev already provides it. Actual SQLite
boolean matching and ranking disprove the alias concern without LIKE fallback.
Read-key upload rejection remains intentional; the runbook documents creating a
read/write key for connected ingestion and replacing existing read-only keys.

| Rebase/review validation | Result |
| --- | --- |
| Rebased ingestion/auth/core suite | 201 passed; zero failures/errors/skips |
| Guarded native PST endpoints/metadata | 10 passed; zero skips/outbound/model attempts |
| Affected review regressions | 167 passed; zero failures/errors/skips |
| Final ordinary-log rendering and preparation response checks | 4 passed |
| Official PostgreSQL quota bootstrap/failure propagation | 3 passed; zero skips |
| Compile/Ruff/Bandit | 93 PR Python files compile; 13 touched paths lint clean; six touched production paths have zero findings/errors |
| Canonical OpenAPI | Unchanged fingerprint `58fa00440a66`; 2,104 paths / 3,244 schemas |

The 167-case run preceded the final message-only diagnostic rendering change;
the four final cases verify those messages and the safe client response. Suites
overlap. Independent review found no blocker. Detailed source and log hashes are
appended to the existing follow-up JSON. Earlier benchmark certificates retain
their original source revisions. Required hosted checks and incremental review
will be checked on the published correction head before the authorized merge.


## CI admission dev refresh (2026-09-29)

Dev advanced to `6110d2ae436c805c3beeda8f84427b4890534ddf` with PR3036's
CI admission, database and fixture corrections. The rebase overlaps four files.
It retains dev's live-thread SQLite shutdown guard with bounded error logging,
the default file-validator fallback and the offline quota fixture. The pagination
diagnostic retains the existing shared route walker and mounted/retired tests.
Independent review found no actionable regression; other incoming files match dev.

The refreshed combined suite passes **374 cases** with zero failures/errors/skips,
including core email, privacy, offline upload/search, pool and pagination checks.
All **ten guarded native PST cases** pass again with zero skips/outbound/model
attempts. **92** current PR Python files compile; all four resolution paths pass
Ruff and both production resolution files have zero Bandit findings/errors.
Canonical generation and the exact hosted OpenAPI drift check pass with the
unchanged `58fa00440a66` fingerprint: 2,104 paths / 3,244 schemas. The PR file count
decreased because the offline ingestion fixture now matches dev exactly.

Source hashes, guard outcomes and log hashes are appended to the follow-up JSON.
The previous reviewed head remains in `codex/email-pr3023-qodo-before-dev-refresh-20260929`.
Earlier benchmark receipts retain their original source revisions. All nine
original Qodo threads are resolved; its native reply confirms the FTS finding is
already dismissed. Hosted checks and incremental review must pass on the new
published head before the authorized merge.


## Metadata-only dev refresh (2026-09-29)

Dev advanced from `6110d2ae436c805c3beeda8f84427b4890534ddf` to
`5910412fba589dc0547fac295bb35948496435ce` through one unrelated Backlog task
record. The rebase is conflict-free. Before this evidence update, every repository
file outside Backlog is byte-identical to the reviewed `8956d0b8effd5f9fbefe601beec93a6717efd632`
head, including all source, tests, fixtures, dependency and CI inputs. All 92
certified Python hashes also match the previous validation receipt.

The 374 combined cases and ten guarded native-PST cases were not rerun for this
metadata-only refresh; their source bindings and security certificates remain
unchanged. The comparison receipt is appended to the existing follow-up JSON.
The previous head and tracking are retained in
`codex/email-pr3023-before-dev5910-20260929`. Required hosted checks and incremental
review must pass on the newly published head before the authorized merge.


## VZ tool dev refresh (2026-09-29)

Dev advanced to `60006a2fed2532d900d27accbc2cda87cb08c24b` with an unrelated
macOS VM drill cleanup correction: three Backlog records and four tool/doc/test
files, with no overlap with this PR. The rebase is conflict-free and all seven
incoming paths match dev byte for byte. All other repository inputs match the
reviewed `17b4120933abb9df91f8e8e9a3db78d87b61d660` head before this evidence
update, including all 92 certified Python hashes and the email/backend/frontend,
dependency and CI inputs. Existing email tests and benchmarks were not rerun.

The incoming portable cleanup/orchestration tests pass **97 cases** with zero
failures/errors/skips. All three incoming Python files compile and pass Ruff;
the incoming production script has zero Bandit findings/errors. This uses tiny
temporary fixtures and mocked VM boundaries, not a live VM drill. Source equality
and verification log hashes are appended to the existing follow-up JSON. The
prior reviewed head and tracking remain in
`codex/email-pr3023-before-dev6000-20260929`. Fresh exact-head review and required
hosted checks remain necessary before the authorized merge.


## FastAPI route dev refresh (2026-09-29)

Dev advanced to `607431154cf10129b5d9afa8f9b57d46636466fc` with the
FastAPI 0.141.1 pin and shared served-route inspection. The rebase is conflict-free:
75 of 76 incoming paths match dev exactly. The combined `User_DB_Handling.py`
retains canonical authority and validated organization selection while using dev's
effective route to detect include-time token-scope guards. The existing pagination
diagnostic uses the same shared walker. Independent review found no actionable
regression. The other 91 certified email Python hashes remain unchanged.

With isolated FastAPI **0.141.1**, Pydantic **2.13.5** and Starlette **1.7.0**,
the combined email/auth/route suite passes **746 cases**, and HTTP metrics plus
monitoring scope checks pass **13 cases**, all with zero failures/errors/skips.
All **ten guarded native PST endpoint/metadata cases** pass again with zero
skips/outbound/model attempts. Suites overlap. All **92** PR Python files and
**72** incoming Python files compile (**163** unique paths). Both retained
resolution paths pass Ruff; the incoming 72-path Ruff findings match dev exactly
(104 inherited, zero new). Nine incoming production/script paths have three
Bandit token-type-label findings matching dev, with zero new findings/errors.

Canonical generation and the hosted OpenAPI drift check pass with the unchanged
`58fa00440a66` fingerprint: 2,104 paths / 3,244 schemas. The route-auth ratchet
reports exactly 178 reviewed unauthenticated routes. The shared virtual environment
is unchanged. Source, dependency, native-guard and log hashes are appended to the
existing follow-up JSON. Previous receipts retain their original bindings; no
benchmark or live VM drill was repeated. The previous head and tracking remain in
`codex/email-pr3023-before-dev6074-20260929`. Fresh exact-head Qodo review and all
seven required contexts must pass before the authorized merge.


## Metadata-only dev refresh (2026-09-30)

Dev advanced from `607431154cf10129b5d9afa8f9b57d46636466fc` to
`955b1d9626a055ca44336a00d3d4c144949cb00f` through PR3062, which updates one
unrelated Backlog task record. The rebase is conflict-free. Before this evidence
update, all 22,370 tracked entries outside Backlog match the reviewed
`df1781b27e9c07f1b41df07f42173af68c5f1aa1` head exactly, including every
source, test, fixture, dependency and CI input. All 163 certified Python hashes
match the latest FastAPI validation receipt.

The 746 combined auth/email/route cases, 13 monitoring cases and ten guarded
native-PST cases retain their original source/dependency bindings and were not
rerun. Existing Ruff, Bandit, OpenAPI and route-auth certificates remain unchanged;
this metadata-only refresh introduces no new code/security scope. Benchmarks were
not rerun. The comparison receipt is appended to the existing follow-up JSON.
The reviewed head and tracking remain in
`codex/email-pr3023-before-dev955b-20260930`. Fresh exact-head incremental review
and all seven required hosted contexts must pass before the authorized merge.
