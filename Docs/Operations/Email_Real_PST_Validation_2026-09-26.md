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


## Frontend chat dev refresh (2026-09-30)

Dev advanced from `955b1d9626a055ca44336a00d3d4c144949cb00f` to
`c867287210d4e85314b00ff22e7d30d0474030a0` through PR3064, adding saved-chat
read-only selection and disabling passive feedback writes in temporary mode.
The rebase is conflict-free. All ten incoming frontend source/test paths and two
Backlog records match dev exactly. Before this evidence update, the other 22,360
non-Backlog tracked entries match reviewed `d4ed59d64e59dd9df8b4853079e7570d836dd9a6`
exactly; all 163 certified Python hashes, backend/schema/dependency/CI inputs and
the email patch remain unchanged. Independent review found no integration blocker.

The five affected frontend suites pass **50 tests**, with zero failures, errors
or skips, using Bun 1.3.2, Node 26.0.0 and the existing Vitest 4.0.18 installation.
Source and log/XML hashes are retained in the existing follow-up JSON. Temporary
dependency symlinks were removed after the tests; shared installations are unchanged.
The 746 combined auth/email/route, 13 monitoring and ten guarded native-PST cases
retain their original FastAPI 0.141.1 source/dependency bindings and were not rerun.
Existing Ruff, Bandit, OpenAPI and route-auth certificates remain unchanged; no
Python code changed and there is no new Bandit scope. Benchmarks were not rerun.
The prior reviewed head and tracking remain in
`codex/email-pr3023-before-devc867-20260930`. Fresh exact-head incremental Qodo
review and all seven required hosted contexts must pass before the authorized merge.


## Buddy workspace dev refresh (2026-09-30)

Dev advanced from `c867287210d4e85314b00ff22e7d30d0474030a0` to
`03043d1c10cbdbfa53945c0641d90e4e37836754` through PR3056. The incoming shared workspace
collection URL uses `/api/v1/workspaces/`, matching the server route and avoiding
a redirect rejected by the unchanged request transport. The rebase is conflict-free.
All 18 incoming paths (three frontend source/test paths, twelve qualification
documents/artifacts and three Backlog paths) match dev exactly. Before this evidence
update, the other 22,367 non-Backlog tracked entries match reviewed
`e692fa0d0af35f3d90817d4ad819b65ee5f41f72` exactly; all 163 certified Python hashes and
email/backend/schema/dependency/CI inputs remain unchanged. Independent review is clear.

Fresh workspace contracts (**37 tests**) and redirect security checks (**13 tests**)
pass across three files, with zero failures, errors or skips, using Bun 1.3.2,
Node 26.0.0 and existing Vitest 4.0.18. Source and log/XML hashes
are appended to the existing follow-up JSON. Exact temporary dependency symlinks
were removed; shared installations are unchanged. The previous FastAPI 0.141.1
email/auth/native-PST/schema/security validation retains its source/dependency
binding and was not rerun. No Python code changed, no new Bandit scope is introduced,
and no benchmark was rerun. The prior reviewed head and tracking remain in
`codex/email-pr3023-before-dev0304-20260930`. Fresh exact-head incremental Qodo
review and all seven required hosted contexts must pass before the authorized merge.


## Broad CI repair on September 30 — TASK-13377.2/.3/.4

Broad CI 36723727218 exposed a shared collection selector KeyError, a real
PostgreSQL bootstrap lock inversion, stale fixtures, and two email unit modules
missing from five full-suite matrices. The selector now interpolates its existing
column whitelist with bound values. The v61 verifier reads stored pg_policy trees
without deparsing them; exact tenant-policy names, schema, forced RLS and both
policy expressions remain mandatory. No retry, authentication or CI gate was weakened.

Final verification: 253 contract cases, 254 Collections cases, 110 affected official
PostgreSQL cases, 30 ChaCha migration/concurrency/shared-chat cases, and 12 final
catalog/privacy cases passed. Watchlists passed 708 cases with one existing
live-ingestion skip, four deselections and one existing xpass. Suites overlap.
Both production files are Ruff-clean and Bandit reports no findings or errors.
The 26 remaining test Ruff findings match the published baseline (27 before);
test-only Bandit comparison excludes expected B101 assertions and retains one
inherited B105 warning for the --cov option label, with no new finding.
Independent final patch review is clear. Source, XML/log and comparison hashes
are appended under full_suite_ci_repair_20260930 in the existing follow-up JSON.
Historical values and benchmark receipts remain unchanged. Changed historical
Python certificate paths are listed explicitly; older validation is not recertified
for modified paths. No full repository suite, native-PST, throughput benchmark,
million-message benchmark or live VM drill rerun is claimed. Publication, latest
dev integration and fresh exact-head review/CI remain pending at this receipt.


## Playground dev refresh on September 30 — TASK-13377

After the reviewed CI repair, latest dev 2256bc82afa154891c635df3ef955ed7a6bc61b3
was integrated without conflicts. All seven incoming Playground test/locale and
Backlog paths match dev; the other 22,377 non-Backlog tracked entries and all
16 repaired Python hashes are unchanged before this evidence update. All 27
affected Playground tests passed across four files with zero failures/errors/skips
using the existing Bun 1.3.2, Node 26.0.0 and Vitest 4.0.18 installations. Exact
temporary dependency links were removed; shared installations were unchanged.
Independent integration review is clear. Equality, source, command and log/XML
receipts are appended under playground_dev_refresh_20260930. Earlier JSON values
and original benchmark/native-PST/schema bindings are preserved without reruns.
Fresh exact-head review and all seven required contexts still gate the merge.


## Resource Governor dev refresh and verified follow-up — 2026-09-30

PR3023 was rebased from dev2256 to `f3f1b4fdbe3fe461b371ece30887c5fff8476d9d` (PR3066). All21 incoming paths exactly matched dev and the other22,371 non-Backlog tracked entries remained identical before repair/evidence edits. The only conflict was two equivalent nested FastAPI capabilities-route test corrections; the entire dev file was retained. Fifteen of sixteen prior repair Python hashes remain unchanged; the capabilities test changed to exact dev.

Validation exposed an inherited import-order failure: importing the governor endpoint first imported `main` before its router existed, leaving eight admin/diagnostic tests at404. The same seed4293640905 and complete immutable dev reproduce the same eight failures. TASK13377.5 removes the eager app import and cached fallback at the existing lazy `_get_app` boundary. Its fresh-process regression failed before the fix and passes after; all admin dependencies remain unchanged. An initial incomplete baseline archive's missing MCP package error is retained as a setup diagnostic and superseded by the complete immutable dev comparison.

Independent review also proved minute token clamping undercounted durable daily quotas in both backends, even after `commit(actuals)`. TASK13377.6 preserves original units for daily checks/consumption and clamps only minute-window reservations; Redis fallback also receives the original request. Four actual SQLite daily-ledger regressions failed before repair and pass after, covering daily denial, exact500-token accounting, reserve idempotency, commit and subsequent exhaustion. Nineteen incoming paths remain exactly dev; the two governor files carry these reviewed minimal repairs.

The final65-file affected suite passes372 cases with two existing expected failures and zero failures/errors/ordinary skips. Focused import/daily/safety-net validation passes42 cases; retained email/auth/route validation passes343 cases with zero failures/errors/skips. Suites overlap and do not certify a full repository suite. All20 incoming/repaired Python paths compile. Ruff7 findings match the correctly configured immutable baseline7; production Bandit11 inherited auth token-type labels match baseline11, with no exclusions or new findings/errors. Twelve test files have zero findings after the B101 assertion exclusion and specifically justified B404/B603 comments on the import-order test's fixed interpreter/literal child code. The comments preserve its AST; the original test source and reproduced scanner warnings are retained.

Final immutable repair patch SHA256 `c6f1d4c03e658d791d3aa50dd9550f0e57d42ca4a9c43ccec9cd341ebb04f292` is independently clear. Fresh supported-overlay canonical generation and drift check pass with fingerprint `58fa00440a664a47cb9ad6e72350b2515a7e6ac061e81a13aca032fe803a3b47`,2104paths/3244schemas; route-auth baseline remains exactly178 unauthenticated routes. Shared installations were unchanged. Full source/dependency/log/XML/quality/review/equality receipts are appended under `governor_dev_refresh_20260930` in the existing follow-up JSON; every prior JSON value is preserved.

Historical163-file certificate changes introduced by this refresh are explicitly listed in that receipt, along with changes already introduced by the earlier repair. The original746-case auth/email/routes,13monitoring and10guarded native-PST validations retain their original source/dependency bindings and were not rerun in full. Native-PST, throughput/million-message benchmarks and live VM drill were not rerun. Publication, exact-head Qodo and all seven fresh required contexts remain pending; old47c3 checks cannot certify this source.


## Strict Workspace startup dev refresh and fixture repair — 2026-10-01

PR3023 was rebased onto dev `85ede1f1df10c03505c603e4183920edcb7cbfef` (PR3041). The only rebase conflict was the generated OpenAPI fingerprint. Of 83 incoming paths, 78 now match dev exactly. Five reviewed exceptions preserve the two email unit modules in all five full-suite matrices and their CI contract, the raw `pg_policy` v61 verifier, the combined canonical fingerprint, and the scoped Chat_NEW fixture repair. All other 22,344 non-Backlog tracked entries were identical before these evidence edits.

The original same-seed affected run passed 1,966 cases, failed two Buddy handoff cases and skipped 26 existing cases. Buddy alone passed all seven; Chat_NEW moderation plus Buddy reproduced the same two failures on both the PR and complete immutable dev. The third completion hit HTTP429 because Chat_NEW set low `TEST_CHAT_*` limits during collection. TASK13377.7 removes only those four redundant assignments. Its existing scoped monkeypatch fixture still sets and reinitializes the limits, and the explicit 429 enforcement test remains intact. The paired tests plus that enforcement test pass all 24 cases after repair. No production quota or authentication behavior was changed.

The final 40-module incoming run passes 1968 cases with 26 existing ordinary skips, zero failures/errors and seed4293640905. The exact skip identities/reasons match the original run: backend-specific transaction/concurrency/privacy behavior, missing live provider keys, TestClient streaming limitations and single-user auth behavior. Retained email/auth/route/media/Collections/governor/v61 regressions pass 638 cases with zero failures/errors/skips. Suites overlap and do not certify a full repository suite. Final source hashes were checked after the incoming run.

All 58 incoming Python paths compile. The 17 production files have zero Bandit findings/errors without exclusions. Ruff70 findings match immutable dev70 after mapping exact unchanged fixture lines through the four-assignment deletion. Test Bandit10 inherited B105 findings match baseline10 after excluding B101 assertions; this test scan is not zero findings. No new findings/errors or warning exclusions were introduced. No tests or CI gates were disabled. Independent final source/artifact patch `849e6bcee16f441852fc674eaf89c86af501bbc2aca3f4d76050ebf289ad587a` is clear, including the applied fixture and focused XML results.

Fresh supported-overlay canonical generation/drift passes fingerprint `bb17b220db46ddb10fd4dc22a4e87d3c63f8a4fd5e38287386c2cf8ca4426cd3`, 2,105 paths/3,245 schemas. Standard `bun run generate:api-types` with openapi-typescript7.13.0 succeeds; its exact temporary frontend dependency link was removed and shared installations were unchanged. The route-auth baseline remains exactly178 unauthenticated routes. Restricted exporter omissions, Ruff cache denial and path/line comparison diagnostics remain retained; corrected complete/no-cache scans are credited.

Full source/equality/dependency/command/log/XML/quality/review receipts are appended under `workspace_startup_dev_refresh_20261001`; all 34 historical JSON values and the report's existing text are preserved. The original163-file Python certificate has seven changed paths, one additional upstream PostgreSQL HTTP-lifecycle test this refresh; the exact list and prior repair hash changes are explicit in the receipt. Original746 auth/email/routes,13monitoring and10guarded native-PST results retain their original source/dependency bindings and were not rerun in full. Native-PST, throughput/million-message benchmarks and live VM drills were not rerun. Publication, fresh exact-head Qodo and all seven required hosted contexts still gate the merge; earlier831dc checks are historical.


## Resource Governor policy-resolver dev refresh — 2026-10-01

The verified Workspace checkpoint was rebased conflict-free onto dev `b365af1827b607fc221f9bf31ca76dde881edb4f` (PR3068). Of 46 incoming paths, 42 match dev exactly. Four reviewed intersections preserve the email persistence/worker/scope changes alongside the upstream server-generated upload ledger operation ID, original daily token units in both governor backends, and the four retained daily-quota regressions. All other 22,389 non-Backlog tracked entries were identical before the subsequent test-helper repair and evidence append. All 58 Python files from the preceding Workspace verification remain byte-identical. Its 1,968-pass/26-skip result remains bound to dev85ed; current transitive governor/auth/persistence inputs changed and are verified separately here.

Before the subsequent TASK13377.8 shared test-helper repair, supported-overlay tests passed 596 affected governor/auth/persistence/fixture cases with two existing Redis xfails, 650 retained email/auth/Collections/ChaCha cases and 47 Billing/authenticated-email cases, with zero failures/errors/ordinary skips. The exact xfail names and reasons remain in the receipt. These suites overlap and do not certify a full repository suite. The authenticated email probe initializes production Resource Governor ingress, uses real AuthNZ and SQLite email fixtures without auth/billing/storage/parser overrides, rejects a read-only key, and reuses the existing selected-organization quota and two-organization upload/search/detail isolation scenario. Rate-limit response headers and the authenticated user's bucket prove ingress executed; outbound and model calls remain guarded.

All 34 incoming Python paths compile. Ruff 45 current findings are mapped to the configured immutable dev 70 baseline with zero new findings. The 16 production/helper paths have 12 Bandit findings: 11 inherited B106 token-type labels and one inherited B113; all match dev exactly, with zero new findings/errors and no production exclusions. The 18 test paths have four exact inherited B105/B404/B603 findings after excluding B101 assertions, with zero new findings/errors. These scans are not zero-findings scans. Initial baseline package-marker and canonical-path comparison diagnostics are retained and corrected, not counted as source improvements. Independent immutable integration review is clear for patch `34cbbecf0113ca7024c9fcbc69eb41b27b0da7c39a127c5b7ee7c3fcf5453585`. No tests or CI gates were disabled.

Canonical OpenAPI generation/drift on this b365 integration passes `bb17b220db46ddb10fd4dc22a4e87d3c63f8a4fd5e38287386c2cf8ca4426cd3`, 2,105 paths / 3,245 schemas. The generated schema is byte-identical to the one used for the prior successful standard type generation; no additional frontend/type-generation run was needed in this refresh. The current official route-auth ratchet remains exactly 178 unauthenticated routes and the official governor route-map lint passes.

Receipts are appended under `policy_resolver_dev_refresh_20261001`, preserving all 35 earlier JSON values and the report's prior text. The original 163-file certificate now has 12 changed paths, explicitly listed with current hashes and affected-suite bindings. Eight of 20 prior governor-certificate paths, 13 of 16 prior CI-repair paths and all 58 Workspace paths remain byte-identical to their respective certificates; the rest are listed explicitly. Original 746 auth/email/routes, 13 monitoring and 10 guarded native-PST results retain their original source/dependency bindings and were not rerun in full. Native-PST, throughput/million-message benchmarks and deployed VM drills were not rerun. Shared installations remain unchanged. The subsequent TASK13377.8 test-helper repair and its final checks are recorded separately below. Publication, fresh exact-head Qodo/all review comments and all seven current-head required hosted contexts still gate merge; old 831dc successes cannot certify the new head.


## Shared test-isolation CI repair — 2026-10-01

TASK13377.8 repairs the shared app-main test helper after broad CI36790042527 failed one platform infrastructure/metrics shard (one failed,478 passed) and its aggregate. All219 jobs across three API pages were inspected. Native tracing on the current source proved that a collected reload module left its numeric ID behind and a distinct plain import naturally reused it; the helper then incorrectly unloaded the first import. A native `weakref.WeakSet` now tracks live module identities, preserving reload restoration and allowing retired modules to be collected. No production behavior, tests or CI gates were weakened.

The deterministic collected-reload identity regression failed at the expected first-import assertion on the original helper and passes after repair. All six singleton-isolation tests pass, including a real nested pytest session. The full failed Infrastructure/Metrics/Monitoring scope passes480 tests with zero failures/errors/skips. Fresh checks on the final helper source pass596 affected governor/auth/persistence cases with the same two existing Redis xfails,650 retained email/auth/Collections/ChaCha cases and47 Billing/authenticated-email cases, with zero failures/errors/ordinary skips. Suites overlap and do not certify a full repository suite. Local Python3.11.13/pytest8.4.1/randomly4.1.0 differs from hosted Python3.12.14/pytest9.1.1/randomly5.0.0; hosted checks must certify the published head.

Both touched Python files compile. Ruff retains two inherited B010 findings and introduces none; Bandit reports zero findings/errors in the shared test helper without exclusions. The regression file retains two inherited B404/B603 findings after B101 assertion exclusion, with zero new findings/errors. Exact original files, RED/GREEN logs/XML, native ID-reuse trace, scanner reports and immutable independent review for patch `0fbb373d3f0bd04e351adf4655d4264f3b276cf7c4d014775456535abd776c50` are retained in `test_reload_identity_ci_repair_20261001`. The preceding b365 integration receipts explicitly remain bound to their pre-helper source; these fresh runs bind the final helper source. All35 historical JSON values and the prior report text are preserved, alongside the separate b365 integration receipt. Publication, fresh exact-head review and all seven required contexts remain pending; old831dc successes cannot certify this repair.


## UX smoke allowlist CI repair — 2026-10-01

TASK13377.9 addresses Frontend UX Gates36815061080/job110221608849: all31 exceptions expired at the September30 UTC boundary. The existing metadata check failed once and prevented100 remaining test cases from running:99 page cases and the second metadata case. Four relevant source/workflow/policy files exactly match devb365. Fresh route evidence supports only four exception classes. The repair removes27 stale or unobserved entries, narrows optional404 matching to the exact moderation-list resource URL on `/moderation`, limits Drawer deprecation to `/kanban`, and retains the two deliberate route-boundary fixture signatures. Stable IDs and existing WebUI owners remain; the four exceptions expire2026-10-08. TASK13406 tracks their source/fixture remediation before that date. The UTC validator, page inventory, assertions, CI workflow and policy are unchanged.

The existing metadata test failed before repair. Four actual classification scenarios produced two failures/two passes on the original broad404 rule; all four pass after preserving the already captured console location through existing diagnostic types. Final focused checks pass6 cases. The strict production all-pages selection passes101 cases, all16 deliberate recovery fixtures pass in supported development mode, and the existing media-render-loop case passes after removing its obsolete exception. Final reports contain zero skipped, unexpected or flaky results. These scopes overlap and are not a full repository suite. Both TypeScript files compile; scoped ESLint and its correctly configured unchanged baseline each have zero errors/warnings. The new test passes formatting. Bandit is inapplicable to this TypeScript-only source change; no unsupported scanner or all-scope zero-findings claim is made. Prior Python source is unchanged from published0f24 and retains its historical evidence bindings.

The frozen Bun1.3.2/Next16.1.4/Playwright1.58.0 task-owned workspace used an advanced standalone UI built with localNode26.0.0 and final test runnersNode20.19.5. The real minimal backend uses the supported Python3.11.13/FastAPI0.141.1 wrapper, task-owned SQLite and a local mock model server. Two provisional development runs encountered Next's verified4GB memory-threshold restarts; final recovery proof uses a task-owned8GB development heap. No shared installation or hosted workflow limit was changed. Temporary31-date and teardown-instrumented diagnostics, the initial scope/cwd/format setup errors, CLI/MCP failures and actual RED results are retained; they are not credited as final repair. Local evidence does not certify hostedNode20/Python3.12 checks.

Raw teardown evidence also captures a late documentation-manifest HTTP500 after smoke assertions. It remains unexpected and unallowlisted. Independent review supports the inherited standalone documentation-source lookup diagnosis and accepts this separate test-maintenance repair; TASK13407 tracks the product repair and deterministic response/content smoke check. Passing metadata/page selections do not establish zero raw browser errors or functional documentation acceptance. Immutable source patch `d001ed8bde3cc2aff39c1a4abe4faa59f68961c282241cae14f19f998740966f` is independently clear. Source, environment, raw-phase, RED/GREEN, quality, policy/follow-up and review receipts are appended under `ux_smoke_allowlist_ci_repair_20261001`, preserving all37 historical JSON values and all prior report text. Publication, fresh exact-head Qodo/all review comments and all seven required hosted contexts remain pending at this receipt.

Final post-append documentation contracts pass212 cases with zero failures/errors/skips on the supported wrapper and seed4293640905. Their log/XML/source binding is retained with the UX repair receipt.


## Sandbox WebSocket data assertions — 2026-10-01

TASK13377.10 addresses broad CI36836299812/job110297946051: the live multi-subscriber test compared raw next-four sequence lists `[1,2,3,4]` and `[2,3,4,5]`, with1 failed/91 passed and34504 warnings. Hosted frame types were not captured, so the original hosted heartbeat cause remains unproven. The relevant test, fixture, stream hub and endpoint were exactly devb365. Production gives heartbeats global sequence values but deliberately does not replay them to later subscribers; the existing sandbox fixture accelerates them to10ms. A native early-heartbeat probe reproduces the original order assertion on unchanged source, proving the test counted transport frames as data. An initial probe affected by frozen endpoint-synthetic settings is retained separately as a rejected setup diagnostic.

The test-only repair reuses one reader across all fixed-count data assertions in this module, ignoring only `type=heartbeat`. It preserves every other frame and sequence. A deterministic leading/interleaved heartbeat regression fails before the reader exists, then passes. The live test now additionally requires complete start/stdout/stdout/end delivery, full frame equality, and distinct increasing data sequences for both subscribers. The same native early-heartbeat probe passes with retained data sequences `[6,9,12,15]` for both subscribers. Heartbeat emission, replay behavior, fixtures, production source, workflow and dependencies are unchanged.

The complete module passes4 cases. The entire failed shard, including dedicated heartbeat, stress and resume coverage, passes93 cases on the supported Python3.11.13 wrapper and93 on an existing read-only Python3.12.11 environment. Final XML contains zero failures/errors/skips. Native warnings remain6 and29757 respectively; scopes overlap and are not a full repository suite. Both use FastAPI0.141.1, but local macOS/Python3.12.11/cryptography50.0.1 does not certify hosted Linux/Python3.12.14/cryptography50.0.2. Synthetic=false is established before native imports to avoid the unrelated cached synthetic setup, matching this test's intent. No shared installation or unowned environment was altered.

The touched Python file compiles. Scoped Ruff has4 inherited findings versus10 in the correctly configured original baseline, with zero new findings; remaining diagnostics are retained. Scoped Bandit has zero findings/errors versus the same baseline after test-onlyB101 assertion exclusion. New helper/regression formatting and diff check pass; the initial cache-write and extracted-format setup failures are retained. Immutable source patch `bb6be00508a0440a47e02d937aa01ebdc68fc09a358068b5b626f43528c47df3` receives independent review with no actionable findings. No production architecture change or new ADR is needed.

Evidence is appended under `sandbox_ws_heartbeat_ci_repair_20261001`, preserving all38 historical JSON values and all prior report text. Earlier certificates and results keep their original source/transitive/dependency bindings, including the12 previously listed changed paths in the original163-file certificate; no new full certificate is claimed. No native-PST, throughput, million-message or live VM drill is repeated. The separate actual Sync/Personal Context restart-recovery failure remains unconfirmed and blocks merge. Normal publication, fresh exact-head review and hosted checks remain pending at this receipt; old6b7 successes cannot certify a later repair commit.
