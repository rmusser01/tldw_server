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
The PR targets `dev` and remains draft until the requester supplies the required
human-written `Change summary` under the repository's merge policy. Creating the
PR is not the separate owner release approval.
