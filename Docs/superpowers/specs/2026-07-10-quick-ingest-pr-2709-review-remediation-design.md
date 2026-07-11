# Quick Ingest PR 2709 Review Remediation Design

## Context

PR #2709 fixes repeat Quick Ingest result classification, restored-job polling,
the reported AntD maximum-update-depth error, and stale YouTube extractor
behavior. Review found several cases where the implementation or regression
coverage did not match the production path. This follow-up addresses those
findings without redesigning the ingestion system.

## Scope

The remediation will:

- preserve existing Webpack watch ignores while appending backend runtime paths;
- make direct-job reattachment tolerate transient status-read failures;
- classify database-level existing-media results as duplicates;
- exercise two consecutive duplicate submissions with the real AntD modal in a
  browser;
- preserve the established global article extraction strategy order while
  honoring the web ingestion APIs' existing analysis declarations when deciding
  whether a request may use LLM extraction;
- surface an actionable warning when the installed yt-dlp is below the supported
  floor;
- redact user-controlled URLs in the touched persistence logging path.

The remediation will not introduce a new job framework, automatically mutate a
user's Python environment, or change the public ingestion API shape beyond the
already-added duplicate counts/status.

## Design

### Webpack Watch Configuration

The Next.js Webpack callback will normalize `watchOptions.ignored` to an array
without converting or dropping valid entries. Empty strings and nullish values
will be removed; strings, regular expressions, and other Webpack-supported
entries will be retained. The following backend runtime patterns will then be
appended as absolute, forward-slash-normalized globs rooted at the repository
workspace so Watchpack can match the absolute paths it observes:

- `<repoWorkspaceRoot>/Databases/**`
- `<repoWorkspaceRoot>/tldw_Server_API/Databases/**`
- `<repoWorkspaceRoot>/tldw_Server_API/Logs/**`
- `<repoWorkspaceRoot>/logs/**`

Tests will exercise representative absolute paths under
`Databases/user_databases/1/Media_DB_v2.db` and
`tldw_Server_API/Logs/server.log`.

### Reattachment Semantics

`reattachQuickIngestSession` will distinguish permanent lookup failures from
transient transport failures. Missing or forbidden jobs remain terminal
interruptions. Network failures, timeouts, rate limits, and server errors will
be retried with a small bounded budget and existing fixed-delay behavior. A
transient failure followed by a valid active or terminal response must preserve
the session.

### Duplicate Persistence

The persistence service will use structured duplicate signals from extraction
and inspect the database repository's returned message in addition to the media
ID. Broad substring matching is not authoritative. Extractor output is a
duplicate only when `is_duplicate is True` or `error_code ==
"duplicate_content"`. Database output is a duplicate only when its result
message matches one of the repository-owned forms:

- `Media '<title>' already exists. Overwrite not enabled.`
- `Media '<title>' already exists (concurrent insert). Overwrite not enabled.`

The canonicalization and handled-overwrite messages are not duplicates. An
existing-media result with overwrite disabled will increment duplicate/skipped
counts and will not increment stored counts. Canonicalization and actual
insert/update results remain successful storage. Regression coverage will
persist the same successful article twice through the real repository contract
rather than injecting `is_duplicate` into extractor output.

### Modal Regression Coverage

The unit test will continue checking stable modal props and will assert that
`styles.body` contains `padding: "0 16px 16px"`, `maxHeight:
"calc(100vh - 180px)"`, and `overflowY: "auto"`; it will not be treated as
proof of the AntD behavior. A browser test using the production AntD component
will submit the same URL twice in one mounted application session, observe a
terminal duplicate/skipped result on the second run, and fail on any `Maximum
update depth exceeded` console or page error. Changed Playwright helpers will use
waiting assertions for controls that appear during transitions; immediate
`isVisible({ timeout })` checks will not be used as waits. Shared option-toggle
helpers will be reused where the same selector and state transition otherwise
remain duplicated across changed tests.

### Extraction Strategy Scope

The global default extraction order will retain `llm` to avoid changing
watchlists and other scraper consumers. No new public request field will be
introduced. The existing API declarations remain authoritative:

- `/api/v1/media/ingest-web-content` uses `perform_analysis`;
- `/api/v1/media/process-web-scraping` uses `summarize_checkbox`;
- `api_name` selects the requested analysis provider when analysis is enabled.

Those declarations already flow into the scraping service as the `summarize`
intent, but the extraction router currently ignores that intent and selects its
strategy order independently. The service/core boundary will carry that existing
boolean intent into extraction. When analysis is false, `llm` will be removed
from the effective extraction order for that request. When analysis is true, the
configured or default order will remain intact, including `llm`. Explicit
per-domain custom scraper ordering remains ordered as configured except for this
request-level LLM permission gate. `auto_chunking_use_llm` remains independent
and controls only chunk planning.

Quick Ingest already sends `summarize_checkbox` from its `perform_analysis`
option, so it needs no new request property or user-facing control.

### yt-dlp Upgrade Diagnostics

The dependency floor remains `yt-dlp>=2026.7.4`. Existing environments will not
be upgraded automatically. The media ingestion startup or request boundary will
compare the installed version with the supported floor and emit one redacted,
actionable warning that includes the supported update command. The check must
not block non-YouTube ingestion.

### Logging Safety

User-controlled article URLs logged in the touched persistence path will use the
existing `redact_url_for_log` helper. Query values and credentials must not be
written to logs; enough origin/path context may remain for diagnosis.

### Result Status Consistency

The repeated conference-item terminal status expression will be represented by
one small local helper returning `skipped_existing`, `failed`, or `completed`.
This is the only production refactor included beyond the behavior fixes.

## Testing

Each behavior change will follow red-green verification outside the sandbox:

1. Webpack config test preserving an existing regular-expression ignore and
   matching representative absolute backend runtime paths.
2. Reattach tests for transient failure then success and permanent 404 failure.
3. Backend test that stores the same URL twice and verifies duplicate counts,
   plus negative and mixed-result cases proving unrelated errors containing the
   word `duplicate` remain failures.
4. Extraction tests proving the global default still includes `llm`, requests
   with `perform_analysis`/`summarize_checkbox` false suppress it, requests with
   analysis enabled retain it, and Quick Ingest continues sending its existing
   `summarize_checkbox` declaration without a new strategy property.
5. Version-warning unit test for stale and current yt-dlp versions.
6. Logging test proving secrets in URL query values are absent.
7. Real-browser repeated duplicate Quick Ingest flow with console/page-error
   capture.

After focused tests, run the affected frontend/backend suites, Bandit on touched
Python files, and the full PDF, local-link, and YouTube Shorts UAT walkthrough.

## Merge Conditions

The branch must be rebased on current `dev`, all actionable PR review threads
must be resolved, required checks must pass, and the human requester must provide
the repository-required human-written change summary before merge. TASK-12946
must contain concrete acceptance criteria, current verification evidence, and
accurate Definition of Done/status fields before closeout.
