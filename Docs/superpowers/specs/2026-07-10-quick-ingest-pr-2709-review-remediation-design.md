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
  opting Quick Ingest's default scrape request out of LLM extraction explicitly;
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
entries will be retained. Backend runtime patterns will then be appended.

### Reattachment Semantics

`reattachQuickIngestSession` will distinguish permanent lookup failures from
transient transport failures. Missing or forbidden jobs remain terminal
interruptions. Network failures, timeouts, rate limits, and server errors will
be retried with a small bounded budget and existing fixed-delay behavior. A
transient failure followed by a valid active or terminal response must preserve
the session.

### Duplicate Persistence

The persistence service will inspect the database repository's returned message
in addition to the media ID. An existing-media message with overwrite disabled
will increment duplicate/skipped counts and will not increment stored counts.
Canonicalization and actual insert/update results remain successful storage.
Regression coverage will persist the same successful article twice through the
real repository contract rather than injecting `is_duplicate` into extractor
output.

### Modal Regression Coverage

The unit test will continue checking stable modal props, but it will not be
treated as proof of the AntD behavior. A browser test using the production AntD
component will submit the same URL twice in one mounted application session,
observe a terminal duplicate/skipped result on the second run, and fail on any
`Maximum update depth exceeded` console or page error.

### Extraction Strategy Scope

The global default extraction order will retain `llm` to avoid changing
watchlists and other scraper consumers. Quick Ingest's web-scraping request path
will explicitly select the non-LLM strategy order unless the user explicitly
requests LLM-backed extraction. This keeps the incident fix at the client/service
boundary that owns Quick Ingest defaults.

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

## Testing

Each behavior change will follow red-green verification outside the sandbox:

1. Webpack config test preserving an existing regular-expression ignore.
2. Reattach tests for transient failure then success and permanent 404 failure.
3. Backend test that stores the same URL twice and verifies duplicate counts.
4. Extraction tests proving the global default still includes `llm` and Quick
   Ingest sends the explicit non-LLM strategy order.
5. Version-warning unit test for stale and current yt-dlp versions.
6. Logging test proving secrets in URL query values are absent.
7. Real-browser repeated duplicate Quick Ingest flow with console/page-error
   capture.

After focused tests, run the affected frontend/backend suites, Bandit on touched
Python files, and the full PDF, local-link, and YouTube Shorts UAT walkthrough.

## Merge Conditions

The branch must be rebased on current `dev`, all actionable PR review threads
must be resolved, required checks must pass, and the human requester must provide
the repository-required human-written change summary before merge.
