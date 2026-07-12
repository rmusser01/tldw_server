# Metadata-Only Web Ingestion Guard Design

**Backlog task:** TASK-12111

## Problem

The web extraction pipeline currently treats JSON-LD as successful when either body content or a structured-data summary is present. A page with a `description` but no `articleBody` therefore stops the extraction pipeline before a full-text strategy runs. Persistent ingestion then wraps the empty body in a metadata envelope and stores a metadata-only media item.

## Design

### Extraction contract

JSON-LD extraction is successful for ingestion only when `content` contains non-whitespace text. Structured `description`, `abstract`, or `summary` values remain available as the result summary, but do not by themselves satisfy the body-extraction contract.

The extraction pipeline remembers a non-empty JSON-LD summary when JSON-LD lacks body content. When a later strategy obtains the page body, finalization adds that remembered summary only if the later result does not already provide a non-whitespace summary. This keeps higher-quality downstream summaries authoritative while retaining useful structured metadata.

Legacy multi-URL processing must not replace a preserved structured summary with `None` when optional LLM summarization is disabled. When LLM summarization is enabled and succeeds, its generated summary remains authoritative.

### Persistence guard

Both persistent web-ingestion implementations validate the extracted body immediately before metadata formatting, chunking, and database writes. Valid body content must be a string containing non-whitespace text. If the string contains a recognized `[METADATA]...[/METADATA]` envelope, validation examines the text after that envelope rather than accepting the envelope itself as body content.

- Enhanced persistence skips an article whose body is missing or whitespace-only, appends a URL-scoped error to the existing batch `errors` list, and continues with other articles.
- Legacy compatibility persistence applies the same rule and returns its existing batch response with a new `errors` field when applicable.

The guard is defense in depth. Extraction should normally supply a body, but persistence must not create a record that contains only the generated metadata envelope.

### Scope

Modify only the shared article extraction module, the two existing persistence paths, and focused tests. Do not add a Wikipedia-specific scraper rule, reorder the global strategy list, introduce configuration, or refactor unrelated web-scraping code.

## Data flow

1. Parse JSON-LD metadata.
2. Save any structured summary.
3. If JSON-LD has a non-empty body, return it normally.
4. Otherwise continue through configured extraction strategies.
5. Finalize the first later body result and attach the saved summary if needed.
6. Before persistence, reject a missing, non-string, whitespace-only, or metadata-envelope-only body.
7. Persist valid content and report skipped URLs without aborting the batch.

## Error handling

Empty-body articles are treated as extraction/persistence failures, not server errors. Batch processing continues. Error messages identify the affected URL consistently with existing extraction errors and do not include page content or exception details.

For response compatibility, both persistence paths retain the existing `persist-ok` batch status. Callers distinguish partial or complete skips through `media_ids`/`stored_articles` and the `errors` list.

## Testing

Use TDD with focused regressions:

- Description-only JSON-LD must not short-circuit the pipeline.
- A later extractor must supply the body while the JSON-LD summary is retained.
- An existing later summary must not be overwritten.
- Legacy processing with optional summarization disabled must not erase a preserved structured summary.
- Enhanced persistence must skip whitespace-only content, report the URL, and persist valid siblings.
- Legacy persistence must do the same.
- Both persistence paths must reject a recognized metadata envelope with no body text.
- Existing JSON-LD body extraction and persistence tests must remain green.

Run focused pytest suites, formatting/lint checks available for touched files, and Bandit over the touched Python scope.
