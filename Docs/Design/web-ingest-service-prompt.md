# Web-content ingestion Service Prompt

Approved scope: TASK-13197, following the merged web-summary setting in PR #2907.

Reuse `media.web.summarization` for `/api/v1/media/ingest-web-content`. Capture
one immutable prompt configuration from the authenticated owner's storage before
scraping; share it across individual URLs, sitemap, URL-level and recursive
scraping. Explicit system/user parts win independently, including empty strings.
Disabled analysis and fully explicit requests bypass saved-prompt lookup.

Keep engine-local defaults, provider configuration, extraction, chunking,
timestamps, cookies and output fields unchanged. Direct service callers and
scheduled workflows do not acquire user prompt configuration. Reuse the current
resolver with explicit input parameters rather than coupling it to either HTTP
request schema. Update the existing Settings description; add no setting ID.

For crawl modes, retrieve articles from the existing ephemeral result envelope
before mapping summaries to ingestion analysis fields. Retain support for inline
article/result lists used by compatibility callers. No new storage mechanism.

Tests exercise HTTP, real owner prompt databases, real orchestration and ephemeral
storage, and model-facing messages; substitute only external extraction/model
operations. Cover all modes, engine fallback, independent overrides, owner edits
mid-request, disabled analysis, reset/defaults, and existing ingestion controls.
