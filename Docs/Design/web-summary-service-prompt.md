# Web article summarization Service Prompt

Approved scope: synchronous `/api/v1/media/process-web-scraping`; TASK-13195.

Add `media.web.summarization` with literal `system` and `user` parts to the
existing registry and shared Settings editor. Resolve saved parts once for the
authenticated request owner before scraping. Explicit request parts, including
empty strings, win independently. Do not read prompt storage when summarization
is disabled or both parts are explicit. Close lookup connections on their worker.

Keep the enhanced individual-URL, enhanced crawl, and legacy fallback defaults
distinct when no saved override exists. Settings presents the deployed web-article
defaults and explains that reset restores each engine's existing defaults.
Carry the request's immutable override mapping through the existing call chain;
do not persist that transient mapping into scraping-job metadata. Fix the existing
individual-URL `system_prompt`/`system_message` forwarding mismatch.
Use the same authenticated owner for media persistence instead of deriving an
owner from the media database wrapper.

Preserve scraping, extraction, provider settings, output shape, persistence,
chunking, and rate limits. Direct callers, scheduled workflows and
`/ingest-web-content` do not resolve Service Prompts in this slice.

Verify model-facing messages through the public adapter; owner isolation,
independent overrides, defaults/reset, multi-page snapshots, disabled-analysis
bypass, corruption/connection cleanup, engine/fallback paths and atomic Settings
editing. Run focused regressions, Bandit and OpenAPI checks before review.
