# Presentations / Slides Module Design

## Summary
- Implement a per-user Slides database and API surface for CRUD, generation, export, and version history.
- Reveal.js ZIP is the primary export format; Markdown (Marp), JSON, and PDF are supported.
- Generation uses existing LLM adapters and per-user data sources (chat, notes, RAG, media).

## Storage
- Per-user database: `Databases/user_databases/{user_id}/Slides.db`.
- Tables: `presentations`, `presentations_fts`, `presentations_versions`, `sync_log`.
- `slides` stores JSON array of slide objects; `slides_text` is app-maintained for FTS.
- `presentations_versions` stores immutable snapshots keyed by presentation_id + version.
- `presentations.template_id` stores the optional template used at create/generate time.

## API
- CRUD routes under `/api/v1/slides/presentations`.
- Generation routes under `/api/v1/slides/generate`.
- Export route under `/api/v1/slides/presentations/{id}/export`.
- Reorder route under `/api/v1/slides/presentations/{id}/reorder`.
- Version routes under `/api/v1/slides/presentations/{id}/versions` (list/get/restore).
- Template routes under `/api/v1/slides/templates` (list/get).
- ETag/If-Match required for mutating operations. Structured routes retain weak tags; dedicated standalone source operations use strong tags.

## Export
- Reveal.js assets are bundled locally and copied into a ZIP along with `index.html`.
- Assets path resolution:
  - Env override: `SLIDES_REVEALJS_ASSETS_DIR` (absolute or repo-relative).
  - Default: `tldw_Server_API/app/core/Slides/revealjs`.
- Bundled assets are a lightweight Reveal.js-compatible set; point `SLIDES_REVEALJS_ASSETS_DIR` to an official Reveal.js dist for full features.
- If assets are missing, export returns a clear error (`slides_assets_missing`).
- Markdown export uses Marp-compatible syntax with a Reveal-to-Marp theme mapping or a stored `marp_theme` override.

## Structured Rendering And Sanitization
- Markdown is converted to HTML via `markdown` and sanitized with `bleach`.
- Allowed tags/attrs are limited to a safe allowlist; raw HTML is stripped.
- Speaker notes are escaped as plain text.
- Custom CSS is sanitized using `bleach` CSS sanitizer and rejects `@import` and `url()`.

## Generation
- Sources: prompt, chat conversation, RAG results, media transcript, notes.
- Source limits:
  - Enforce `max_source_tokens` or `max_source_chars`.
  - If chunking is disabled and limits are exceeded, return 413.
  - If chunking is enabled, split by token estimate and optionally summarize chunks.
- LLM prompt requests JSON output matching the Slide schema.

## Testing
- Unit tests for Slides DB CRUD, FTS, export renderers, sanitization, and generation parsing.
- Integration tests for CRUD, export headers/body, and generation with mocked LLM.

## Standalone HTML Content Kind

`standalone_html` is a second durable presentation kind alongside `structured_slides`. It stores one complete validated UTF-8 HTML document, matching SHA-256/byte/slide metadata, immutable generation provenance, and `slides: []`. Content kind cannot change in V1.

The document is executable, untrusted, opaque text. It is not passed through the structured Markdown/Bleach rendering path and is never described as sanitized or safe. Backend validation enforces a bounded storage contract before generation commit, save, restore, and saved export. It does not execute the script and cannot make arbitrary JavaScript trustworthy.

### No-Execution Boundary

V1 provides no preview or execution surface. The WebUI uses an inert text editor plus an application-owned, bounded text-only Safe outline. The outline drops URLs, active/resource subtrees, notes from ordinary text, and generated chrome, and never loads or runs the document. The browser extension reads source-free metadata only and hands standalone projects to the canonical WebUI.

Saved standalone document source exists only in owner-authorized
detail/save/version-content/restore/export responses, component-local editor
memory, and capped 24-hour principal/origin/project-scoped `sessionStorage`
recovery. Owner-scoped generation input is retained for no more than 24 hours
and is transmitted only to the one bound provider target for the accepted
attempt. Saved source and generation input are excluded from global UI caches,
extension storage/messages, logs, traces, error payloads, source-free
summaries, and Jobs receipts.

### Generation And Persistence

`POST /api/v1/slides/generations` resolves and snapshots prompt/chat/media/notes/RAG input, claims an owner-scoped idempotency receipt, and enqueues `presentation.generate`. The in-process Jobs worker calls exactly one configured closed adapter/target per normal attempt, validates output in a killable bounded subprocess pool, and atomically commits Slides plus source-free Jobs result metadata. Provider side effects are at-least-once across a precommit crash.

Slides schema version 2 adds the content discriminator, standalone payload metadata, generation receipts, and ephemeral generation inputs. Migration is transactional, backup-first, and forward-only for old binaries. Standalone versions retain complete bounded snapshots; the newest 25 are kept by default.

List and search use source-free projections. Legacy clients that omit `X-Slides-Accept-Content-Kinds` see only structured rows. Opted-in summaries include kind, bounded provenance, slide count, and byte count but never source.

Standalone source saves use the complete raw document plus a strong ETag. There is no autosave or merge. The WebUI preserves local source on failure and requires an explicit overwrite, discard/load, or download choice after a conflict.

### Operations

Standalone generation is independently default-off and egress-gated. Configuration binds one exact provider/model/built-in-adapter/endpoint tuple and an environment-only HMAC keyring. Saved standalone read/edit/version/download access remains available when generation or provider egress is disabled.

The reconciler resolves Jobs/receipt ambiguity, enforces 24-hour input expiry, and retains terminal receipt metadata for 30 days. Logs and metrics contain bounded source-free metadata only. Rollback disables egress and generation, drains workers, retains schema v2 readability, and never starts an old binary against a migrated database.
