# Standalone HTML+JavaScript Presentations Design

**Date:** 2026-07-15

**Status:** Human-approved design; independently approved after three spec-review passes; awaiting requester review

**Backlog:** TASK-12115

**Related:** [Slides and Infographics Work Products PRD](../../Product/Slides_Infographics_Workproducts_PRD.md)

**Prompt source:** User-provided “HTML PPT Studio Agent” prompt, adapted by this design for tldw_server's actual runtime and security boundaries

## Summary

tldw_server will add a first-class `standalone_html` presentation content kind alongside the existing structured Slides model. A standalone HTML presentation is one saved, editable UTF-8 HTML document containing its own CSS and JavaScript. It can be generated from every source family currently supported by Slides—direct material, chat, media, notes, and RAG—through one shared generation service. The first product surface will be a form on `/presentation-studio/new`, followed by a dedicated code-and-preview workspace.

Standalone HTML is executable, untrusted model output. It is never treated as ordinary application markup, never rendered by a server-side browser, and never served inline from an application route. The Studio shows a sanitized static preview automatically. Interactive execution requires a user gesture for the current document version and runs in a minimal sandboxed `srcdoc` iframe with an application-enforced content security policy. This is best-effort browser containment, not a claim of complete network or resource isolation.

Existing structured presentations, clients, generation endpoints, exports, and render jobs keep their current behavior when the new mode is not requested.

## Context

The current Slides implementation has five relevant properties:

1. `SlidesGenerator` asks an LLM for structured JSON and normalizes it into a list of slide objects.
2. REST exposes separate generation routes for prompt, chat, media, notes, and RAG sources.
3. Presentations are stored in a per-user `Slides.db`, with immutable full-payload version snapshots and FTS text derived from slides.
4. Presentation Studio assumes every project has structured slides and uses a debounced autosave and automatic `412` merge path.
5. Reveal.js, Markdown, PDF, and video rendering assume trusted structured content rather than arbitrary JavaScript.

The supplied HTML prompt cannot be installed verbatim. It references a nonexistent `html-ppt-skill`, missing template and runtime directories, relative assets, CDN GSAP dependencies, and a multi-turn discovery conversation. It also contradicts itself by requiring both zero external dependencies and CDN assets. Presentation Studio v1 instead supplies the discovery values through a form and requires a self-contained browser-native document.

The existing product PRD lists arbitrary JavaScript as a non-goal. Implementing this feature therefore requires an explicit PRD policy update that narrows the exception to the bounded, untrusted `standalone_html` content kind described here.

## Goals

1. Generate complete HTML+CSS+JavaScript slide decks from direct material, chats, media, notes, and RAG sources through one mode-aware backend service.
2. Persist HTML decks as first-class Presentation Studio projects with list, get, title update, source update, version, search, duplicate, delete, and attachment-download behavior.
3. Give users a form-first creation flow and a dedicated source editor, static preview, opt-in interactive preview, explicit save, conflict recovery, and download workflow.
4. Preserve the existing structured Slides behavior and wire contracts by default.
5. Keep executable model output outside the application DOM and every server-side rendering path.
6. Enforce bounded requests, model output, stored documents, previews, versions, and generation retries.
7. Make unsupported operations and unavailable capabilities explicit rather than silently falling back or producing empty structured decks.
8. Preserve source ownership, authentication, provenance, citations supplied by source material, and per-user database isolation.

## Non-Goals

- Full PowerPoint, Keynote, or browser-based design-tool parity
- Converting a project between `structured_slides` and `standalone_html` in place
- Executing generated HTML in the browser extension or narrow sidepanel
- Server-side HTML execution, screenshots, thumbnails, PDF, video, or PPTX export
- Reveal.js or Marp conversion for standalone HTML projects
- Remote fonts, images, scripts, stylesheets, iframes, plugins, or CDN dependencies in v1
- A guarantee that arbitrary JavaScript is immune to CPU, memory, renderer, or self-navigation abuse
- A network-intercepted browser or container runtime in v1
- Automatic LLM repair loops for malformed or truncated HTML
- Importing standalone HTML or JSON presentation payloads in v1
- Collaborative editing, automatic document merges, or a visual HTML diff editor
- Full MCP HTML generation or returning full HTML documents inline in MCP tool results
- Exposing chat, media, notes, or RAG source pickers in the new form during the first UI release

## Product Decisions

- The persistent discriminator and request mode are both named `standalone_html`; the existing kind is `structured_slides`.
- Presentation content kind is immutable after creation. A later conversion feature must create a new presentation.
- HTML generation is asynchronous through the existing Jobs subsystem and accepts an idempotency key.
- One unified generation request supports all current source kinds. Existing per-source structured endpoints keep their current response contracts.
- The WebUI exposes direct-material HTML generation first. Other product surfaces may adopt the same backend contract later without another generator.
- HTML generation is disabled unless an administrator enables the feature and configures at least one allowed provider/model pair.
- Provider/model allowlisting improves operational control but does not make model output trusted.
- V1 documents use inline browser-native HTML, CSS, and JavaScript only. Semantic theme names influence generated CSS tokens but do not load theme files.
- Automatic preview is static and sanitized. Interactive execution is explicit, version-scoped, and never automatic after generation, editing, save, restore, duplication, or reload.
- HTML editing uses explicit Save. It does not reuse the structured editor's whole-project autosave or automatic conflict merge.
- Raw/current-draft HTML is downloadable only as an authenticated attachment. The application never offers “open preview in new tab.”
- Structured-only operations fail with a stable content-kind error at both API admission and worker execution.

## Terminology And Trust Model

### Canonical source

`html_document` is the user's canonical editable source. It is opaque untrusted text even when it was produced by an allowed model and passed generation validation. It may contain executable inline JavaScript.

### Static preview

A static preview is a derived, noncanonical view. Scripts, event handlers, forms, navigation elements, embedded browsing contexts, external URLs, refresh metadata, and active objects are removed. CSS capable of external requests is removed or neutralized. A restrictive CSP remains in force as defense in depth.

### Interactive preview

An interactive preview executes the current document's inline JavaScript after an explicit user action. It is isolated from the parent DOM and application APIs by an opaque-origin iframe sandbox. It is still capable of resource exhaustion and frame self-navigation. The UI must state that limitation before execution.

### Validation is not a sandbox

Static document validation catches malformed output, forbidden markup, direct resource references, and unreasonable size. It cannot prove arbitrary JavaScript safe or detect every dynamically constructed URL. Browser sandboxing, user consent, output limits, and the prohibition on server execution remain separate controls.

## User Experience

### Creation form

`/presentation-studio/new` keeps the existing structured option and adds an **Interactive HTML** option marked experimental. Capability discovery must complete before the option is enabled. Unknown or failed capability discovery fails closed and shows Retry rather than exposing a form that may submit to an older server.

The HTML form contains independently labelled fields:

1. **Subject and material** — required nonblank source text, using the existing source-size ceiling.
2. **Presentation type** — one of:
   - `pitch-deck`
   - `tech-sharing`
   - `product-launch`
   - `weekly-report`
   - `course-module`
   - `keynote`
   - `data-report`
   - `training`
   - `social-media`
   - `case-study`
   - `comparison`
   - `roadmap`
3. **Audience** — required bounded text.
4. **Approximate slide count** — integer from 1 through 30.
5. **Visual direction** — `auto` or one curated semantic direction. Initial directions are `dark-technical`, `minimal-light`, `editorial`, `corporate`, `soft-pastel`, `bold-creative`, and `neo-brutalist`.
6. **Delivery style** — explained radio choices:
   - `speaker-led`: include concise private presenter notes on every slide;
   - `self-running`: emphasize visible context and omit long presenter scripts.

The user may choose an allowed provider/model using the existing model selection pattern. The server, not the form, is authoritative for the allowlist.

Submitting the form:

- validates every field locally and server-side;
- creates one generation job using an idempotency key;
- disables duplicate submission while the request is pending;
- preserves all form values on error, timeout, or Stop waiting;
- shows real job state without invented percentage progress;
- uses “Stop waiting” unless Jobs provides genuine cancellation for the queued/running state;
- opens the created HTML project after the job reports its `presentation_id`.

The client creates a cryptographically random 16–200 character idempotency key for each deliberate submission and sends it in the `Idempotency-Key` header. It retains that key for transport retries, unknown submission outcomes, and resuming a stopped poll. Retrying the same canonical request with the same key returns the existing job or result instead of creating a duplicate presentation. Reusing the key with a different canonical request returns a conflict. After a known terminal generation failure, **Try again** preserves the form values but creates a new key because it is a deliberate new model attempt.

### HTML workspace

An HTML project's workspace is separate from the structured Slide Rail, Slide Editor, Media Rail, and render controls.

On desktop it uses a stable two-column code/preview grid. On narrow screens it uses explicit **Code** and **Preview** tabs. Resizable panes are deferred.

The primary controls are:

- **Save**
- **Run interactive preview**
- **Reset preview**
- **Download HTML**
- **Back to presentations**

The editor reuses the repository's lazy Monaco dependency and textarea fallback. It uses value/text APIs only; source is never inserted through `dangerouslySetInnerHTML`, Markdown rendering, error markup, or list snippets.

Editing immediately marks the project dirty, destroys any running interactive iframe, and updates the sanitized static preview after a bounded idle delay. The UI may retain the last valid static preview when the current buffer is temporarily malformed, but it must visibly mark the preview stale. Local size/preflight checks improve responsiveness but are not authoritative for save or execution.

Save states are user-facing and announced through `aria-live`: `Saved`, `Saving`, `Not saved`, and `Conflict`. A capped session recovery draft is retained per presentation and cleared only after the matching source revision saves successfully. A navigation/unload warning protects unsaved work.

On an optimistic-concurrency conflict, the Studio does not merge or retry automatically. It preserves the local buffer and offers:

- **Load server version**
- **Keep my version**, requiring an explicit force/replace confirmation against the newly fetched version
- **Download my draft**

The force path is an explicit user decision, not a normal autosave behavior.

### Navigation and source handoff

The Presentation Studio index includes a prominent **New presentation** action. HTML projects display a distinct badge and do not report an empty structured slide count as their primary summary.

The first UI release accepts pasted/direct material only. The backend source union is nevertheless complete. Later chat, media, notes, and RAG entry points hand off source identifiers or a short-lived server-side handoff key; they must not place full source content in a URL.

The browser extension can open the WebUI creation or project route. It does not execute generated JavaScript.

## Prompt Contract

### Storage and resolution

The packaged default is loaded through the existing prompt loader using the logical key `slides.standalone_html_system`. It lives in the Slides prompt asset set and keeps the repository's existing deployment-file/environment override behavior. A small built-in fallback protects startup if the packaged asset cannot be loaded.

The prompt is application-owned. Source material and form options are placed in clearly delimited user content and never concatenated as additional system instructions.

### Adaptation of the supplied prompt

The default preserves the useful portions of the supplied prompt:

- the twelve presentation-type narrative flows;
- one `<section class="slide">` per page;
- token-driven CSS;
- keyboard navigation;
- responsive layout and reduced-motion handling;
- concise audience-facing copy;
- optional private notes;
- a single complete HTML document.

It removes or changes incompatible requirements:

- Discovery questions are removed because the form supplies typed answers.
- References to `html-ppt-skill`, `templates/`, `base.css`, `runtime.js`, and theme files are removed.
- GSAP, ScrollTrigger, webfonts, Font Awesome, images, and other CDN references are removed.
- Browser-native CSS transitions, Web Animations, and small inline JavaScript replace required GSAP behavior.
- “Zero hardcoded colors” becomes: raw color values belong in `:root` theme tokens, while component rules consume variables.
- Presenter scripts are concise and generated only for `speaker-led`; 150–300 words on every slide is not required.
- Reduced-motion mode skips entrance animation and applies final visible state rather than freezing an animation timeline.
- The model does not ask follow-up questions, describe assumptions outside the document, or wrap output in Markdown fences.

### Required generated structure

The model must return exactly one complete document containing:

- `<!doctype html>`
- `<html>`, `<head>`, and `<body>`
- UTF-8 charset and viewport metadata
- a nonblank `<title>`
- inline CSS
- one to thirty `<section class="slide">` elements
- inline JavaScript for keyboard-first navigation
- accessible document landmarks and usable focus states
- a reduced-motion path that leaves every active element visible
- notes inside each slide only when requested, using a stable `.notes` class hidden from the audience view

The model must not return Markdown fences, relative resources, external URLs, remote imports, base URLs, frames, forms, popups, service workers, workers, storage-dependent state, analytics, telemetry, or network calls. It must not invent citations. When bounded source citations are supplied, it preserves them in visible or notes content as appropriate.

The generator tolerates and strips one outer Markdown fence before validation because some providers add one despite instructions. It does not perform a hidden repair call in v1.

## Generation Architecture

### Shared generation service

A mode-aware Slides generation service is the single orchestration boundary for new generation jobs. It owns:

- source-kind dispatch and ownership checks;
- source-size normalization and provenance;
- provider/model allowlist enforcement;
- prompt resolution and request assembly;
- provider timeout and output-token ceilings;
- mode dispatch to the existing structured generator or the standalone HTML generator;
- deterministic validation and normalization;
- atomic Slides persistence plus crash-reconciled Jobs result metadata.

REST and background workers use this service rather than duplicating normalization or persistence rules. Existing per-source structured REST routes remain compatibility wrappers with their current synchronous response shape. They may delegate internally, but omission of the new mode must execute the existing structured behavior.

### Unified generation request

`POST /api/v1/slides/generations` is the single new asynchronous submission route. In v1, `generation_mode` is required and its only accepted value on this route is `standalone_html`. Structured generation remains on the existing per-source endpoints; it does not gain a second public transport in this release. The new route requires the `Idempotency-Key` header and accepts a discriminated source union:

```json
{
  "generation_mode": "standalone_html",
  "source": {
    "kind": "prompt",
    "prompt": "..."
  },
  "title_hint": "Optional title",
  "provider": "configured-provider",
  "model": "allowed-model",
  "html_options": {
    "presentation_type": "tech-sharing",
    "audience": "backend engineers",
    "slide_count": 10,
    "visual_direction": "dark-technical",
    "delivery_style": "speaker-led"
  }
}
```

The exact source variants reuse the field names, ownership, and selection semantics of the current routes:

- `prompt`: `{ "kind": "prompt", "prompt": "..." }`;
- `chat`: `{ "kind": "chat", "conversation_id": "..." }`;
- `media`: `{ "kind": "media", "media_id": 123 }`;
- `notes`: `{ "kind": "notes", "note_ids": ["..."] }`;
- `rag`: `{ "kind": "rag", "query": "...", "top_k": 8 }`.

The endpoint returns `202 Accepted`:

```json
{
  "job_id": "job-uuid",
  "status": "queued",
  "status_url": "/api/v1/slides/generations/job-uuid",
  "presentation_id": null
}
```

`GET /api/v1/slides/generations/{job_id}` is owner-scoped and returns `job_id`, one of `queued | running | completed | failed | cancelled`, optional bounded progress text, and—only when applicable—`presentation_id`, `content_kind`, `error_code`, and a safe error message. It does not return generated HTML.

The job result contains bounded metadata such as `presentation_id`, `content_kind`, document byte count, slide count, and validation status. The client fetches the authenticated presentation detail after completion.

### Provider execution

Standalone HTML generation is available only when:

- the feature flag is enabled;
- an allowed provider/model pair is configured and healthy;
- the requested or resolved pair exactly matches that allowlist.

Custom OpenAI-compatible URLs and user-provided provider overrides do not bypass this check. Provider calls use the existing abstraction, run without blocking the FastAPI event loop, and have a server-enforced timeout and maximum output-token budget.

### Crash-safe idempotency and worker correlation

Slides and Jobs use separate persistence stores, so the design does not rely on a cross-database transaction.

Each user's Slides database contains a durable `slides_generation_receipts` ledger. A receipt stores the hashed client key, canonical request digest, bound Jobs UUID, optional committed presentation ID, creation/update timestamps, and expiry. The default idempotency guarantee is 30 days. Receipt cleanup and Jobs active/archive retention for this job type must both preserve lookup for at least that window.

At submission, the server canonicalizes the validated request and computes its SHA-256 digest. It claims the receipt in a Slides transaction before creating a job. Because each Slides database is already owner-scoped, the hashed key is unique for that owner. The endpoint compares an existing receipt's request digest:

- same digest: return or recover the bound job/result;
- different digest: return `409 generation_idempotency_conflict`.

Because the Jobs unique index is not owner-scoped, the server separately derives the internal Jobs key as `slides:v1:` plus the hexadecimal SHA-256 of the canonical owner ID, a NUL separator, and the client key. The raw client key is not accepted as the global Jobs key. The fixed scope is `domain=slides`, `queue=slides`, and `job_type=presentation.generate`.

If the process crashes after claiming the receipt but before binding the job, retry creates or retrieves the same Jobs row through that derived key and then binds its UUID to the receipt. Submission and status lookup query active and archived Jobs rows. A receipt with `presentation_id` is sufficient to return a completed result even if the Jobs row was later pruned. If a Jobs row disappears before the 30-day receipt expiry and no presentation was committed, the server returns `503 generation_receipt_unresolved`; it does not silently enqueue another model call.

Every generated presentation stores the immutable originating Jobs UUID as `generation_job_id`, protected by a unique index within the owner's Slides database. The worker algorithm is:

1. Load the receipt and return its committed presentation if present.
2. Resolve sources, call the model, and validate output.
3. In one Slides write transaction, recheck the receipt, insert the presentation with its initial entity version, derived search text, provenance, and `generation_job_id`, then set the receipt's `presentation_id`. If another retry already committed it, return that row instead.
4. Complete the Jobs result with the committed presentation ID.

If the worker crashes after step 3 but before step 4, retry finds the committed presentation through the receipt and completes the same job. Concurrent retries serialize on the receipt transaction and fetch the winning row. Jobs pruning cannot erase the presentation binding during the 30-day guarantee because the owner-scoped receipt is retained independently. After receipt expiry, the key is no longer guaranteed idempotent and clients must create a new deliberate-submission key.

## Standalone HTML Validation

One pure Python backend validator is authoritative after generation, before persistence, on every HTML save, through draft validation, and before saved-document export where applicable. The browser has a separate local preflight and sanitizer for responsiveness; it is never authoritative for persistence or interactive execution.

`POST /api/v1/slides/presentations/{presentation_id}/validate-html` accepts the current authenticated editor buffer without saving it. It applies the authoritative backend validator and returns only:

```json
{
  "valid": true,
  "html_sha256": "...",
  "html_bytes": 12345,
  "slide_count": 10,
  "diagnostics": []
}
```

Diagnostics are bounded safe codes and locations. The source is never echoed. **Run interactive preview** is enabled only after a successful validation response whose digest still matches the current buffer. Editing invalidates that result. The editor may download a malformed current draft for recovery, but it cannot save or execute it.

Default hard limits are:

- 1 MiB UTF-8 document size;
- 1 through 30 `.slide` sections;
- 10,000 parsed HTML elements;
- 20,000 total attributes;
- 256 KiB aggregate data-URI payload, with 128 KiB per item;
- 250,000 visible-text characters before indexing truncation;
- the existing source-input ceiling and a server-enforced generation token ceiling.

Limits are server-configurable downward. Raising them above packaged safety maxima requires an explicit advanced configuration setting.

Validation rejects:

- invalid UTF-8, NUL bytes, incomplete document structure, or likely truncation;
- missing doctype, head, body, title, or slide sections;
- excess size, slide count, nodes, attributes, or inline assets;
- relative, protocol-relative, HTTP, HTTPS, FTP, file, websocket, or other resource URLs;
- `<base>`, `<iframe>`, `<frame>`, `<object>`, `<embed>`, `<form>`, refresh metadata, and declarative popup/navigation elements;
- external script/style/font/media/image references;
- CSS `@import` and resource-bearing `url()` values;
- disallowed data-URI MIME types, including SVG and HTML;
- workers, service-worker registration, and manifest resources when statically identifiable.

Inline scripts remain allowed. Static inspection of their content supplies diagnostics for obvious network/navigation APIs but is not treated as a security decision.

Validation derives:

- a normalized title from `<title>` or the bounded title hint;
- slide count;
- UTF-8 byte count;
- SHA-256 source digest;
- bounded visible search text excluding script, style, template, noscript, notes marked private, and hidden chrome;
- safe diagnostic codes without echoing source or secrets into logs.

Invalid generated output completes the job as failed with a stable validation code and bounded field diagnostics. The form retains its inputs and offers Retry. V1 does not silently fall back to Markdown or structured slides.

## Persistence And Invariants

### Presentation record

The presentation record gains:

- `content_kind TEXT NOT NULL DEFAULT 'structured_slides'`
- `html_document TEXT NULL`
- `html_sha256 TEXT NULL`
- `html_bytes INTEGER NULL`
- `generation_job_id TEXT NULL`

The existing `slides` column remains non-null.

The owner-scoped `slides_generation_receipts` table contains:

- `idempotency_key_sha256 TEXT PRIMARY KEY`
- `request_sha256 TEXT NOT NULL`
- `job_uuid TEXT NULL`
- `presentation_id TEXT NULL`, referencing `presentations.id`
- `created_at`, `updated_at`, and `expires_at`

It stores no raw client key, source material, prompt, model output, or HTML. The presentation foreign-key behavior preserves a committed binding for the full receipt-retention window.

The canonical invariant is:

- `structured_slides`: validated slide list; `html_document`, `html_sha256`, and `html_bytes` are null.
- `standalone_html`: nonblank validated `html_document`; matching digest and byte count; stored `slides` is `[]`.

`slides_text` remains the canonical derived FTS source for both kinds. Clients can never supply it.

All create, replace, patch, restore, duplicate, REST, MCP, and worker paths enforce the invariant through one domain service. Partial updates first merge with the current record, then validate the complete candidate inside the optimistic-concurrency operation. Omitting `content_kind` preserves the current kind; it never converts an HTML project into an empty structured project. A partial unique index on nonnull `generation_job_id` provides worker retry deduplication. User duplication creates a new presentation with a null generation job ID.

Content kind cannot change in v1. A request that attempts to change it returns `409 content_kind_immutable`.

### Migration

Slides schema versioning becomes authoritative for this change. Migration runs under a SQLite write transaction (`BEGIN IMMEDIATE`), rechecks schema state after acquiring the lock, adds/backfills the presentation fields, creates the receipt table and indexes, and advances the schema version atomically. Legacy rows become `structured_slides` with null HTML fields.

Tests cover a legacy database, an already migrated database, and concurrent first access from separate connections/processes. New row mapping and summary queries use explicit projections rather than depending on `SELECT *` positional/dataclass compatibility. Deployment documentation requires a database backup before upgrade and treats this migration as forward-only for old binaries.

### Summaries and search

List and search queries use lightweight projections that exclude `html_document` and full version payloads. Summaries include `content_kind`, title, provenance summary, timestamps, version, and either structured slide count or HTML slide/byte metadata.

HTML search indexes only the bounded visible text derived by the server. Raw markup and JavaScript are never indexed or emitted as result snippets.

### Version snapshots

`presentation.version` is the entity revision used by ETags. It advances on every accepted mutation that changes any canonical mutable field, including title, provenance, Studio metadata, or HTML source. A snapshot is a complete entity snapshot for that revision; v1 does not introduce a separate content-blob version model.

New snapshots include:

- `snapshot_schema_version`
- `content_kind`
- the active content payload only
- source digest and byte/slide metadata
- title, provenance, and other existing mutable presentation fields

Snapshots without a content kind are interpreted as `structured_slides`. Restore validates the snapshot against the current kind and content policy before one atomic update. A mismatched/corrupt kind is rejected rather than converted.

The SHA-256 digest is an optimization for comparing HTML source, not the entity-version definition. A merged update that is byte-for-byte and metadata-for-metadata identical is a no-op and creates neither a new entity revision nor a snapshot. If HTML is unchanged but title, provenance, or another canonical field changes, the entity version advances and a complete snapshot is created.

To bound full-document amplification, standalone HTML retains the newest 25 entity snapshots per presentation by default; retention is configurable downward. Retention cleanup occurs only after a successful new snapshot and never removes the current entity. Full-document duplication inside those bounded snapshots is accepted for v1 instead of introducing delta storage.

## API Contracts

### Capabilities

Capabilities separate persistence/editor support from generation availability. A temporary model or configuration problem must never make already-saved HTML projects inaccessible. The contract is conceptually:

```json
{
  "presentation_content_kinds": {
    "structured_slides": {
      "read": true,
      "edit": true
    },
    "standalone_html": {
      "read": true,
      "edit": true,
      "export_attachment": true,
      "interactive_preview_policy": "explicit_best_effort",
      "max_document_bytes": 1048576,
      "max_slides": 30
    }
  },
  "presentation_generation_modes": {
    "structured_slides": {
      "enabled": true,
      "transport": "existing_source_endpoints"
    },
    "standalone_html": {
      "enabled": false,
      "reason": "no_allowed_model",
      "transport": "slides_generation_job"
    }
  }
}
```

Servers implementing this schema advertise `standalone_html` read/edit/export support even when generation is disabled. The creation form checks `presentation_generation_modes.standalone_html.enabled`; list, detail, editor, validation, versions, and attachment download check the content-kind capability. The frontend does not infer either axis from unrelated Slides routes.

### Presentation response

Detail responses form a discriminated union:

- structured detail contains `content_kind: "structured_slides"` and `slides`;
- HTML detail contains `content_kind: "standalone_html"`, `html_document`, `html_sha256`, `html_bytes`, and derived slide count.

HTML source is present only in authenticated detail/version-content responses, not summaries, search snippets, job summaries, error bodies, or logs. Unknown content kinds render as an unsupported read-only state in clients.

### Update and concurrency

HTML Save uses a kind-aware PATCH with a strong ETag/`If-Match` contract. The server returns strong tags such as `"v7"` and temporarily accepts legacy weak tags produced by existing clients. A stale tag returns `412` with bounded current-version metadata, not the remote HTML body unless the client explicitly fetches it.

HTML updates may change title and `html_document` together. Structured-only fields explicitly supplied for an HTML record are rejected rather than ignored. A successful save returns the new detail and ETag. Only complete merged-entity equality is a no-op; an unchanged HTML digest does not suppress a title, provenance, or other metadata revision.

### Operation matrix

| Operation | Structured slides | Standalone HTML |
| --- | --- | --- |
| List/get/search/delete/duplicate | Supported | Supported |
| Title/source update and versions | Supported | Supported |
| Reorder or per-slide mutation | Supported | Rejected |
| JSON export | Supported | Supported, with HTML source only in authenticated explicit export |
| Reveal.js/Markdown/Marp | Supported | Rejected |
| PDF/video/thumbnail/render job | Supported | Rejected at API and worker |
| HTML attachment | Existing Reveal behavior | Exact standalone source attachment |
| Interactive WebUI preview | Not applicable | Explicit best-effort sandbox |
| Extension execution | Not applicable | Rejected/handoff only |

Structured-only rejections use `409 operation_not_supported_for_content_kind` and include the operation and actual kind.

### HTML attachment and draft recovery

Saved HTML export uses the presentation export endpoint with `format=html`. Downloading the current unsaved editor buffer uses an authenticated POST attachment endpoint scoped to the presentation; it validates ownership, UTF-8/no-NUL, and the hard byte ceiling but does not persist the buffer or require it to be a valid deck. This makes malformed-draft recovery possible.

Both responses use a sanitized filename and:

- `Content-Type: application/octet-stream`
- `Content-Disposition: attachment`
- `X-Content-Type-Options: nosniff`
- `X-Download-Options: noopen`
- `Cache-Control: private, no-store`
- `Referrer-Policy: no-referrer`
- `Cross-Origin-Resource-Policy: same-origin`

Saved-version downloads include ETag and Last-Modified. No route serves stored HTML inline as `text/html`. Before download, the UI states that the file contains executable code and that opening it locally occurs outside Presentation Studio's sandbox and CSP wrapper.

## Preview Security Contract

### Static preview pipeline

The automatic preview:

1. parses the current source without executing it;
2. removes scripts, event-handler attributes, active embeds, forms, base/refresh metadata, external references, and navigation-capable markup;
3. strips CSS imports and resource URLs;
4. serializes content into an application-built wrapper whose charset and CSP precede every untrusted byte;
5. renders the result in a sandboxed iframe.

The static iframe is created through `srcdoc` with an empty sandbox attribute (`sandbox=""`), `referrerPolicy="no-referrer"`, the same deny-all Permissions Policy used by the interactive iframe, a fixed title, and no `allow-*` capability. Its application-built wrapper enforces:

```text
default-src 'none';
script-src 'none';
style-src 'unsafe-inline';
img-src data:;
font-src data:;
connect-src 'none';
media-src 'none';
worker-src 'none';
frame-src 'none';
child-src 'none';
object-src 'none';
manifest-src 'none';
base-uri 'none';
form-action 'none'
```

Only sanitizer-approved capped raster/font data URIs may remain. The application must not inject a CSP into an arbitrary model string with a regular expression. Static preview never calls the backend validation endpoint merely to repaint; the backend remains authoritative when the user saves or requests interactive Run.

### Interactive preview pipeline

Interactive Run is enabled only for a buffer that passes full standalone HTML validation. The action presents concise risk copy explaining that generated JavaScript can freeze the preview or navigate its own frame. Consent is scoped to the current source digest.

The application parses the source and builds a fixed preview wrapper. It removes untrusted CSP, `<base>`, refresh metadata, and forbidden resource elements before serializing permitted markup and inline scripts after the trusted policy. It sets the iframe through `srcdoc`; it does not create an application-origin Blob URL.

The outer iframe uses exactly:

- `sandbox="allow-scripts"`
- `referrerPolicy="no-referrer"`
- a restrictive Permissions Policy/`allow` value denying camera, microphone, geolocation, clipboard, display capture, payment, USB, serial, HID, Bluetooth, MIDI, sensors, and fullscreen
- a fixed title and bounded 16:9 viewport

The iframe attribute, not a model-supplied or meta CSP `sandbox` directive, enforces sandboxing. The application-owned meta CSP is the first policy element in the `srcdoc` wrapper and is conceptually:

```text
default-src 'none';
script-src 'unsafe-inline';
style-src 'unsafe-inline';
img-src data:;
font-src data:;
connect-src 'none';
media-src 'none';
worker-src 'none';
frame-src 'none';
child-src 'none';
object-src 'none';
manifest-src 'none';
base-uri 'none';
form-action 'none'
```

Only validator-approved capped raster image/font data URIs may survive. SVG/HTML data documents and `blob:` script, worker, frame, or object sources are not permitted.

V1 uses no parent/preview `postMessage` command bridge. A sandboxed deck can still call `parent.postMessage`, so message events from the preview are treated as attacker-controlled no-ops. The preview component registers no functional message listener. Every application-level `message` consumer must require its own expected `event.source`, origin, strict schema, and unguessable per-instance token, and must explicitly reject a registered preview frame's `contentWindow`. Repository tests enumerate existing message consumers and verify that preview-origin messages cannot invoke APIs, modify the editor, copy to clipboard, download, navigate, resize, or trigger any other side effect. Fixed dimensions remove any need for a preview resize channel.

Any edit, saved-version change, restore, duplication, regeneration, or unmount destroys the running iframe. There is no automatic rerun and no open-in-new-tab control.

### Explicit limitations

The CSP blocks ordinary fetch, image, media, worker, and frame egress, but a script can still attempt to navigate its own frame. A tight JavaScript loop can still consume renderer resources. The warning and documentation state these limits. A future network-intercepted, process-isolated runtime is required before the product can claim no egress or reliable preemption.

## MCP And Secondary Consumers

The MCP Slides module currently duplicates REST generation, persistence, restore, and export behavior. This feature must not add another independent implementation.

The shared domain service becomes authoritative for content invariants, summary mapping, update/restore, operation guards, and export dispatch. V1 behavior is mandatory:

- MCP list and search results always include `content_kind`. HTML summaries include `html_bytes`, derived slide count, and `html_available: true`; they never include source.
- MCP get preserves the current structured response for structured records. For HTML it returns the same bounded metadata plus provenance/version fields and omits `html_document`.
- MCP structured generation remains supported.
- MCP HTML generation, source retrieval, source mutation, restore, and attachment transfer return a typed unsupported tool result whose data includes `code: operation_not_supported_for_content_kind`, `operation`, and `content_kind`.
- Any remaining MCP mutation or export that assumes structured slides performs the same guard before touching persistence.

A later artifact/resource-handle design may expose HTML through MCP without embedding large executable documents in tool results.

The shared WebUI/extension API types also always preserve `content_kind`. In extension runtime contexts:

- list/search include HTML records and their bounded metadata rather than filtering or coercing them to empty structured decks;
- selecting an HTML record shows its title, kind, provenance summary, and an **Open in WebUI** handoff;
- the extension does not request `html_document`, mount the HTML workspace, create a preview iframe, or expose Run;
- an unknown future content kind uses the same metadata-only handoff/read-only fallback;
- structured records retain their existing extension behavior.

## Error Handling

Stable errors include:

- `standalone_html_not_supported`
- `standalone_html_model_not_allowed`
- `standalone_html_output_too_large`
- `standalone_html_invalid_document`
- `standalone_html_storage_limit`
- `content_kind_immutable`
- `operation_not_supported_for_content_kind`
- `presentation_version_conflict`
- `generation_idempotency_key_required`
- `generation_idempotency_conflict`
- `generation_receipt_unresolved`

Use `400` for a missing/invalid idempotency header, `413` for hard byte/token/asset ceilings, `422` for malformed documents or invalid options, `409` for kind/operation/idempotency conflicts, `412` for stale ETags, `503` for an unresolved receipt whose Jobs row disappeared inside the guarantee window, and the existing Jobs failure representation for provider or worker failures.

Diagnostics identify bounded fields, limits, and machine-readable codes. They do not echo source documents, prompt bodies, API keys, model output, notes, or JavaScript into logs or error responses.

## Compatibility

- Existing request omission follows the current structured path exactly.
- Existing per-source generation endpoints keep their response types and behavior.
- Legacy database rows and version snapshots default to `structured_slides`.
- List clients receive an additive `content_kind` field; HTML source stays out of summary payloads.
- Frontend normalization preserves both discriminated variants and never converts unknown/missing HTML into `slides: []` for mutation.
- Old servers are detected through explicit capabilities; the HTML form never assumes support from a generic Slides route.
- JSON export preserves the discriminator and active payload. Presentation import remains out of scope for v1.
- Structured Reveal, PDF, and render workers retain their existing behavior, with an added kind check.
- The browser extension can list or hand off HTML projects but cannot preview them interactively.

## Observability And Operations

Record only safe metadata:

- generation mode and source kind;
- provider/model identifier after secret removal;
- job duration and terminal state;
- document byte and slide counts;
- validation/error code;
- preview mode selection as local product telemetry only if the project's no-telemetry posture permits local logs; no external telemetry is introduced.

Never log source material, full HTML, JavaScript, notes, prompts, API keys, or download bodies. Existing auth, per-user isolation, rate limiting, and Jobs ownership checks apply.

Standalone HTML **generation** is disabled by default until configured. Startup validates the provider/model allowlist and packaged prompt asset. Invalid generation configuration keeps structured generation and standalone HTML read/edit/export available, advertises the HTML generation mode as disabled with a safe reason code, and logs a safe administrator-facing reason.

## Testing Strategy

### Backend unit and property tests

- request/content-kind enum mapping and cross-field invariants;
- partial-update merge-before-validation behavior;
- immutable kind and structured-only operation guards;
- outer-fence removal and deterministic prompt assembly;
- every presentation type, visual direction, delivery style, and boundary slide count;
- HTML parser limits, malformed/truncated documents, URL/CSS/data-URI rejection, title/search extraction, and digest calculation;
- generated source never appears in logs or error messages;
- property/fuzz cases for parser limits and invariant preservation;
- exact no-op saves, metadata-only revisions with unchanged HTML, and version retention.

### Database and migration tests

- legacy schema migration and default structured rows;
- repeated and concurrent migration attempts;
- create/get/update/search/duplicate/delete for both kinds;
- lightweight summaries do not load full HTML;
- old and new snapshot restore behavior;
- FTS excludes scripts/styles/private notes;
- strong ETags and legacy weak-tag acceptance;
- generation-receipt claim, bind, atomic presentation commit, replay, and retention cleanup.

### API, Jobs, and MCP integration tests

- successful HTML generation from prompt, chat, media, notes, and RAG with mocked LLMs;
- model allowlist, feature capability, ownership, size, timeout, and idempotency behavior;
- same-key/same-digest replay, same-key/different-digest conflict, cross-owner isolation, receipt-claim/job-bind crash recovery, crash recovery after the presentation commit but before Jobs completion, active/archive lookup, premature Jobs-row loss failure, and 30-day receipt expiry semantics;
- failed validation creates no presentation;
- job result contains metadata rather than HTML;
- draft validation returns a digest-bound verdict without persistence or source echo;
- saved HTML remains readable/editable/exportable while HTML generation is disabled;
- all structured-only operations reject HTML before dispatch;
- render workers independently recheck content kind;
- saved and draft download headers/content;
- MCP cannot mutate/export HTML through structured assumptions;
- omission of mode preserves current structured behavior.

### Frontend tests

- capability loading, failure, unsupported, and supported states;
- form validation, payload mapping, duplicate-submit prevention, preserved retry state, and job completion handoff;
- discriminated client normalization and unknown-kind fallback;
- code/preview responsive layout and absence of structured controls;
- dirty/save/error/conflict/recovery behavior;
- editing invalidates interactive consent and destroys the active preview;
- static preview sanitization and current-buffer download request;
- static iframe's empty sandbox/CSP and interactive Run's authoritative current-digest validation;
- keyboard access, labels, focus behavior, `aria-live`, and reduced motion.

### Adversarial browser tests

- generated scripts cannot access the parent DOM or application storage;
- no `allow-same-origin`, forms, popups, downloads, modals, top navigation, or privileged Permissions Policy capability;
- direct external resource attempts are blocked;
- self-navigation is detected/documented as an accepted best-effort limitation rather than claimed impossible;
- malicious base/meta-refresh/pre-policy scripts cannot precede the application policy;
- no preview Blob URL or open-in-new-tab path exists;
- iframe teardown occurs after edits and route changes;
- oversized DOM/data assets are rejected before execution;
- a keyboard-focused desktop/mobile E2E covers generate, edit, static preview, explicit Run, save, conflict recovery, reopen, and download.

### Quality gates

- Focused Python and frontend tests pass.
- Relevant integration and browser E2E tests pass.
- Type/lint checks pass for touched files.
- Bandit runs against every touched Python scope with no new findings.
- Documentation and API examples match the implemented wire contract.

## Documentation Changes

Implementation updates:

- `Docs/API/Slides.md`
- the Slides core README and configuration documentation
- Presentation Studio user documentation
- the product PRD's arbitrary-JavaScript non-goal and security section
- deployment guidance for capability enablement, model allowlisting, migration backup, preview limitations, and attachment handling

The PRD exception must be narrow: arbitrary JavaScript remains prohibited in structured exports and all server renderers; only explicit `standalone_html` projects use the untrusted-code workflow.

## Delivery Decomposition

This is one product design but should be delivered as reviewable child tasks:

1. Shared contracts, schema migration, persistence invariants, version/search behavior, and operation guards.
2. Prompt asset, validator, shared generation service, Jobs integration, source variants, and mocked backend tests.
3. Capabilities, HTML attachment endpoints, MCP guards, and security headers.
4. Presentation Studio form, discriminated client/store model, and generation-job UX.
5. HTML editor, static preview, explicit interactive preview, save/conflict/recovery, responsive/accessibility behavior, and browser security tests.
6. Documentation, PRD reconciliation, end-to-end verification, Bandit, and rollout evidence.

Each child task must remain independently reviewable and may tighten implementation details without weakening this design's content invariant or security boundary.

## Success Criteria

The feature is complete when an authenticated user can submit direct material through the new form, receive one idempotently generated first-class HTML presentation, inspect it without automatic code execution, explicitly run it in the bounded preview, edit and save without silent conflicts, reopen and search it, and download the current source as an attachment.

The same backend service must successfully generate HTML from mocked chat, media, notes, and RAG sources even though those source selectors are not yet exposed in the form. Existing structured Slides behavior must remain compatible. No server renderer, MCP structured operation, browser-extension surface, or older client may execute or silently reinterpret the HTML payload.
