# Standalone HTML+JavaScript Presentations Design

**Date:** 2026-07-15

**Status:** Requester-approved V1 design; independently approved; awaiting requester review before implementation planning

**Backlog:** TASK-12115

**Related:** [Slides and Infographics Work Products PRD](../../Product/Slides_Infographics_Workproducts_PRD.md)

**Prompt source:** User-provided “HTML PPT Studio Agent” prompt, adapted by this design for tldw_server's product and security boundaries

## Summary

tldw_server will add a first-class `standalone_html` presentation content kind alongside the existing structured Slides model. A standalone HTML presentation is one saved, editable UTF-8 HTML document containing its own CSS and JavaScript. It can be generated from every source family currently supported by Slides—direct material, chat, media, notes, and RAG—through one shared generation service. The first product surface will be a form on `/presentation-studio/new`, followed by a dedicated code-and-outline workspace.

Standalone HTML is executable, untrusted model output. It is never treated as ordinary application markup, never executed or rendered by tldw_server or its WebUI, and never served inline from an application route. Presentation Studio shows only a trusted safe outline that discards model markup, CSS, scripts, attributes, and assets. Users may download the source as an attachment and run it outside tldw at their own discretion; V1 provides no in-application execution path.

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

1. Generate complete HTML+CSS+JavaScript slide decks from direct material, chats, media, notes, and RAG sources through one shared standalone-HTML backend service.
2. Persist HTML decks as first-class Presentation Studio projects with list, get, search, version, restore, delete, explicit-save, and attachment-download behavior.
3. Give users a form-first creation flow and a dedicated source editor, automatic safe outline, explicit save, conflict recovery, and download workflow.
4. Preserve the existing structured Slides behavior and wire contracts by default.
5. Keep executable model output outside the application DOM and every server-side rendering path.
6. Enforce bounded requests, model output, stored documents, safe outlines, versions, and generation retries.
7. Make unsupported operations and unavailable capabilities explicit rather than silently falling back or producing empty structured decks.
8. Preserve source ownership, authentication, provenance, citations supplied by source material, and per-user database isolation.

## Non-Goals

- Full PowerPoint, Keynote, or browser-based design-tool parity
- Converting a project between `structured_slides` and `standalone_html` in place
- Executing or fidelity-rendering generated HTML anywhere inside tldw_server, the WebUI, the browser extension, or the narrow sidepanel in v1
- Server-side HTML execution, screenshots, thumbnails, PDF, video, or PPTX export
- Reveal.js or Marp conversion for standalone HTML projects
- Remote or data-URI fonts, images, scripts, stylesheets, iframes, plugins, or CDN dependencies referenced by generated documents in v1
- An iframe, browser sandbox, network-intercepted browser, or container execution runtime in v1
- Automatic LLM repair loops for malformed or truncated HTML
- Importing standalone HTML or JSON presentation payloads in v1
- Collaborative editing, automatic document merges, or a visual HTML diff editor
- Full MCP HTML generation or returning full HTML documents inline in MCP tool results
- Exposing chat, media, notes, or RAG source pickers in the new form during the first UI release
- A provider/model picker for standalone HTML generation in v1
- Standalone HTML duplication, independent title editing, or mutable generation provenance in v1

## Product Decisions

- The persistent discriminator and request mode are both named `standalone_html`; the existing kind is `structured_slides`.
- Presentation content kind is immutable after creation. A later conversion feature must create a new presentation.
- HTML generation is asynchronous through the existing Jobs subsystem and accepts an idempotency key.
- One unified generation request supports all current source kinds. Existing per-source structured endpoints keep their current response contracts.
- The WebUI exposes direct-material HTML generation first. Other product surfaces may adopt the same backend contract later without another generator.
- HTML generation is disabled unless an administrator enables the feature and configures one default provider/model pair contained in the standalone HTML allowlist.
- Provider/model allowlisting improves operational control but does not make model output trusted.
- V1 documents use inline browser-native HTML, CSS, JavaScript, and optional inline SVG only. All URL-bearing resources, including `data:` assets, are rejected. Semantic theme names influence generated CSS tokens but do not load theme files.
- The automatic view is a trusted safe outline, not a fidelity preview. V1 never executes generated JavaScript in the application, including after generation, editing, save, restore, or reload.
- HTML editing uses explicit Save. It does not reuse the structured editor's whole-project autosave or automatic conflict merge.
- Raw/current-draft HTML is downloadable only as an authenticated attachment. The application never offers an inline or **Open in new tab** view.
- Structured-only operations fail with a stable content-kind error at both API admission and worker execution.
- The HTML document's normalized `<title>` is the project title. V1 has no separate HTML title control that can drift from the source.

## Terminology And Trust Model

### Canonical source

`html_document` is the user's canonical editable source. It is opaque untrusted text even when it was produced by an allowed model and passed generation validation. It may contain executable inline JavaScript.

### Safe outline preview

A safe outline preview is a derived, noncanonical view of slide boundaries and bounded semantic text. The application parses without executing, extracts text from headings, paragraphs, lists, tables, figures, and notes, and rebuilds the view with trusted components and `textContent`. It preserves no model markup, CSS, style attributes, event handlers, images, fonts, SVG, MathML, templates, links, or active objects. It is intentionally not a visual-fidelity preview.

### Execution boundary

Generated JavaScript is stored and edited as text only. The WebUI creates no iframe, `srcdoc`, Blob URL, data URL, popup, worker, server-render request, or other execution context from `html_document`. No control enables in-application execution in V1. Download warnings make clear that opening the attachment leaves tldw's security boundary.

### Validation does not make code safe

Static document validation catches malformed output, forbidden markup, direct resource references, and unreasonable size. It cannot prove arbitrary JavaScript safe or detect every dynamically constructed URL. That limitation is why validation never enables execution; output limits and the prohibition on all tldw execution remain separate controls.

## User Experience

### Creation form

`/presentation-studio/new` keeps the existing structured option and adds a **Standalone HTML + JavaScript** option marked experimental, with adjacent copy: **Studio shows a text-only outline; generated code runs only if you download and open the file outside tldw.** Capability discovery must complete before the option is enabled. Unknown or failed capability discovery fails closed and shows Retry rather than exposing a form that may submit to an older server.

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
   - `speaker-led`: include concise speaker notes in the HTML, hidden from audience view;
   - `self-running`: emphasize visible context and omit long presenter scripts.

V1 has no provider/model picker. The server resolves the administrator-configured default allowlisted pair and returns its nonsecret identifiers in capability metadata so the form can explain what will run.

Submitting the form:

- validates every field locally and server-side;
- creates one generation job using an idempotency key;
- disables duplicate submission while the request is pending;
- preserves all form values on error, timeout, or Stop waiting;
- shows real job state without invented percentage progress;
- uses “Stop waiting” unless Jobs provides genuine cancellation for the queued/running state;
- opens the created HTML project after the job reports its `presentation_id`.

Before sending the POST, the client stores bounded resume metadata—`{ jobId: number | null, idempotencyKey, requestDigest, timestamp }`—in `sessionStorage`; it fills `jobId` after receiving `202`. A separate capped form draft preserves the canonical entered fields for replay and terminal-error recovery. Both keys are namespaced by canonical server origin and a stable, nonsecret authenticated subject/client identifier. Before hydration, the application derives the current principal scope from trusted auth state and compares it with the stored namespace; it never trusts a persisted scope value to choose an account. Records are ignored and cleared on subject change, logout, successful project handoff, or explicit Discard. If session storage fails, the current in-memory form remains usable and the UI warns that reload recovery is unavailable. On reload it offers **Resume** or **Discard**: Resume polls when `jobId` exists, otherwise it resubmits the preserved canonical request with the same idempotency key. Polling uses bounded exponential backoff and handles authentication loss, missing jobs, throttling, server errors, cancellation, quarantine, and a completed job missing its presentation binding without discarding the form state.

The client creates a cryptographically random 16–200 character idempotency key for each deliberate submission and sends it in the `Idempotency-Key` header. It retains that key for transport retries, unknown submission outcomes, and resuming a stopped poll. Retrying the same canonical request with the same key returns the existing job or result instead of creating a duplicate presentation. Reusing the key with a different canonical request returns a conflict. After a known terminal generation failure, **Try again** preserves the form values but creates a new key because it is a deliberate new model attempt.

### HTML workspace

An HTML project's workspace is separate from the structured Slide Rail, Slide Editor, Media Rail, and render controls.

On desktop it uses a stable two-column code/outline grid. On narrow screens it uses explicit **Code** and **Outline** tabs. Resizable panes are deferred.

The primary controls are:

- **Save**
- **Download current draft**
- **Back to presentations**

The editor reuses the repository's lazy Monaco dependency and textarea fallback. It uses value/text APIs only; source is never inserted through `dangerouslySetInnerHTML`, Markdown rendering, error markup, or list snippets.

Editing immediately marks the project dirty and updates the safe outline after a bounded idle delay. Outline state tracks the buffer digest, rendered digest, and `current | stale | failed` state. Parsing is latest-wins; stale work cannot replace a newer outline. Local size/preflight checks improve responsiveness but are not authoritative for save.

The automatic view is visibly labelled **Safe outline — text only; code never runs in Studio**. No fidelity or interactive state exists.

The editor enforces the same fixed 1 MiB UTF-8 hard ceiling as the server. A paste or edit that would cross it is rejected while preserving the previous buffer and announcing the limit. Therefore every accepted draft remains eligible for the authenticated draft-download endpoint.

Per-project recovery uses capped `sessionStorage`, never `localStorage` or extension storage. A record is `{ schemaVersion, principalScope, presentationId, baseEtag, baseDigest, source, updatedAt }`, where `principalScope` is derived from the canonical server origin and a stable, nonsecret authenticated subject/client identifier. The application computes the current scope before reading `source`, compares it in constant application logic, and never lets persisted data select the active account. Records expire after 24 hours or the browser session, are ignored and cleared on subject change, logout, or Discard, and never autoapply. Full HTML remains component-local plus this scoped recovery record: it is excluded from React Query/SWR or equivalent global/persisted caches, Redux-style action/devtools logging, analytics, and error-report payloads. When a nonmatching draft is found, the workspace offers **Restore draft**, **Download draft**, or **Discard draft** before replacing the editor buffer. A matching saved digest clears the record. If browser quota or storage access fails, the UI keeps the in-memory buffer, shows a persistent **Recovery unavailable** warning, and retains the navigation/unload warning.

Save states are user-facing and announced through `aria-live`: `Saved`, `Saving`, `Not saved`, and `Conflict`. A capped session recovery draft is retained per presentation and cleared only after the matching source revision saves successfully. A navigation/unload warning protects unsaved work.

On an optimistic-concurrency conflict, the Studio does not merge or retry automatically. It preserves the local buffer and offers:

- **Load server version**
- **Keep my version**, requiring an explicit force/replace confirmation against the newly fetched version
- **Download my draft**

The force path is an explicit user decision, not a normal autosave behavior. **Keep my version** first refetches the current entity and ETag, asks for replacement confirmation, and retries against that ETag; a second race returns to the same conflict state rather than overwriting silently.

If a save response is lost, the Studio refetches the detail. When the fetched canonical source digest and server-derived title equal the complete local candidate, it treats the save as confirmed and adopts the returned ETag. Otherwise it keeps the recovery draft and shows `Not saved` or `Conflict` as appropriate.

### Navigation and source handoff

The Presentation Studio index ships a minimal real project list rather than its current informational copy: a prominent **New presentation** action, recent projects, kind badges, open actions, cursor/page-based **Load more**, and explicit loading, empty, error, Retry, and offline states. Every project remains reachable through pagination. HTML projects do not report an empty structured slide count as their primary summary. User-facing search and richer list management remain separate follow-up work; backend HTML search indexing remains part of this release.

The first UI release accepts pasted/direct material only. The backend source union is nevertheless complete. Later chat, media, notes, and RAG entry points hand off source identifiers or a short-lived server-side handoff key; they must not place full source content in a URL.

The browser extension shows metadata only and an **Open in WebUI** handoff. It does not request HTML detail/version bodies or execute generated JavaScript.

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
- optional speaker notes embedded in the HTML and hidden from audience view;
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
- inline CSS contained in `<style>` elements, with no `style` attributes
- one to thirty `<section class="slide">` elements
- exactly one attribute-free classic inline `<script>` as the final child of `<body>`, containing self-contained browser-native keyboard navigation for the downloaded file
- accessible document landmarks and usable focus states
- a reduced-motion path that leaves every active element visible
- notes inside each slide only when requested, using a stable `.notes` class hidden from the audience view

The model must not return Markdown fences, additional scripts, script attributes or module syntax, event-handler or `style` attributes, URL-bearing attributes, CSS `url()` or `@font-face`, relative/remote/data/blob resources, remote imports, base URLs, frames, forms, popups, service workers, workers, storage-dependent state, analytics, telemetry, source-map directives, or network calls. It must not invent citations. When bounded source citations are supplied, it preserves them as nonlinked visible text or notes content as appropriate.

The generator tolerates and strips one outer Markdown fence before validation because some providers add one despite instructions. It does not perform a hidden repair call in v1.

## Generation Architecture

### Shared standalone HTML generation service

A standalone HTML Slides generation service is the single orchestration boundary for the new job across all five source adapters. It owns:

- submission-time source-kind dispatch, ownership checks, bounded snapshotting, and provenance;
- provider/model allowlist enforcement;
- prompt resolution and request assembly;
- provider timeout and output-token ceilings;
- deterministic validation and normalization;
- atomic Slides persistence plus crash-reconciled Jobs result metadata.

The new REST route and background worker use this service rather than duplicating normalization or persistence rules. V1 does not refactor the existing structured generator or its per-source REST routes; those routes keep their current synchronous response shapes and behavior when their existing requests omit any new mode field.

Source adapters accept explicit per-user database/service dependencies; they do not depend on a FastAPI `Request` object or endpoint-only dependency overrides. The submission path resolves and snapshots source content before queueing. The worker consumes that immutable snapshot and never rereads mutable chats, media, notes, or RAG results.

### Unified generation request

`POST /api/v1/slides/generations` is the single new asynchronous submission route. In v1, `generation_mode` is required and its only accepted value on this route is `standalone_html`. Structured generation remains on the existing per-source endpoints; it does not gain a second public transport in this release. The new route requires the `Idempotency-Key` header and accepts a discriminated source union:

```json
{
  "generation_mode": "standalone_html",
  "source": {
    "kind": "prompt",
    "prompt": "..."
  },
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
  "job_id": 4242,
  "status": "queued",
  "status_url": "/api/v1/slides/generations/4242",
  "presentation_id": null
}
```

`GET /api/v1/slides/generations/{job_id}` takes the existing numeric Jobs ID, is owner-scoped, and returns `job_id`, one of `queued | running | completed | failed | cancelled`, optional bounded progress text, and—only when applicable—`presentation_id`, `content_kind`, `error_code`, and a safe error message. It does not return generated HTML. Internal Jobs state maps as follows:

- `queued` → `queued`;
- `processing` → `running`;
- `completed` → `completed`;
- `failed` → `failed`;
- `cancelled` → `cancelled`;
- `quarantined` → `failed` with `generation_quarantined`.

The job result contains bounded metadata such as `presentation_id`, `content_kind`, document byte count, slide count, and validation status. The client fetches the authenticated presentation detail after completion.

### Provider execution

Standalone HTML generation is available only when:

- the feature flag is enabled;
- one default provider/model pair is configured;
- provider aliases normalize to a canonical pair contained in the standalone HTML allowlist.

The public request cannot override the pair. Custom OpenAI-compatible URLs and provider aliases do not bypass canonical allowlist comparison. Capability availability depends on valid configuration, not transient provider health; provider-call outages become safe job failures. For a newly claimed request, the service also resolves the exact bounded system-prompt text, computes its SHA-256 digest, and snapshots both with the canonical provider/model. An idempotent replay uses that stored pair and prompt rather than substituting the current default or a mutable prompt override. The worker still revalidates the stored pair against the current allowlist before use.

Provider calls use the existing abstraction, run without blocking the FastAPI event loop, and have a server-enforced timeout and maximum output-token budget. Source material is sent to that configured provider as untrusted user content. Delimiters and model allowlisting are not prompt-injection defenses: the generation call receives no tools, application credentials, cookies, internal service URLs, or authority to fetch additional content.

### Crash-safe idempotency and worker correlation

Slides and Jobs use separate persistence stores, so the design does not rely on a cross-database transaction or place source material in a Jobs JSON payload.

Each user's Slides database contains a durable `slides_generation_receipts` ledger and an ephemeral one-to-one `slides_generation_inputs` row. The receipt stores an internal receipt UUID, canonical owner ID, hashed client key, a client-request digest that excludes server configuration, a separate execution digest, numeric Jobs ID, immutable Jobs UUID, optional presentation ID, receipt/terminal state, bounded safe error fields, timestamps, and terminal expiry. The input row stores the resolved source kind/text/digest/byte count, bounded provenance JSON, normalized HTML options, exact bounded system-prompt text/digest/version, and resolved canonical provider/model. It contains no provider secret or raw idempotency key.

At submission, the server first validates and canonicalizes only client-supplied fields and computes `client_request_sha256`. Before resolving current capabilities, the default provider/model, or mutable source content, it checks the owner-scoped receipt by hashed key:

- same client-request digest: return or recover the bound job/result from the receipt and, while nonterminal, its stored execution input without resolving the source or current default again;
- different client-request digest: return `409 generation_idempotency_conflict`.

For a new key only, the endpoint checks generation capability, resolves the default allowlisted pair and exact system prompt, authorizes and snapshots the source through the owner-scoped adapter, applies the source ceiling and a fixed 128 KiB UTF-8 system-prompt ceiling, and builds immutable provenance. It computes `execution_sha256` from a canonical manifest containing the client-request digest, source kind/digest/byte count, normalized options, canonical provider/model, and exact prompt digest/version. A `BEGIN IMMEDIATE` Slides transaction rechecks the key and atomically inserts the receipt plus input snapshot. A racing loser discards its temporary resolution and returns the winner when the client-request digest matches.

Because the Jobs unique index is not owner-scoped, the server derives the internal Jobs key as `slides:v1:` plus the hexadecimal SHA-256 of the canonical owner ID, a NUL separator, and the client key. The raw client key is not used as the global Jobs key. The fixed scope is `domain=slides`, `queue=default`, and `job_type=presentation.generate`. The Jobs payload contains only the canonical owner ID, internal receipt UUID, `client_request_sha256`, and `execution_sha256`; it contains no source content or secret.

This job type uses a new Jobs creation option with an **exact payload** policy. After secret hygiene and canonical JSON serialization, any redaction or structural mutation is rejected, and an oversized envelope is rejected before insertion even when `JOBS_JSON_TRUNCATE=true`; the generic `{"_truncated": ...}` replacement is forbidden for this type. Size failure maps to bounded `413`, while invalid/nonexact construction maps to `422`. The normalized decrypted payload returned from create/idempotent lookup must deep-equal the expected canonical envelope before receipt binding. Thus a queued presentation job can never silently lose its receipt UUID or digests.

Recovery cannot depend on `create_job`'s admission order. Before calling `create_job`, the Slides service uses a Jobs helper that looks up an active or archived row by the complete `(domain, queue, job_type, internal_idempotency_key)` scope without rerunning fair-share, quota, or capacity admission. A found row is accepted only when its owner, scope, type, key, and normalized decrypted payload all match; otherwise recovery fails safely. Only a genuine miss enters normal Jobs admission and exact-payload creation.

After finding or creating the Jobs row, the server binds both `job.id` and `job.uuid` to the receipt. Public status and archive lookup use the numeric ID; UUID remains the immutable worker-correlation key. If the process crashes after receipt/input creation or Jobs insertion but before binding, retry finds the same Jobs row before admission and finishes the binding even when current quota would reject a new job.

A deterministic enqueue rejection that proves no Jobs row was created removes the still-unbound receipt/input transactionally before returning `413` or `422`. An ambiguous enqueue exception never deletes them; the response tells the client to retry the same idempotency key so the derived Jobs key can recover a possibly created row.

Every generated presentation stores the immutable originating Jobs UUID as `generation_job_uuid`, protected by a partial unique index within the owner's Slides database. Before resolving any per-user database path, the worker treats the Jobs row's owner as canonical and verifies `job.owner_user_id == payload.owner_user_id`. After opening that owner's database it also verifies the receipt owner, receipt UUID, bound Jobs UUID/ID when present, client-request digest, and execution digest against the Jobs row and payload. Any mismatch is a nonretryable `generation_correlation_mismatch` failure with no provider call or cross-owner lookup. The worker algorithm is:

1. Load the verified owner-scoped receipt and input snapshot; rederive both digests and return an already committed presentation if present.
2. Revalidate the stored canonical provider/model pair, call the model with the stored source and exact stored system prompt, and validate output.
3. In one Slides write transaction, recheck the receipt, insert the presentation with its initial entity snapshot, derived FTS text, stored HTML byte/slide counts, bounded immutable `generation_provenance_json` copied from the input, and `generation_job_uuid`, update the receipt to `completed` with its presentation ID and terminal expiry, and delete the input row. If another retry already committed it, return that row instead.
4. Complete the Jobs result with bounded presentation metadata.

Before marking a provider, validation, cancellation, or quarantine outcome terminal in Jobs, the domain path records the corresponding public terminal state and safe error in the receipt and deletes the input row. Status lookup reconciles a Jobs terminal state into a nonterminal receipt if a crash interrupted that sequence. Worker retry, cancellation, and quarantine hooks must use the same helper.

The 30-day idempotency promise is one logical Jobs row and at most one committed presentation for the same owner/key/client request. A terminal receipt replays its stored safe result even after Jobs archival. Provider invocation is necessarily at-least-once under the existing WorkerSDK retry model: a process crash after the provider returns but before the Slides commit may repeat the provider call and its cost. Transport replay does not create another Jobs row, and any retry after the presentation commit returns the committed row without calling the provider. This design does not claim exactly-once external side effects.

Active and archived Jobs retention for `presentation.generate` remains at least 30 days so interrupted reconciliation can finish. Receipt expiry is assigned only on terminal transition. Cleanup never removes a receipt or input bound to a nonterminal Jobs row, regardless of age; an unresolved row must first be terminally failed or cancelled through reconciliation/operator action. If both the Jobs row and a terminal receipt result are missing inside the guarantee window, status returns `503 generation_receipt_unresolved` and never enqueues implicitly. Thirty days after terminal state, cleanup may remove receipt metadata and any remaining input; a deliberate retry then uses a new key.

## Standalone HTML Validation

One pure Python backend validator is authoritative after generation, before persistence, on every HTML save or restore, and before every saved standalone HTML or JSON export. The browser's non-loading safe-outline parser is only a responsiveness and display aid; it is never authoritative for persistence and never executes the source.

The validator uses namespace-aware HTML parsing and the repository's existing token-aware Bleach/tinycss2 CSS path to inspect declarations, at-rules, escaped identifiers, and URL tokens. It does not make security decisions with regular expressions, and it fails closed if the required parser support is unavailable.

Save and restore return bounded diagnostic codes and locations when validation fails; error responses never echo source. The editor may download a malformed current draft for recovery, but cannot persist it. V1 adds no standalone draft-validation endpoint because no execution flow needs a separate validation handshake.

Default hard limits are:

- 1 MiB UTF-8 document size;
- 1 through 30 `.slide` sections;
- 10,000 parsed HTML elements;
- 20,000 total attributes;
- 250,000 visible-text characters before indexing truncation;
- the existing source-input ceiling and a server-enforced generation token ceiling.

These HTML document/parser limits are immutable in V1 so every saved document remains editable and downloadable. Changing them later requires a capability-schema/version change and an explicit compatibility policy. The existing source-input ceiling and generation token budget remain independently configurable.

Validation rejects:

- invalid UTF-8, NUL bytes, incomplete document structure, or likely truncation;
- missing doctype, head, body, title, or slide sections;
- excess size, slide count, nodes, or attributes;
- anything other than exactly one attribute-free classic inline script as the final body child, including modules and source-map directives;
- event-handler and `style` attributes; all executable code and CSS must live in validated `<script>` and `<style>` elements;
- every relative, fragment, protocol-relative, HTTP, HTTPS, FTP, file, websocket, `data:`, `blob:`, or other URL-bearing attribute or CSS value;
- `<base>`, `<iframe>`, `<frame>`, `<object>`, `<embed>`, `<form>`, refresh metadata, and declarative popup/navigation elements;
- external script/style/font/media/image references;
- CSS `@import` and resource-bearing `url()` values;
- namespace-aware SVG/MathML links and resource attributes;
- workers, service-worker registration, and manifest resources when statically identifiable.

The single inline script remains allowed. Static inspection of its content supplies diagnostics for obvious network/navigation APIs but is not treated as a security decision.

Title normalization decodes text content, applies Unicode NFC, collapses HTML whitespace, then rejects remaining C0/C1 controls and bidi formatting/override/isolate characters. The result must be nonblank, at most 200 Unicode scalar values, and at most 512 UTF-8 bytes. The same rule applies after generation and on every save or restore.

Validation derives:

- a normalized nonblank project title from `<title>`;
- slide count;
- UTF-8 byte count;
- SHA-256 source digest;
- bounded visible search text excluding script, style, template, noscript, speaker notes, and hidden chrome;
- safe diagnostic codes without echoing source or secrets into logs.

Invalid generated output completes the job as failed with a stable validation code and bounded field diagnostics. The form retains its inputs and offers Retry. V1 does not silently fall back to Markdown or structured slides.

## Persistence And Invariants

### Presentation record

The presentation record gains:

- `content_kind TEXT NOT NULL DEFAULT 'structured_slides'`
- `html_document TEXT NULL`
- `html_sha256 TEXT NULL`
- `html_bytes INTEGER NULL`
- `html_slide_count INTEGER NULL`
- `generation_job_uuid TEXT NULL`
- `generation_provenance_json TEXT NULL`, immutable canonical JSON capped at 4,096 UTF-8 bytes

The existing `slides` column remains non-null.

The owner-scoped `slides_generation_receipts` table contains:

- `id TEXT PRIMARY KEY`, an internal UUID
- `owner_user_id TEXT NOT NULL`, the canonical Jobs/Slides owner identifier
- `idempotency_key_sha256 TEXT UNIQUE NOT NULL`
- `client_request_sha256 TEXT NOT NULL`
- `execution_sha256 TEXT NOT NULL`
- `job_id INTEGER NULL`
- `job_uuid TEXT NULL`
- `presentation_id TEXT NULL`, referencing `presentations.id`
- `receipt_status TEXT NOT NULL`, constrained to `claimed | queued | running | completed | failed | cancelled`
- bounded nullable `error_code` and `error_message`
- `created_at`, `updated_at`, and nullable `expires_at` assigned only at terminal transition

The owner-scoped `slides_generation_inputs` table contains one ephemeral row per nonterminal receipt:

- `receipt_id TEXT PRIMARY KEY`, referencing the receipt;
- `source_kind`, `source_text`, `source_sha256`, and `source_bytes`;
- bounded `provenance_json` and normalized `html_options_json`;
- canonical `provider` and `model` identifiers;
- bounded exact `system_prompt`, `prompt_sha256`, and `prompt_contract_version`;
- `created_at`.

The receipt stores no raw client key, source material, prompt, model output, or HTML. The input row stores no credentials; its system prompt is administrator-resolved application configuration, not source material, and is deleted with the other execution input. Successful commit copies the exact bounded normalized provenance into `presentations.generation_provenance_json` before deleting the input; failed/cancelled terminal reconciliation deletes the input. The receipt and presentation binding remain for the idempotency window.

Generation provenance is a server-built closed schema with no extra fields:

```json
{
  "schema_version": 1,
  "source_kind": "prompt",
  "source_ref": null,
  "source_snapshot_sha256": "<hex>",
  "source_bytes": 1234,
  "provider": "canonical-provider",
  "model": "allowed-default-model",
  "prompt_sha256": "<hex>"
}
```

`source_ref` is null for direct material, an owner-scoped identifier capped at 256 UTF-8 bytes for a single chat/media source, or a fixed hexadecimal canonical hash for multi-note and RAG selections; it never contains prompt, note, chat, media text, or a RAG query. The summary projection exposes only `{source_kind, provider, model}`. The complete object is immutable, copied into version snapshots, and validated against the 4,096-byte cap before the source input can be deleted.

The canonical invariant is:

- `structured_slides`: validated slide list; `html_document`, `html_sha256`, `html_bytes`, `html_slide_count`, `generation_job_uuid`, and `generation_provenance_json` are null.
- `standalone_html`: nonblank validated `html_document`; matching digest, byte count, and slide count; stored `slides` is `[]`; `generation_job_uuid` is nonnull; and bounded valid generation provenance is present and immutable in v1.

`slides_text` remains the canonical derived FTS source for both kinds. Clients can never supply it.

All paths that create or mutate persisted records, restore versions, or export saved content—including REST, MCP, and workers—enforce the invariant through one domain service. The bounded `draft-attachment` recovery echo is not a persistence or saved-export path and intentionally applies only its owner, kind, size, UTF-8, and NUL checks. Only the verified generation worker may create a `standalone_html` record in v1; generic presentation-create and MCP-create requests for that kind return `409 standalone_html_creation_requires_generation`. Partial updates first merge with the current record, then validate the complete candidate inside the optimistic-concurrency operation. Omitting `content_kind` preserves the current kind; it never converts an HTML project into an empty structured project. A partial unique index on nonnull `generation_job_uuid` provides worker retry deduplication. Standalone HTML provenance and generation correlation are immutable; HTML duplication is not exposed in v1.

HTML Save accepts the complete `html_document`; the server derives the record title, digest, byte count, `html_slide_count`, and `slides_text` from that candidate in the same write transaction. A client-supplied standalone title, provenance, FTS text, digest, byte/slide count, or generation correlation field is rejected rather than ignored.

Content kind cannot change in v1. A request that attempts to change it returns `409 content_kind_immutable`.

### Migration

Slides schema versioning becomes authoritative at schema version 2. New databases are created directly at v2. Existing schema 0/1 databases enter `BEGIN IMMEDIATE`, re-read the version and actual columns after acquiring the lock, apply individual migration statements without `executescript`, add/backfill the presentation fields, create both generation tables and indexes, normalize `schema_version` to exactly one row containing `2`, and commit atomically. Any statement failure rolls back the whole migration. Legacy rows become `structured_slides` with null HTML fields.

The v2 runner replaces the ad-hoc ensure-column path for fields introduced by this feature; unrelated legacy compatibility helpers may remain until separately migrated. Tests cover a legacy database, an empty version table, inconsistent multiple version rows, an already migrated database, rollback after an injected failure, and concurrent first access from separate connections/processes. New row mapping and summary queries use explicit projections rather than depending on `SELECT *` positional/dataclass compatibility. Deployment documentation requires a database backup before upgrade and treats this migration as forward-only for old binaries.

### Summaries and search

List and search queries use lightweight projections that exclude `html_document` and full version payloads. Summaries include `content_kind`, title, the bounded `{source_kind, provider, model}` provenance summary, timestamps, version, and either structured slide count or stored HTML slide/byte metadata.

HTML search indexes only the bounded visible text derived by the server in the authoritative create/save/restore transaction. Raw markup, JavaScript, CSS, and speaker notes are never indexed or emitted as result snippets.

### Version snapshots

`presentation.version` is the entity revision used by ETags. For standalone HTML it advances when a save or restore changes the canonical HTML source and its derived title; existing delete semantics remain unchanged. Generation provenance remains immutable, and v1 exposes no other HTML metadata editor. A snapshot is a complete entity snapshot for that revision; v1 does not introduce a separate content-blob version model.

New snapshots include:

- `snapshot_schema_version`
- `content_kind`
- the active content payload only
- source digest and byte/slide metadata
- derived title, immutable provenance, and existing record metadata

Snapshots without a content kind are interpreted as `structured_slides`. Restore validates the snapshot against the current kind and content policy before one atomic update. It restores only canonical mutable fields and can never overwrite IDs, `content_kind`, creation time, owner/client identity, or `generation_job_uuid`. A mismatched/corrupt kind is rejected rather than converted.

The SHA-256 digest is an optimization for comparing HTML source, not the entity-version definition. A standalone save whose complete canonical source and derived title already match is a no-op and creates neither a new entity revision nor a snapshot.

To bound full-document amplification, standalone HTML retains the newest 25 entity snapshots per presentation by default; retention is configurable downward. Retention cleanup occurs in the same successful mutation transaction, never removes the current entity, and a request for a pruned version returns `404 presentation_version_not_found`. Full-document duplication inside those bounded snapshots is accepted for v1 instead of introducing delta storage.

## API Contracts

### Capabilities

`GET /api/v1/slides/capabilities` separates persistence/editor support from generation availability. A temporary provider failure must never make already-saved HTML projects inaccessible. V1 returns this exact shape:

```json
{
  "schema_version": 1,
  "content_kinds": {
    "structured_slides": {
      "read": true,
      "edit": true
    },
    "standalone_html": {
      "read": true,
      "edit": true,
      "export_attachment": true,
      "limits": {
        "max_document_bytes": 1048576,
        "max_slides": 30
      }
    }
  },
  "generation_modes": {
    "structured_slides": {
      "enabled": true,
      "transport": "existing_source_endpoints"
    },
    "standalone_html": {
      "enabled": true,
      "reason": null,
      "transport": "slides_generation_job",
      "source_kinds": ["prompt", "chat", "media", "notes", "rag"],
      "provider": "canonical-provider",
      "model": "allowed-default-model"
    }
  }
}
```

Disabled generation returns `enabled: false`, null provider/model identifiers, and one safe reason: `feature_disabled`, `default_model_not_configured`, `default_model_not_allowed`, or `generation_worker_unavailable`. Read/edit/safe-outline/download support remains available independently. Capability discovery performs no live provider health check. The endpoint uses `Cache-Control: private, no-store`; the creation page fetches it on entry and on explicit Retry.

Servers advertise `standalone_html` read/edit/export support even when generation is disabled. Safe outline is fixed V1 frontend behavior, not a server-configurable capability. The creation form checks `generation_modes.standalone_html.enabled`; list, detail, editor, versions, and attachment download check the content-kind capability. Generation failure disables only generation and blocks neither structured creation nor existing HTML access. The frontend does not infer support from unrelated Slides routes.

### Presentation response

Detail responses form a discriminated union:

- structured detail contains `content_kind: "structured_slides"` and `slides`;
- HTML detail contains `content_kind: "standalone_html"`, `html_document`, `html_sha256`, `html_bytes`, and stored `html_slide_count`.

HTML source is present only in authenticated detail, save/restore, version-content, and explicit export responses, not summaries, search snippets, job summaries, error bodies, or logs. Every non-attachment response containing `html_document` uses `Content-Type: application/json`, `X-Content-Type-Options: nosniff`, `Cache-Control: private, no-store`, and varies on the authentication mechanism in use (`Authorization`, `X-API-KEY`, and/or `Cookie`). No source-bearing route returns or content-negotiates `text/html`. Request/response body logging and error-report payload capture are disabled for every raw-source route. Unknown content kinds render as an unsupported read-only state in clients.

### Update and concurrency

HTML Save uses a kind-aware PATCH with a strong ETag/`If-Match` contract. Standalone responses return strong tags such as `"v7"` and temporarily accept either strong or legacy weak tags when parsing `If-Match`. Existing structured endpoints continue returning and accepting their current weak `W/"v7"` contract; this feature does not change it. A stale tag returns `412` with bounded current-version metadata, not the remote HTML body unless the client explicitly fetches it.

HTML Save accepts only the complete `html_document`; the server derives its title and other source metadata. Structured-only, independently mutable title, provenance, and correlation fields explicitly supplied for an HTML record are rejected rather than ignored. A successful save returns the new detail and ETag. A complete canonical match is a no-op.

### Operation matrix

| Operation | Structured slides | Standalone HTML |
| --- | --- | --- |
| List/get/search/delete | Supported | Supported |
| Versions and restore | Supported | Supported |
| Independent title/source/provenance update | Existing behavior | Rejected; title is derived and provenance is immutable |
| Reorder or per-slide mutation | Supported | Rejected |
| JSON export | Supported | Supported, with HTML source only in authenticated explicit export |
| Reveal.js/Markdown/Marp | Supported | Rejected |
| PDF/video/thumbnail/render job | Supported | Rejected at API and worker |
| HTML attachment | Existing export behavior | Exact standalone source attachment |
| WebUI preview | Existing structured preview | Trusted safe outline only; no HTML/JavaScript execution |
| Extension execution | Not applicable | Rejected/handoff only |

Structured-only rejections use `409 operation_not_supported_for_content_kind` and include the operation and actual kind.

### HTML attachment and draft recovery

Saved HTML export uses the presentation export endpoint with `format=html`. **Download current draft** calls `POST /api/v1/slides/presentations/{presentation_id}/draft-attachment` with `Content-Type: application/octet-stream` and the editor's exact UTF-8 bytes as the request body. It requires an existing owner-scoped `standalone_html` presentation and returns `200` with the same exact bytes; the operation does not persist or require a valid deck. A different content kind returns `409 operation_not_supported_for_content_kind`, a body above 1 MiB returns `413 standalone_html_storage_limit`, and invalid UTF-8 or NUL returns `422 standalone_html_invalid_document`. Because the editor rejects over-cap edits, malformed accepted drafts remain downloadable.

Both responses use the fixed ASCII filename `presentation.html`, independent of the model-derived title, and emit the RFC 6266-safe header `Content-Disposition: attachment; filename="presentation.html"`. No user/model text, path separator, control character, quote, reserved platform name, alternate extension, or CR/LF can enter the filename. They also use:

- `Content-Type: application/octet-stream`
- `Content-Disposition: attachment; filename="presentation.html"`
- `X-Content-Type-Options: nosniff`
- `X-Download-Options: noopen`
- `Cache-Control: private, no-store`
- `Referrer-Policy: no-referrer`
- `Cross-Origin-Resource-Policy: same-origin`

Saved-version downloads include ETag and Last-Modified. No route serves stored HTML inline as `text/html`. Before download, the UI states that the file contains executable code and that opening it locally occurs outside tldw's security boundary.

## Safe Outline Security Contract

### Safe outline pipeline

The automatic preview:

1. checks the UTF-8 cap and runs a conservative one-pass, linear-time source scanner that stops after 10,000 potential start tags or 20,000 potential attributes; malformed quoting fails the preflight rather than consuming unbounded work;
2. declines to invoke the outline parser and shows **Outline unavailable — document too complex** when preflight exceeds either budget;
3. otherwise parses the current source into a non-DOM AST with the frontend's existing lazily loaded `cheerio/slim` dependency, which performs no browser DOM construction, custom-element upgrade, or subresource load;
4. locates bounded `.slide` sections in that AST and extracts semantic text from an explicit HTML-only allowlist;
5. discards every original attribute and all CSS, scripts, SVG, MathML, templates, noscript content, links, images, fonts, forms, embeds, and active objects;
6. rebuilds slide cards with application-owned components and `textContent` only;
7. renders those trusted components directly in the WebUI using application classes and text nodes.

The scanner is a responsiveness guard, not an HTML validator or sanitizer. It does not use a catastrophic-backtracking regular expression, and false-positive refusal affects only the outline; backend validation remains authoritative.

The safe outline creates no iframe and uses no `srcdoc`, `DOMParser`, `innerHTML`, `insertAdjacentHTML`, Blob/data URL, popup, worker, or generated-HTML route. It preserves no model DOM node and never calls a backend validation endpoint merely to repaint. No regular-expression CSS sanitizer is introduced because no model CSS reaches this view. It never claims visual fidelity; the backend remains authoritative for generation, save, restore, and export.

### No V1 execution surface

Validation and safe-outline extraction may parse `html_document` into non-executing ASTs, while the code editor handles it only through value/text APIs. No WebUI component, server renderer, MCP path, extension surface, worker, iframe, new window, or attachment endpoint inserts or renders it as browser markup or executes its script. The application CSP remains unchanged: this feature adds no `unsafe-inline`, `unsafe-eval`, runtime origin, or `frame-src` exception.

The only user-facing standalone-file handoff is an authenticated attachment using `application/octet-stream`, `nosniff`, a fixed filename, and `private, no-store`. Authenticated detail, save/restore, version-content, and explicit JSON export carry source only as inert JSON, text, or octet-stream data under `private, no-store`, never as `text/html`. The WebUI provides no inline view or **Open in new tab** action. Its warning states that opening the downloaded file may execute code outside tldw's security boundary.

Adding execution later requires a new design and security review; implementation must not expose a dormant runtime flag or reuse the safe outline as an execution substrate.

## MCP And Secondary Consumers

The MCP Slides module currently duplicates REST generation, persistence, restore, and export behavior. This feature must not add another independent implementation.

The shared domain service becomes authoritative for content invariants, summary mapping, update/restore, operation guards, and export dispatch. V1 behavior is mandatory:

- MCP list and search results always include `content_kind`. HTML summaries include `html_bytes`, stored `html_slide_count`, and `html_available: true`; they never include source.
- MCP get preserves the current structured response for structured records. For HTML it returns the same bounded metadata plus provenance/version fields and omits `html_document`.
- MCP structured generation remains supported.
- MCP HTML generation, source retrieval, source mutation, restore, and attachment transfer return a normal structured tool result rather than raising a transport exception: `{ "success": false, "error": { "code": "operation_not_supported_for_content_kind", "operation": "...", "content_kind": "standalone_html" } }`.
- Any remaining MCP mutation or export that assumes structured slides performs the same guard before touching persistence.

A later artifact/resource-handle design may expose HTML through MCP without embedding large executable documents in tool results.

The shared WebUI/extension API types also always preserve `content_kind`. In extension runtime contexts:

- list/search include HTML records and their bounded metadata rather than filtering or coercing them to empty structured decks;
- selecting an HTML record shows its title, kind, provenance summary, and an **Open in WebUI** handoff;
- the HTML detail/editor route is not registered in extension builds, and the extension never requests, renders, or executes `html_document`;
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
- `standalone_html_creation_requires_generation`
- `operation_not_supported_for_content_kind`
- `presentation_version_conflict`
- `generation_idempotency_key_required`
- `generation_idempotency_conflict`
- `generation_receipt_unresolved`
- `generation_correlation_mismatch`
- `generation_quarantined`
- `presentation_version_not_found`

Use `400` for a missing/invalid idempotency header, `413` for hard byte/token ceilings and oversized Jobs envelopes, `422` for malformed documents, invalid options, or rejected Jobs payload construction, `409` for kind/operation/idempotency conflicts, `412` for stale ETags, `503` for an unresolved receipt whose Jobs row disappeared inside the guarantee window, and the existing Jobs failure representation for provider or worker failures.

Diagnostics identify bounded fields, limits, and machine-readable codes. They do not echo source documents, prompt bodies, API keys, model output, notes, or JavaScript into logs or error responses.

## Compatibility

- Existing request omission follows the current structured path exactly.
- Existing per-source generation endpoints keep their response types and behavior.
- Legacy database rows and version snapshots default to `structured_slides`.
- List clients receive an additive `content_kind` field; HTML source stays out of summary payloads.
- Frontend normalization preserves both discriminated variants and never converts unknown/missing HTML into `slides: []` for mutation.
- Old servers are detected through explicit capabilities; the HTML form never assumes support from a generic Slides route.
- JSON export preserves the discriminator and active payload and uses attachment/no-store handling when it contains HTML. Presentation import remains out of scope for v1.
- Existing structured weak ETags remain unchanged; strong ETags are scoped to standalone HTML responses.
- Structured Reveal, PDF, and render workers retain their existing behavior, with an added kind check.
- The browser extension can list or hand off HTML projects but never requests, renders, or executes their source.

## Observability And Operations

Record only safe metadata:

- generation mode and source kind;
- provider/model identifier after secret removal;
- job duration and terminal state;
- document byte and slide counts;
- validation/error code.

Never log source material, source snapshots, full HTML, JavaScript, notes, prompts, API keys, download bodies, or raw-source request/response bodies. Existing auth, per-user isolation, rate limiting, and Jobs ownership checks apply.

Standalone HTML **generation** is disabled by default until configured. Startup validates the canonical default provider/model pair against the standalone allowlist and loads the packaged prompt asset. Invalid generation configuration keeps structured generation and standalone HTML read/edit/export available, advertises the HTML generation mode as disabled with a safe reason code, and logs a safe administrator-facing reason.

`presentation.generate` is registered explicitly in the Slides Jobs worker and in any job-type allowlist. In-process startup/shutdown follows the existing Jobs worker lifecycle; external-worker deployments receive an equivalent documented registration command/configuration. Startup fails the generation capability closed if the handler is not registered. A lightweight domain reconciler checks nonterminal receipts against numeric active/archive Jobs state so admin cancellation or quarantine terminally updates the receipt and only then purges the input snapshot even when no client polls. Cleanup purges terminally expired receipts and terminal orphan inputs without deleting committed presentations; it never purges a receipt/input still bound to a nonterminal Jobs row.

## Testing Strategy

### Backend unit and property tests

- request/content-kind enum mapping and cross-field invariants;
- partial-update merge-before-validation behavior;
- immutable kind and structured-only operation guards;
- outer-fence removal and deterministic prompt assembly;
- every presentation type, visual direction, delivery style, and boundary slide count;
- HTML parser limits, malformed/truncated documents, exact one-classic-script placement, all URL/CSS/data-URI/style/event-attribute rejection, namespace-aware SVG attributes, bounded Unicode title normalization, search extraction, and digest calculation;
- generated source never appears in logs or error messages;
- property/fuzz cases for parser limits and invariant preservation;
- exact no-op saves, metadata-only revisions with unchanged HTML, and version retention.

### Database and migration tests

- direct v2 creation plus schema 0/1 migration, default structured rows, and stored HTML slide-count/provenance columns;
- empty/multiple version rows, injected rollback, repeated migration, and concurrent migration attempts;
- cross-field constraints/domain validation reject structured rows with HTML/correlation fields and standalone rows missing valid source metadata, provenance, or generation UUID;
- create/get/update/search/delete for both kinds;
- lightweight summaries do not load full HTML and return the stored HTML slide count plus bounded provenance projection;
- old and new snapshot restore behavior;
- restore preserves immutable identity/kind/generation fields and pruned versions return 404;
- FTS excludes scripts/styles/speaker notes and is committed with the authoritative entity write;
- standalone strong ETags plus unchanged structured weak ETags;
- receipt/input atomic claim, numeric/UUID bind, exact prompt/source snapshots, durable presentation provenance before source-snapshot cleanup, terminal replay, nonterminal retention, and terminal cleanup.

### API, Jobs, and MCP integration tests

- successful HTML generation from prompt, chat, media, notes, and RAG with mocked LLMs;
- default-model canonicalization/allowlist, exact capability schema, ownership, size, timeout, and idempotency behavior;
- submission-time immutable source snapshots for all source kinds and proof that workers do not reread changed sources;
- minimal Jobs payload uses `queue=default`, survives secret scanning byte-for-byte after normalization, and maps oversized/redacted/nonexact envelopes to 413/422 without inserting a job, including with `JOBS_JSON_TRUNCATE=true` and a deliberately tiny limit;
- client-request digests remain independent of provider/default changes, execution digests bind the stored source/pair/exact prompt digest, mutable prompt overrides cannot alter a queued job, and same-key replay uses the stored pair/prompt;
- same-key/same-client-digest replay, same-key/different-client-digest conflict, cross-owner isolation, pre-admission active/archive idempotency lookup, quota-independent job-bind crash recovery, receipt/Jobs owner-correlation mismatch failure, crash recovery after the presentation commit but before Jobs completion, numeric active/archive lookup, terminal-state reconciliation, premature Jobs-row loss failure, and 30-day one-job/one-presentation semantics;
- provider calls may repeat only across the documented pre-commit worker-crash window, while retries after commit never call the provider;
- queued/processing/completed/failed/cancelled/quarantined public status mapping;
- failed validation creates no presentation;
- invalid standalone saves and restores are atomic and create no revision; corrupt/invalid stored standalone content is rejected before both `format=html` and JSON export, while an in-bounds malformed draft remains downloadable through `draft-attachment`;
- generic REST/MCP create cannot construct a standalone HTML record, and the generation provenance closed schema/4,096-byte cap is enforced;
- job result contains metadata rather than HTML;
- every non-attachment raw-source response is `application/json` plus `nosniff`, is `private, no-store`, varies for auth, never negotiates `text/html`, and is excluded from body logging/error capture;
- saved HTML remains readable/editable/exportable while HTML generation is disabled;
- all structured-only operations reject HTML before dispatch;
- render workers independently recheck content kind;
- saved export and exact draft-attachment route/body/error/header behavior;
- MCP cannot mutate/export HTML through structured assumptions;
- omission of a new mode field on the existing per-source endpoints preserves current structured behavior; the new `/slides/generations` route always requires `standalone_html`.

### Frontend tests

- capability loading, Retry, unsupported-server, generation-disabled, and supported states;
- minimal paginated index loading/empty/error/retry states, Load more, New/open actions, and HTML kind metadata;
- form validation, server-default model display, payload mapping, duplicate-submit prevention, trusted-principal-scoped form-draft recovery, reload-before-`202` same-key replay, session Resume/Discard, storage/account-switch behavior, polling failures, and job completion handoff;
- discriminated client normalization and unknown-kind fallback;
- code/outline responsive layout and absence of structured controls;
- fixed hard-cap edit rejection, trusted-principal-scoped/expiring recovery records with Restore/Download/Discard, exclusion of HTML from global caches/devtools/error capture, session quota/account-switch behavior, dirty/save/error/conflict behavior, and unknown-save reconciliation;
- safe-outline extraction is latest-wins under rapid edits, enforces the linear pre-parser budgets, discards model markup/CSS/scripts/assets, and current-draft download sends exact bytes;
- keyboard access, labels, focus behavior, `aria-live`, reduced motion, 44px touch targets, and focus-preserving Code/Outline tabs at narrow viewports.

### Safe-outline security and browser tests

- sentinel JavaScript never runs during generation handoff, open, edit, outline refresh, save, restore, reload, Back/Forward, or download in Chromium, Firefox, and WebKit;
- no model DOM node, attribute, CSS, script, SVG, MathML, template, link, or asset survives into the component-rendered outline;
- the feature creates no iframe, `srcdoc`, Blob/data URL, popup, worker, generated-HTML route, `dangerouslySetInnerHTML`, `DOMParser`, `innerHTML`, or `insertAdjacentHTML` path from source;
- parsing valid, invalid, and adversarial drafts emits no subresource or navigation request and never calls a backend validation endpoint;
- direct API requests and browser navigation to every source-bearing route receive only the contracted JSON/octet-stream MIME plus `nosniff`; none renders HTML or executes a sentinel;
- the ordinary application CSP remains unchanged with no new `unsafe-inline`, `unsafe-eval`, or `frame-src` allowance;
- malformed/over-budget outline input shows bounded trusted UI, leaves source unchanged, and cannot replace a newer outline result;
- malicious titles cannot inject response headers and every attachment uses the fixed `presentation.html` filename;
- a keyboard-focused desktop/mobile E2E covers generate, edit, safe outline, save, conflict recovery, reopen, and download without horizontal page overflow.

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
- deployment guidance for capability enablement, model allowlisting, migration backup, safe-outline limitations, no-execution guarantees, and attachment handling

The PRD exception must be narrow: explicit `standalone_html` projects may generate, store, edit, version, and download JavaScript as opaque text. Arbitrary JavaScript execution remains prohibited in every tldw surface, extension context, worker, and server renderer.

## Delivery Decomposition

This is one product design but should be delivered as reviewable child tasks:

1. Shared contracts, schema migration, persistence invariants, version/search behavior, and operation guards.
2. Prompt asset, validator, shared generation service, submission-time source snapshots, Jobs integration, source variants, and mocked backend tests.
3. Capabilities, HTML attachment endpoints, MCP guards, and no-inline-render protections.
4. Presentation Studio index/form, discriminated client/store model, and resumable generation-job UX.
5. HTML editor, safe outline, save/conflict/recovery, responsive/accessibility behavior, and no-execution browser tests.
6. Documentation, PRD reconciliation, end-to-end verification, Bandit, and rollout evidence.

Each child task must remain independently reviewable and may tighten implementation details without weakening this design's content invariant or security boundary.

## Success Criteria

The feature is complete when an authenticated user can submit direct material through the new form, receive one idempotently generated first-class HTML presentation, inspect its text-only safe outline, edit and save without silent conflicts, reopen it from the paginated index, and download the current source as an attachment. Presentation Studio and every other tldw surface must never render or execute the generated HTML/JavaScript. Backend tests separately verify HTML search indexing and search API results.

The same backend service must successfully generate HTML from mocked chat, media, notes, and RAG sources even though those source selectors are not yet exposed in the form. Existing structured Slides behavior must remain compatible. No server renderer, MCP structured operation, browser-extension surface, or older client may execute or silently reinterpret the HTML payload.
