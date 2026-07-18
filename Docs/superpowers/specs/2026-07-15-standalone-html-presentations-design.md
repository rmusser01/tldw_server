# Standalone HTML+JavaScript Presentations Design

**Date:** 2026-07-15

**Status:** Requester-approved V1 direction; backend, security, and product re-review approved; ready for implementation planning

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
- HTML generation is disabled unless an administrator enables the feature and configures one concrete provider/model/adapter/endpoint target contained in the standalone HTML allowlist.
- Execution-target allowlisting and endpoint binding improve operational isolation but do not make model output trusted.
- V1 documents use inline browser-native HTML, CSS, JavaScript, and optional inline SVG only. All URL-bearing resources, including `data:` assets, are rejected. Semantic theme names influence generated CSS tokens but do not load theme files.
- The automatic view is a trusted safe outline, not a fidelity preview. V1 never executes generated JavaScript in the application, including after generation, editing, save, restore, or reload.
- HTML editing uses explicit Save. It does not reuse the structured editor's whole-project autosave or automatic conflict merge.
- Raw/current-draft HTML is downloadable only as an authenticated attachment. The application never offers an inline or **Open in new tab** view.
- Structured-only operations fail with a stable content-kind error at both API admission and worker execution.
- The HTML document's normalized `<title>` is the project title. V1 has no separate HTML title control that can drift from the source.

## Terminology And Trust Model

### Canonical source

`html_document` is the user's canonical editable source. It is opaque untrusted text even when it was produced by an allowed model and passed generation validation. It may contain executable inline JavaScript.

### Safe outline

A safe outline is a derived, noncanonical view of slide boundaries and bounded semantic text. The application parses without executing, extracts text from headings, paragraphs, lists, tables, figures, and notes, and rebuilds the view with trusted components and text nodes. It preserves no model markup, CSS, style attributes, event handlers, images, fonts, SVG, MathML, templates, links, or active objects. It is intentionally not a visual-fidelity preview.

### Execution boundary

Generated JavaScript is stored and edited as text only. The WebUI creates no iframe, `srcdoc`, data URL, popup, server-render request, generated-code worker, or other execution context from `html_document`. Trusted application-owned Monaco or parser workers may receive the document strictly as inert text; they must never construct worker code, modules, URLs, or executable functions from it. The sole Blob URL exception is the short-lived attachment-download transport defined below: it is used only by the fixed-name download anchor and is never assigned to a browsing/rendering context, opened in a new tab, embedded, previewed, or retained. No control enables in-application execution in V1. Download warnings make clear that opening the attachment leaves tldw's security boundary.

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
   - `self-guided`: include enough visible context for the deck to stand alone and omit speaker notes. This mode does not autoplay or auto-advance.

The form's source and audience text controls are labelled, are not persisted by native browser form restoration, and set `spellcheck="false"`, `autocorrect="off"`, `autocapitalize="off"`, and `autocomplete="off"` (plus supported password-manager opt-out). An edit/paste containing an unpaired UTF-16 surrogate is rejected before component state or recovery persistence, so later UTF-8 encoding cannot silently substitute U+FFFD. The application's scoped, expiring recovery record is the only intentional browser persistence for these values.

V1 has no provider/model picker. The server resolves the administrator-configured concrete allowlisted execution target and returns its nonsecret provider, model, adapter, and endpoint identity in capability metadata so the form can explain exactly what will run and where source will be sent.

Submitting the form:

- validates every field locally and server-side;
- creates one generation job using an idempotency key;
- disables duplicate submission while the request is pending;
- preserves all form values on error, timeout, or Stop waiting;
- shows real job state without invented percentage progress;
- uses “Stop waiting” unless Jobs provides genuine cancellation for the queued/running state;
- opens the created HTML project after the job reports its `presentation_id`.

Immediately when submission starts—before the POST—the client captures and displays one immutable canonical request snapshot and locks every source/option field. The snapshot remains locked while admission is in flight, its outcome is unknown, or the generation is nonterminal. A definitive pre-admission rejection that proves no receipt/job was created unlocks it for correction; a timeout/network ambiguity does not. **Stop waiting** stops polling only and returns to the index; generation continues and Resume remains available. **Forget this job; generation continues** clears only browser resume/form-recovery records and explicitly warns that it neither cancels the job nor deletes a presentation the job may still create. An explicit **Start a different request** action during an unknown outcome requires confirmation that the original may still complete, then creates a separate request and idempotency key. A terminal failure unlocks the preserved fields for **Try again** with a new key.

Before sending the POST, the client stores bounded resume metadata—`{ generationId: string | null, idempotencyKey, requestDigest, timestamp }`—in `sessionStorage`; it fills the opaque UUID `generationId` after receiving `202`. A separate capped form draft preserves the canonical entered fields for replay and terminal-error recovery. Both records have a fixed 24-hour maximum age, including across browser session restore, and are expired/cleared before any value is read. Both keys are namespaced by canonical server origin and a stable, nonsecret authenticated subject/client identifier. Before hydration, the application derives the current principal scope from trusted auth state and compares it with the stored namespace; it never trusts a persisted scope value to choose an account. Records are ignored and cleared on subject change, logout, successful project handoff, explicit forget, or expiry. If session storage fails, the current in-memory form remains usable and the UI warns that reload recovery is unavailable. On reload it offers **Resume** or **Forget this job; generation continues**: Resume polls when `generationId` exists, otherwise it resubmits the preserved canonical request with the same idempotency key. Forgetting clears only local recovery state. Polling uses bounded exponential backoff and handles authentication loss, missing generations, throttling, server errors, cancellation, quarantine, and a completed generation missing its presentation binding without discarding the form state.

The creation/submitted-state page applies the workspace's principal and history boundary independently. It subscribes to trusted auth-principal changes; logout/scope change synchronously clears in-memory source, immutable request display, resume objects, and matching storage. For a still-matching principal, `pagehide` first flushes the latest bounded form or submitted snapshot and then clears source-bearing component memory before bfcache capture. `pageshow`/visibility restoration revalidates origin, principal, and 24-hour age before rehydration and otherwise shows an empty guarded form. The client never relies on the framework hydration cycle alone to protect this state.

The client creates a cryptographically random 16–200 character idempotency key for each deliberate submission and sends it in the `Idempotency-Key` header. The server accepts only 16–200 printable ASCII characters from the closed URL-safe alphabet `[A-Za-z0-9._~-]`, rejects surrounding whitespace/duplicates, and never echoes or logs the key. The client retains it for transport retries, unknown submission outcomes, and resuming a stopped poll. Retrying the same canonical request with the same key returns the existing job or result instead of creating a duplicate presentation. Reusing the key with a different canonical request returns a conflict. After a known terminal generation failure, **Try again** preserves the form values but creates a new key because it is a deliberate new model attempt.

### HTML workspace

An HTML project's workspace is separate from the structured Slide Rail, Slide Editor, Media Rail, and render controls.

On desktop it uses a stable two-column code/outline grid. On narrow screens it uses explicit **Code** and **Outline** tabs. Resizable panes are deferred.

The primary controls are:

- **Save**
- **Download current draft**
- **Back to presentations**

The editor reuses the repository's lazy Monaco dependency and textarea fallback. Both have a programmatically associated visible **HTML source** label. Their actual text-input elements set `spellcheck="false"`, `autocorrect="off"`, `autocapitalize="off"`, and `autocomplete="off"`, are not successful named form controls, and opt out of supported password-manager capture; browser form restoration must not persist source. The Monaco model uses a dedicated inert-text language ID (plain text is acceptable in V1); it does not load/register the HTML language-service contribution, `DocumentLinkProvider`, hover provider, or any provider that turns source into a URL or Markdown. A syntax-only tokenizer may be added only if it emits display tokens and no links/actions. The editor is created with `links: false`, and its scoped opener override rejects every source-originated open request. Source-looking `href`, `src`, CSS URL, and plain-URL text therefore remain nonclickable under hover, Cmd/Ctrl-click, keyboard commands, and the context menu. No global Monaco provider is disposed in a way that changes unrelated editors. Application-owned workers may process source as inert text, but no worker source, module, import, URL, link target, or function is constructed from the document. The editor uses value/text APIs only; source is never inserted through `dangerouslySetInnerHTML`, Markdown rendering, error markup, or list snippets.

Editing immediately marks the project dirty and updates the safe outline after a bounded idle delay. Outline state tracks the buffer digest, rendered digest, and `current | stale | failed` state. Parsing is latest-wins; stale work cannot replace a newer outline. Local size/preflight checks improve responsiveness but are not authoritative for save.

The automatic view is visibly labelled **Safe outline — text only; code never runs in Studio**. No fidelity preview or generated-deck interactive state exists; only application-owned outline cards and the optional safe-text speaker-notes disclosure are interactive.

The editor enforces the same fixed 1 MiB UTF-8 hard ceiling, U+0000 prohibition, and Unicode-scalar invariant as the server. A paste or edit that would cross the limit, introduce NUL, or contain an unpaired UTF-16 surrogate is rejected before `TextEncoder`, component state, or recovery persistence while preserving the previous buffer and announcing the reason. Recovery hydration revalidates the same rules before installing a value. Therefore every accepted draft has one exact UTF-8 encoding and remains byte-for-byte eligible for raw save and authenticated draft download.

Per-project recovery uses capped `sessionStorage`, never `localStorage` or extension storage. A record is `{ schemaVersion, principalScope, presentationId, baseEtag, baseDigest, source, updatedAt }`, where `principalScope` is derived from the canonical server origin and a stable, nonsecret authenticated subject/client identifier. The application computes the current scope before reading `source`, compares it in constant application logic, and never lets persisted data select the active account. Records expire after 24 hours or the browser session, are ignored and cleared on subject change, logout, or confirmed discard, and never autoapply. Full HTML remains component-local plus this scoped recovery record: it is excluded from React Query/SWR or equivalent global/persisted caches, Redux-style action/devtools logging, analytics, and error-report payloads. When a nonmatching draft is found, the workspace offers **Restore recovered draft**, **Download recovered draft**, or **Discard recovered draft**; discard requires confirmation before removing the record. A matching saved digest clears the record. If browser quota or storage access fails, the UI keeps the in-memory buffer, shows a persistent **Recovery unavailable** warning, and retains the navigation/unload warning.

The workspace subscribes directly to trusted auth-principal changes. Logout, account switch, token subject change, and scope mismatch synchronously dispose the Monaco model, in-memory source, outline AST/worker state, response objects, and matching recovery record before another account can use the route. When the principal still matches, `pagehide` first synchronously flushes the latest accepted buffer/base ETag/digest into the bounded recovery record, then disposes source-bearing memory before the page becomes eligible for the back-forward cache; a storage failure follows the already-visible recovery-unavailable path. A later Back/Forward visit must reauthenticate and rehydrate only from a scope-matching record. `pageshow` (including `event.persisted`) and visibility restoration rederive the current scope before any source fetch or recovery read and keep an empty guarded shell on mismatch. This prevents `no-store` API responses from surviving indirectly in bfcache/history memory across principals without losing the last accepted keystroke for the same principal.

Save states are user-facing and announced through `aria-live`: `Saved`, `Saving`, `Not saved`, and `Conflict`. A capped session recovery draft is retained per presentation and cleared only after the matching source revision saves successfully. A navigation/unload warning protects unsaved work.

On an optimistic-concurrency conflict, the Studio does not merge or retry automatically. It preserves the local buffer and offers:

- **Discard my changes and load server version**, requiring confirmation before replacing the local buffer and clearing its recovery record
- **Overwrite server with my draft**, requiring an explicit replace confirmation against the newly fetched version
- **Download my draft**

The overwrite path is an explicit user decision, not a normal autosave behavior. **Overwrite server with my draft** first refetches the current entity and ETag, asks for replacement confirmation, and retries against that ETag; a second race returns to the same conflict state rather than overwriting silently. Confirmed discard is the only recovery-clear path other than a successful matching save or the separately confirmed recovery-record discard action.

If a save response is lost, the Studio refetches the same owner-scoped presentation detail. When its presentation ID and canonical source digest match the local candidate digest, it treats the save as confirmed and adopts the server-derived title and returned ETag; the browser never attempts to reproduce backend title normalization. A same-digest response whose stored title violates the server's own source-derived invariant is a bounded server-integrity error, not a client conflict. Otherwise the Studio keeps the recovery draft and shows `Not saved` or `Conflict` as appropriate.

### Navigation and source handoff

The Presentation Studio index ships a minimal real project list rather than its current informational copy: a prominent **New presentation** action, recent projects, kind badges, open actions, and offset-based **Load more** using the existing `limit`, `offset`, `pagination.has_more`, and `pagination.next_offset` contract. Accumulated records are deduplicated by presentation ID. Explicit loading, empty, error, Retry, and offline states are required, and every project remains reachable through pagination. V1 introduces no cursor protocol. HTML projects do not report an empty structured slide count as their primary summary. User-facing search and richer list management remain separate follow-up work; backend HTML search indexing remains part of this release.

The first UI release accepts pasted/direct material only. The backend source union is nevertheless complete. Later chat, media, notes, and RAG entry points hand off source identifiers or a short-lived server-side handoff key; they must not place full source content in a URL.

The browser extension shows metadata only and an **Open in WebUI** handoff. It does not request HTML detail/version bodies or execute generated JavaScript.

## Prompt Contract

### Storage and resolution

The packaged default is loaded through the existing prompt loader using the logical key `slides.standalone_html_system`. It lives in the Slides prompt asset set and keeps the repository's existing deployment-file/environment override behavior. A missing, unreadable, over-limit, or malformed prompt asset disables standalone generation with `prompt_asset_unavailable`; there is no shorter silent fallback that weakens the maintained contract. Existing structured generation and saved HTML access remain available.

The prompt is application-owned. Source material and form options are placed in clearly delimited user content and never concatenated as additional system instructions.

### Adaptation of the supplied prompt

The default preserves the useful portions of the supplied prompt:

- the twelve presentation-type narrative flows;
- one `<section class="slide">` per page;
- token-driven CSS;
- keyboard navigation;
- responsive layout and reduced-motion handling;
- concise audience-facing copy;
- optional speaker notes embedded in the HTML, hidden from audience view, and available through a labelled in-document notes control in the downloaded deck;
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

The following tracked narrative-flow table is normative and replaces any implementation-time dependency on the ephemeral original attachment:

| Presentation type | Canonical narrative flow |
| --- | --- |
| `pitch-deck` | Cover → Problem → Solution → Traction → Market → Business Model → Team → Ask → Closing |
| `tech-sharing` | Cover → Agenda → Architecture → Deep Dive → Code/Demo → Benchmarks → Challenges → Takeaways → Q&A |
| `product-launch` | Cover → Teaser → Problem → Solution → Demo → Features → Roadmap → Pricing → CTA |
| `weekly-report` | Cover → Summary → Metrics → Wins → Blockers → Action Items → Next Week → Closing |
| `course-module` | Cover → Objectives → Concept 1 → Concept 2 → Example → Exercise → Summary → Resources |
| `keynote` | Cover → Hook → Story 1 → Story 2 → Insight → Vision → Call to Action → Manifesto |
| `data-report` | Cover → Agenda → Key Findings → Data Deep Dive → Trends → Implications → Recommendations → Appendix |
| `training` | Cover → Agenda → Theory → Demo → Hands-on → Common Mistakes → Best Practices → Assessment |
| `social-media` | Cover → Hook → Punchy Point 1 → Punchy Point 2 → Visual Proof → CTA → Share Prompt |
| `case-study` | Cover → Client/Context → Challenge → Approach → Execution → Results → Testimonial → CTA |
| `comparison` | Cover → Criteria → Option A → Option B → Head-to-Head → Verdict → Recommendation |
| `roadmap` | Cover → Vision → Phase 1 → Phase 2 → Phase 3 → Dependencies → Risks → CTA |

These flows are narrative guidance, not fixed slide counts. The generator may merge or omit stages to fit the requested count and supplied evidence. The complete 211-line source prompt is historical input, not a runtime or implementation dependency; this design and the packaged prompt asset are the maintained contract.

### Required generated structure

The model must return exactly one complete document containing:

- `<!doctype html>`
- `<html>`, `<head>`, and `<body>`
- UTF-8 charset and viewport metadata
- a nonblank `<title>`
- inline CSS contained in `<style>` elements, with no `style` attributes
- one to thirty `<section class="slide">` elements
- exactly one attribute-free classic inline `<script>` as the final child of `<body>`, containing self-contained browser-native keyboard navigation for the downloaded file and, for `speaker-led`, a labelled current-slide notes toggle with the `N` shortcut
- accessible document landmarks and usable focus states
- a reduced-motion path that leaves every active element visible
- for `speaker-led`, exactly one `.notes` element inside each slide, hidden from the audience view but readable through the downloaded deck's labelled notes control; for `self-guided`, no `.notes` elements

The model must not return Markdown fences, additional scripts, script attributes or module syntax, event-handler or `style` attributes, URL-bearing attributes, CSS `url()` or `@font-face`, relative/remote/data/blob resources, remote imports, base URLs, frames, forms, popups, service workers, workers, storage-dependent state, analytics, telemetry, source-map directives, or network calls. The notes control uses the same document only—no popup, storage, worker, or network access. It must not invent citations. When bounded source citations are supplied, it preserves them as nonlinked visible text or notes content as appropriate.

The generator tolerates and strips one outer Markdown fence before validation because some providers add one despite instructions. It does not perform a hidden repair call in v1.

## Generation Architecture

### Shared standalone HTML generation service

A standalone HTML Slides generation service is the single orchestration boundary for the new job across all five source adapters. It owns:

- submission-time source-kind dispatch, ownership checks, bounded snapshotting, and provenance;
- concrete provider/model/adapter/endpoint allowlist enforcement;
- prompt resolution and request assembly;
- provider timeout and output-token ceilings;
- deterministic validation and normalization;
- atomic Slides persistence plus crash-reconciled Jobs result metadata.

The new REST route and background worker use this service rather than duplicating normalization or persistence rules. V1 does not refactor the existing structured generator or its per-source REST routes; those routes keep their current synchronous response shapes and behavior when their existing requests omit any new mode field.

Source adapters accept explicit per-user database/service dependencies; they do not depend on a FastAPI `Request` object or endpoint-only dependency overrides. The submission path resolves and snapshots source content before queueing. The worker consumes that immutable snapshot and never rereads mutable chats, media, notes, or RAG results. Each adapter must query or iterate incrementally and stop after accumulating `max_source_chars + 1`; it must not first materialize an unbounded chat, transcript, note set, or result set. Crossing the character ceiling aborts without storing a partial snapshot or running the tokenizer. Token counting runs only on an assembled in-bounds character snapshot. The RAG adapter additionally bounds document count and per-document/aggregate characters while accumulating results.

The standalone-HTML RAG adapter is an owner-local, retrieval-only source resolver. It uses a closed `slides_source_retrieval_v1` profile with answer generation, HyDE, LLM/VLM query expansion or rewriting, query decomposition, pre-retrieval clarification, LLM reranking, post-verification, adaptive generative reruns, and multi-turn synthesis disabled. Web fallback, discussion search, URL scraping, image search, video search, and every other external retrieval/egress path are also disabled. Caller input and stored RAG profiles cannot re-enable any of those stages. Its reranking strategy is limited to preinstalled, locally configured `flashrank`, `cross_encoder`, or `none`; request-time model download and any external or completion-capable service are rejected. The adapter formats only bounded owner-scoped `rag_result.documents`, never reads or falls back to `generated_answer`, and returns `404 rag_no_results` when no documents remain. Existing owner-local embeddings and non-generative retrieval may still operate only when they require no request-time external call. Consequently, each worker attempt has only the configured allowlisted standalone-HTML completion call; a pre-commit crash or retryable provider failure may repeat that call under the documented at-least-once model.

### Unified generation request

`POST /api/v1/slides/generations` is the single new asynchronous submission route. In v1, `generation_mode` is required and its only accepted value on this route is `standalone_html`. Structured generation remains on the existing per-source endpoints; it does not gain a second public transport in this release. The new route requires the `Idempotency-Key` header and accepts a discriminated source union:

```json
{
  "generation_mode": "standalone_html",
  "generation_config_revision": "sha256:<64-lowercase-hex>",
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

The top-level request, every source variant, and `html_options` are closed schemas: a strict JSON decoder rejects duplicate member names and nonfinite numbers before Pydantic, and unknown fields are rejected with `422`, including provider/model overrides. V1 bounds are:

- every JSON string must contain valid Unicode scalar values only; escaped or literal lone surrogate code points are rejected with a redacted `422` before trimming, UTF-8 byte counts, HMAC construction, tokenization, or source resolution;

- `generation_mode`: exactly `standalone_html`;
- `generation_config_revision`: exactly `sha256:` followed by 64 lowercase hexadecimal characters;
- `prompt`: nonblank after whitespace checking and subject to the advertised effective resolved-source character and token ceilings;
- `conversation_id`: nonblank and at most 256 UTF-8 bytes;
- `media_id`: integer from 1 through `9223372036854775807`;
- `note_ids`: 1 through 100 unique identifiers, each nonblank and at most 256 UTF-8 bytes;
- `query`: nonblank and at most 20,000 Unicode scalar values;
- `top_k`: default 8, from 1 through 100;
- `presentation_type`, `visual_direction`, and `delivery_style`: exactly the closed values listed in the creation form;
- `audience`: nonblank after trimming and at most 500 Unicode scalar values;
- `slide_count`: 1 through 30.

Every resolved source snapshot—including concatenated notes or retrieved RAG documents—must satisfy the same effective source character and token ceilings before queueing. The client cannot request chunking or override either ceiling on this route.

The endpoint returns `202 Accepted`:

```json
{
  "generation_id": "018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
  "status": "queued",
  "status_url": "/api/v1/slides/generations/018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
  "presentation_id": null
}
```

`generation_id` is the owner-scoped receipt UUID, not the reusable SQLite integer Jobs ID. `GET /api/v1/slides/generations/{generation_id}` performs an owner-scoped receipt lookup; unknown, malformed, and other-owner UUIDs are indistinguishable bounded `404` responses. A successful lookup returns that UUID, one of `queued | running | completed | failed | cancelled`, optional bounded progress text, and—only when applicable—`presentation_id`, `content_kind`, `error_code`, and a safe error message. It does not return generated HTML. Internal Jobs state maps as follows:

- `queued` → `queued`;
- `processing` → `running`;
- `completed` → `completed`;
- `failed` → `failed`;
- `cancelled` → `cancelled`;
- `quarantined` → `failed` with `generation_quarantined`.

The job result contains bounded metadata such as `presentation_id`, `content_kind`, document byte count, slide count, and validation status. The client fetches the authenticated presentation detail after completion.

A newly accepted request and a same-key replay whose stored state is `queued` or `running` return `202` with exactly `generation_id`, `status`, `status_url`, `presentation_id: null`, and optional bounded `progress_text`. A same-key `completed` replay returns `200` with `generation_id`, `status: "completed"`, `status_url`, nonnull `presentation_id`, and `content_kind: "standalone_html"`; it has no error fields. A same-key `failed` replay returns `200` with `generation_id`, `status: "failed"`, `status_url`, `presentation_id: null`, `error_code`, and bounded `error_message`. A same-key `cancelled` replay returns `200` with `generation_id`, `status: "cancelled"`, `status_url`, `presentation_id: null`, and `error_code: "generation_cancelled"`; it has no provider/error-detail echo. No other fields appear in these closed response variants. A replay never implicitly enqueues a replacement job. A different canonical request under the same key returns `409 generation_idempotency_conflict`.

### Request admission and validation redaction

Every REST route or MCP transport/tool that accepts **or rejects** a standalone discriminator, `html_document`, or other standalone source-bearing field is covered by pre-materialization inspection, fixed redacted validation/errors, and no body/trace/error-capture logging. New standalone transports use hard limits: `POST /api/v1/slides/generations` has a fixed 4 MiB ceiling, while the dedicated HTML-source save route and `draft-attachment` each have an exact 1,048,576-byte ceiling and accept raw bytes rather than JSON-escaped source. These fixed ceilings cover the closed V1 request unions without relying on an incorrect character-to-JSON expansion formula; decoded generation source still must satisfy the lower advertised character and token limits.

Existing generic structured Slides REST/MCP transports retain their current total-body, string, inline-image, and frame semantics; this feature does not impose an 8 MiB or other new aggregate cap on them. Instead, a specialized constant-memory JSON rejection prefilter scans the stream before ordinary object materialization and tracks only the exact shallow schema paths where `content_kind` or `html_document` could be supplied (REST top level; MCP arguments and their `updates`/`patch` object). Protocol adapters feed it at most 64 KiB of decoded data at a time; any supported request decompressor uses a bounded-output incremental API and never expands more than the next 64 KiB chunk in memory. The scanner decodes escaped key names, distinguishes keys from string content, and rejects `content_kind: "standalone_html"` or an `html_document` member before consuming/materializing that member's value. At most one bounded input chunk/lookbehind is held while earlier bytes follow the existing structured request path; implementations may use a restricted ephemeral spool to avoid adding heap amplification, but must delete it and must scan a chunk before spooling bytes at or after a forbidden key. Deep/tiny-node prefixes are processed lexically rather than built into Python/JavaScript containers. If no forbidden field appears, the exact payload proceeds through the existing structured decoder unchanged. Thus rejected HTML source is bounded and redacted without narrowing any legal structured request.

On the fixed-cap standalone routes, the ASGI receive wrapper permits only an absent or single `Content-Encoding: identity`; compressed request bodies are rejected. It rejects multiple, comma-joined, conflicting, negative, nondecimal, or otherwise invalid `Content-Length` values; rejects a valid declared length above the route limit; counts every decoded `http.request` chunk; stops buffering as soon as the limit is crossed; and drains or closes according to the server protocol before returning bounded `413` JSON. Missing or chunked `Content-Length` does not bypass the count. An endpoint dependency that calls `await request.body()` is not sufficient for this guarantee.

Before JSON materialization, an incremental UTF-8/JSON lexical state machine consumes those bounded chunks, handles split escapes/code points, and enforces structure without constructing arrays, mappings, or strings. The fixed V1 budgets are:

| Envelope | Max depth | JSON lexical tokens | Containers | Members + array items | Max encoded string token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standalone generation POST | 8 | 4,096 | 512 | 2,048 | 3 MiB |
| Provider success response | 64 | 200,000 | 25,000 | 100,000 | 7 MiB |

Crossing a standalone-generation structural budget returns fixed redacted `422 json_structure_too_complex`; crossing a provider budget produces the stored `standalone_html_provider_response_invalid` failure. Only a preflight-passing envelope reaches the strict duplicate-key/nonfinite/lone-surrogate decoder. These are allocation budgets, not schema allowances; the much smaller closed request/source limits still apply after decoding. Generic structured requests use the compatibility-preserving rejection prefilter above rather than this table.

For MCP WebSocket, the protocol server—not a route after `receive_json`—disables per-message compression for this endpoint and streams frame payload through the same schema-path rejection prefilter in at most 64 KiB pieces before concatenation/tool dispatch, even when the peer sends one large frame. It retains only bounded scanner state and stops before a forbidden source value; a permitted structured message is delivered byte-for-byte to the existing MCP path with its existing size semantics. HTTP MCP runs the prefilter across the bounded ASGI/protocol chunks above. No deployment may advertise standalone-aware MCP guards unless these protocol-layer settings are active.

The shared source-route sanitizer covers both supported and intentionally rejected standalone inputs. Its route-scoped `RequestValidationError`/MCP equivalent returns only a bounded allowlisted shape containing a stable code, sanitized field location, and fixed message. It emits at most 20 errors and four location components per error; each component is mapped through the closed schema's known field/index vocabulary, and any attacker-controlled or unknown key becomes the fixed token `unknown_field`. It never serializes raw Pydantic/MCP `loc`, arguments, `input`, `ctx`, `url`, exception representations, malformed JSON excerpts, or validator-provided source text. Every response route that can carry `html_document` also has a source-redacting `ResponseValidationError` and serialization-failure boundary: it returns a fixed bounded `500 standalone_html_response_invalid`, and logs only request ID, route, response schema/code, and presentation ID—never exception `input`, `repr`, response data, or source. Validation logs contain only request ID, route/tool, error count, and stable codes. Client error surfaces, reverse-proxy logs, tracing, and error-report integrations redact request/response bodies for all of these routes; existing unrelated-route validation behavior remains unchanged.

Supported separate-origin WebUI deployments require readable concurrency/download/backoff headers. Normal `CORSMiddleware`, any custom preflight path, and maintenance/drain/error gates therefore allow the existing authenticated request headers plus `Idempotency-Key`, `If-Match`, and `X-Slides-Accept-Content-Kinds`, and expose at least `Content-Disposition`, `ETag`, `Last-Modified`, `Retry-After`, `Content-Length`, and the existing request/trace headers on applicable responses. They preserve the configured origin allowlist, credential policy, and `Vary: Origin`; this feature never substitutes wildcard credentialed CORS. The frontend still verifies status, MIME, and fixed attachment name before Blob creation. A cross-origin test must exercise preflight, raw save/ETag adoption, attachment-header inspection, and `Retry-After` polling through both normal and drain/error paths.

### Provider execution

Standalone HTML generation is available only when:

- the feature flag is enabled;
- the emergency standalone-egress kill switch is clear;
- one concrete default execution target is configured as `(canonical_provider, model, adapter_id, endpoint_identity)`;
- that exact tuple is contained in the standalone HTML allowlist.

`endpoint_identity` is the normalized nonsecret `scheme://host:port/path` identity used by the adapter, with default ports normalized and userinfo, query, fragment, and credentials forbidden. Remote targets must use canonical HTTPS with ordinary hostname/certificate verification; `verify=False`, insecure custom transports, and plain remote HTTP fail configuration closed. The only HTTP exception is a built-in local adapter whose fixed destination is the literal IPv4 `127.0.0.1` or IPv6 `::1` loopback address. `localhost`, wildcard, private/LAN, link-local, DNS-resolved, IPv4-mapped nonloopback, and user-overridden destinations do not qualify. V1 standalone generation otherwise permits only built-in provider adapters using fixed product-owned endpoint identities. It rejects custom OpenAI-compatible adapters, provider base-URL overrides, proxies, app/environment endpoint overrides, fallback routers, and model/provider substitution. Aliases may normalize names but cannot change or bypass the concrete target comparison. The public request cannot override any target component.

Capability availability depends on valid static configuration and handler registration, not transient provider health or an external worker heartbeat; provider-call outages become safe job failures. The capability response includes an opaque nonsecret `generation_config_revision` derived from the enabled state, exact execution-target tuple, exact prompt digest/version, statically configured worker registration, digest-key availability, and effective input limits. A genuinely new request must echo it. Submission reads one immutable configuration snapshot, recomputes and compares its revision, and then uses the target, prompt, and limits from that same snapshot. A stale revision returns `409 generation_configuration_changed` before source resolution or enqueue so the form can refetch capabilities and ask the user to reconfirm the provider/model now shown. A matching existing idempotency receipt is located before this comparison and replays its stored configuration.

For a newly claimed request, the service resolves the exact bounded system-prompt text, computes its SHA-256 digest, and snapshots both with the complete execution-target tuple. An idempotent replay uses that stored tuple and prompt rather than substituting current defaults or a mutable prompt override. Immediately before every provider call, the worker rechecks the standalone feature flag, emergency egress kill switch, adapter's current normalized endpoint identity, and exact stored tuple against the current allowlist. If either kill control is active, the attempt terminally fails with `standalone_html_egress_disabled`, deletes the retained input under the normal terminal CAS, and makes zero provider calls; re-enablement requires a deliberate new generation. Any target change/removal fails with `standalone_html_endpoint_not_allowed` or `standalone_html_model_not_allowed`. The worker never falls back to another endpoint, adapter, provider, or model. Secrets and URL query data never enter the identity, receipt, provenance, logs, capabilities, or errors.

Provider calls use the existing abstraction through a standalone-specific bounded HTTP path, run without blocking the FastAPI event loop, and have a server-enforced connect/read/overall timeout and maximum output-token budget. This path explicitly bypasses the shared eager error-body capture hook, uses `trust_env=False` and `follow_redirects=False`, sends `Accept-Encoding: identity`, and sends only to the exact bound endpoint identity; proxy environment variables and HTTP redirects cannot reroute source. Before reading either success or non-2xx body bytes, it rejects every nonidentity or conflicting `Content-Encoding`. With identity encoding, `Content-Length > 8 MiB` rejects immediately and raw streamed iteration aborts at `remaining + 1` bytes when length is missing, dishonest, or chunked; no HTTP decompressor can allocate an oversized expansion first.

The incremental JSON preflight above consumes the same bounded success chunks before materialization. Only an in-bounds, structure-bounded successful response may be buffered and decoded with the strict JSON parser before extracting text. Non-2xx bodies are read/discarded only through the same raw 8 MiB ceiling and reduced to status plus a fixed provider error code; body bytes never enter an exception, log, trace, retry message, or receipt. Extracted HTML must then satisfy the independent 1 MiB document limit. An over-limit body is cancelled without logging or echoing it and fails with `standalone_html_provider_response_too_large`. Source material is sent to the one bound provider as untrusted user content. Delimiters and model allowlisting are not prompt-injection defenses: the generation call receives no tools, application credentials, cookies, internal service URLs, fallback route, or authority to fetch additional content.

### Crash-safe idempotency and worker correlation

Slides and Jobs use separate persistence stores, so the design does not rely on a cross-database transaction or place source material in a Jobs JSON payload.

Each user's Slides database contains a durable `slides_generation_receipts` ledger and an ephemeral one-to-one `slides_generation_inputs` row. The receipt UUID is the owner-scoped public `generation_id`; a reusable numeric Jobs ID is never a public identifier or correlation authority. The receipt stores canonical owner ID, keyed client-idempotency and client-request digests, the safe derived Jobs idempotency key, a keyed execution digest, digest-key ID, optional numeric Jobs ID as a nonauthoritative lookup hint, immutable Jobs UUID, optional presentation ID, state, bounded safe error fields, timestamps, and terminal expiry. The input row stores the resolved source kind/text/keyed digest/byte count, bounded provenance JSON, normalized HTML options, exact bounded system-prompt text/digest/version, complete execution-target tuple, and an `input_expires_at` fixed to 24 hours after receipt creation. It contains no provider secret or raw client idempotency key.

All receipt equality/correlation digests that could otherwise become offline dictionary oracles use domain-separated HMAC-SHA-256 under a dedicated installation keyring and are compared with constant-time primitives; this includes the client idempotency key, canonical client request, resolved source snapshot, and execution manifest. Canonical manifests use sorted-key, compact, UTF-8 JSON with `ensure_ascii=False`, integer-only numbers, closed schemas, and the field-specific trimming rules above; direct source text otherwise remains byte-for-byte significant.

The keyring has a maximum of four active/retiring keys and a source-free persistent metadata registry containing only key IDs, activation/retirement timestamps, and current/retiring state. Secrets remain outside databases. New receipts use only the current key. A retiring key cannot be removed until at least 32 days after it stopped being current, covering the 24-hour nonterminal source window plus the 30-day terminal receipt window; rotation faster than the bounded keyring permits is rejected. A key is actually removable only after a complete fenced dormant-database sweep proves that no unexpired receipt references it. Startup and configuration reload require secret material for every nonretired registry entry. If any is absent, the entire generation POST path—including apparently new keys and same-key replay—is disabled with `503 generation_digest_key_unavailable` before idempotency lookup, so an old replay can never be misclassified as a new request. The generation handler/acquire gate is also disabled: queued inputs are retained without a provider call until key restoration or their 24-hour logical expiry. A worker that observes key loss after acquisition releases/retries safely without classifying it as correlation failure; it discards any post-call output and cannot commit without the execution-HMAC check. Generation-ID status reads and already-saved presentation access remain available. Key secrets never enter Slides/Jobs databases, logs, provenance, capabilities, or errors; nonsecret key IDs do. Ordinary HTML content digests and the nonsecret packaged-prompt digest may remain unkeyed SHA-256.

At submission, the server first validates and canonicalizes only client-supplied fields and computes `client_request_hmac_sha256` over `generation_mode`, the echoed `generation_config_revision`, the complete discriminated source selector/content, and normalized `html_options`. Before resolving current capabilities, the default execution target, or mutable source content, it checks owner-scoped receipts by the bounded set of keyed client-key candidates:

- same client-request HMAC: return or recover the bound job/result from the receipt and, while nonterminal, its stored execution input without resolving the source or current default again;
- different client-request HMAC under the matched key: return `409 generation_idempotency_conflict`.

The original selector/query is intentionally not persisted merely to rederive the client digest. On incoming replay, it is recomputed from that incoming canonical request and compared with the stored authoritative HMAC. Worker correlation validates the stored HMAC format and key ID but never claims to reconstruct it from the execution input.

For a new key only, the endpoint checks generation capability, resolves the concrete allowlisted execution target and exact system prompt, authorizes and boundedly snapshots the source through the owner-scoped adapter, applies the source ceiling and a fixed 128 KiB UTF-8 system-prompt ceiling, and builds immutable provenance. It computes `execution_hmac_sha256` from a canonical manifest containing the stored client-request HMAC, source kind/HMAC/byte count plus the canonical provenance digest, normalized options, complete execution-target tuple, and exact prompt digest/version. A `BEGIN IMMEDIATE` Slides transaction rechecks the key and atomically inserts the receipt plus input snapshot. A racing loser discards its temporary resolution and returns the winner when the client-request HMAC matches.

Because the Jobs unique index is not owner-scoped, the server derives the internal Jobs key as `slides:v1:` plus a domain-separated HMAC over the canonical owner ID, a NUL separator, and the client key. The raw client key is not used as the global Jobs key; the safe derived key is stored on the receipt so recovery never depends on the client retrying with raw material. The fixed scope is `domain=slides`, `queue=default`, and `job_type=presentation.generate`.

The Jobs payload is deliberately tiny: exactly `{ "receipt_id": "<generation-uuid>" }`. Canonical owner identity comes from the Jobs row, while request/execution digests come from the owner-scoped receipt and input. The Slides submission path runs the existing Jobs payload normalization/secret-hygiene logic before creation and rejects any size, redaction, truncation, or structural mutation that would prevent that exact envelope. It does not add a generic exact-payload mode to JobManager. Every returned active/archive row and every worker claim must still deep-equal this envelope before binding or source access; a mutated payload fails closed without a provider call.

Recovery cannot depend on `create_job`'s admission order. Before calling `create_job`, the Slides service uses a Jobs helper that looks up an active or archived row by the complete `(domain, queue, job_type, jobs_idempotency_key)` scope without rerunning fair-share, quota, or capacity admission. Active lookup is preferred. Archive candidates are ordered newest first and are backed in SQLite and PostgreSQL by a nonunique partial scope index plus a unique partial index on nonnull `jobs_archive.uuid`. The migration scans first: duplicate nonnull UUIDs are treated as corruption and fail this generation capability closed until repaired; archive rows missing UUID cannot bind a standalone generation. Archive compression, mutation, and recovery address a row by immutable UUID, never by reusable numeric ID. A found row is accepted only when its UUID, owner, scope, type, key, and normalized decrypted payload all match; otherwise recovery fails safely. Only a genuine miss enters normal Jobs admission.

Submission and worker startup use the same idempotent compare-and-set bind helper. It may fill an unbound receipt only when owner, receipt UUID, Jobs scope, immutable Jobs UUID, derived Jobs key, and exact payload all match. A numeric Jobs ID may be stored alongside that UUID as a diagnostic/query optimization, but is accepted only with the same verified UUID and never establishes identity. For a nonterminal receipt, the helper recomputes the stored source and prompt digests from the input, validates the persisted client-request HMAC format/key ID without rederiving it, and recomputes the execution HMAC from that authoritative client HMAC plus the verified input. An already identical UUID binding is a no-op and any different binding is `generation_correlation_mismatch`. The API calls it before returning `202`. Because a worker may acquire the Jobs row first, the worker calls it again as its first domain action and may repair the unbound receipt.

A deterministic enqueue rejection that proves no Jobs row exists removes the still-unbound receipt/input transactionally before returning a bounded error. An ambiguous result leaves the receipt `claimed` and returns `503 generation_receipt_unresolved` with bounded `Retry-After`, no invented Jobs identifier, and instructions for the client to retry the same idempotency key. Retry and the background reconciler use its stored derived Jobs key to search active and archive state without normal admission. A matching row is bound; a confirmed miss persisting for 15 minutes terminally fails the receipt with `generation_receipt_unresolved` and logically deletes its input. Temporary Jobs-store unavailability is not a confirmed miss.

Every generated presentation stores the immutable originating Jobs UUID as `generation_job_uuid`, protected by a partial unique index within the owner's Slides database. Before resolving a per-user database path, the worker treats the Jobs row's owner as canonical. It validates the exact receipt-only payload and opens only that owner's Slides database. Any owner, UUID, scope, key, payload, digest, lease, or state mismatch is a nonretryable `generation_correlation_mismatch` failure with no provider call or cross-owner lookup. The worker algorithm is:

1. Correlate or compare-and-set bind by immutable Jobs UUID, then reload the owner-scoped receipt before requesting execution input. If the receipt is `completed`, verify that `presentation_id` exists and that the presentation's `generation_job_uuid` matches, then return bounded result metadata without loading input or calling the provider. If it is `failed` or `cancelled`, return the explicit WorkerSDK terminal outcome described below without loading input or calling the provider. Only a nonterminal bound receipt may load its unexpired input and recompute source, prompt, and execution digests as defined above.
2. Recheck feature/egress enablement and revalidate the exact stored execution-target tuple; only then reserve validator-queue capacity, call the model with the stored source and exact stored system prompt, and consume the reservation to validate output.
3. Immediately before the Slides commit, query Jobs by immutable UUID and verify exact owner/scope/payload, processing state, live lease/claim, and absence of cancellation or terminal state. A failed check discards the result and enters reconciliation; it never commits.
4. In one Slides write transaction, recheck the binding, nonterminal receipt, and unexpired input; insert the presentation with its initial entity snapshot, derived FTS text, stored HTML byte/slide counts, bounded immutable `generation_provenance_json` copied from the input, and `generation_job_uuid`; update the receipt to `completed` with its presentation ID and terminal expiry; and delete the input row. If another retry already committed it, verify and return that row instead. If the receipt expired or became terminal while the provider ran, discard the result.
5. Return bounded presentation metadata so WorkerSDK can complete Jobs.

Cancellation is best-effort across the two stores. The final Jobs check closes the ordinary race, but no cross-database transaction can eliminate the last interval between that check and the Slides commit. Deterministic reconciliation therefore gives a valid `completed` receipt plus its matching committed presentation precedence: it is immutable and is never downgraded by a later or racing Jobs `cancelled`, `failed`, TTL, quarantine, or prune state. A Jobs terminal state maps into a receipt only while the receipt is nonterminal and no matching committed presentation exists. If generic cancellation wins the receipt CAS before the Slides transaction, the transaction's nonterminal check fails and no presentation is created. Status reports `completed` when the presentation commit wins and emits a safe correlation metric when Jobs retains a conflicting terminal state.

Every typed retryable precommit failure—including provider/transport failure, transient Jobs/Slides-store unavailability or lock, and a recoverable lease/state check—does not terminalize the receipt or delete input. The domain path compare-and-set returns it to `queued` with bounded retry metadata and raises the typed retryable failure for WorkerSDK. Success, explicit cancellation, nonretryable validation/correlation failure, validator watchdog failure after a completed provider response, quarantine, expired input, or exhausted retry budget terminalizes the receipt and logically deletes input. Generic Jobs-terminal-first paths are reconciled under the precedence rule above. Public generation status performs the same synchronous idempotent reconciliation before returning.

This feature adds a closed `WorkerTerminalOutcome` return variant to WorkerSDK with only `failed` or `cancelled`, stable error code, and bounded safe message. On that variant WorkerSDK calls a JobManager expected-state/lease/UUID CAS terminalizer and explicitly skips `complete_job`; an already identical terminal Jobs state is a no-op, while a conflicting completed state is a correlation error. A normal bounded metadata return remains the only path to Jobs `completed`, and typed retryable exceptions retain existing retry behavior. Thus a preterminalized receipt can never be converted into a completed Jobs row merely because its handler returned normally.

`input_expires_at` is an absolute logical access deadline. Before it, a processing Jobs row with an unexpired lease is not stale. At 24 hours, no new provider call may begin; the reconciler terminally fails a nonterminal receipt with `generation_expired` and deletes its input row even if Jobs is unavailable. That overdue transition uses logical `terminal_at = input_expires_at` and `expires_at = input_expires_at + 30 days`, even when applied after long downtime; restart time never extends retention or key requirements. A worker already inside the provider call rechecks the receipt and input in the commit transaction and discards its result. When Jobs is reachable, a new Slides-scoped JobManager transition supports expected-state/UUID/owner/domain/type CAS for queued or processing `presentation.generate` rows; it does not reuse another domain's allowlist.

The reconciler must discover dormant per-user Slides databases, not only request-opened cache entries. A source-free shared Jobs-store coordination row provides a renewable single-leader lease, fencing token, continuation cursor, last-complete epoch, and measured sweep lag. Only the current fenced leader advances the cursor; lease expiry permits takeover after a crash, and stale leaders cannot publish progress. Receipt transitions themselves remain idempotent owner-scoped CAS operations. Every API/worker process keeps standalone generation admission disabled until it observes a shared startup-complete epoch produced after its deployment/config epoch; Jobs-store/leader unavailability therefore fails generation closed rather than letting one process skip cleanup.

Under that lease, startup before generation-handler admission and continuous runtime sweeps stream the canonical one-level user-database registry/root with path-containment, regular-file, schema, and no-symlink checks; open at most one database at a time; and resume through the fenced cursor. Active Jobs owners are prioritized. A complete sweep must finish at least every 15 minutes; if observed cardinality or I/O cannot meet that bound, new standalone generation fails closed with `generation_reconciler_overloaded` until the lag recovers. The startup sweep first deletes receipts whose deterministic expiry already passed, then verifies every retiring-key reference before allowing key retirement or handler admission. This bounds open handles/memory and makes the running-system logical purge bound 24 hours plus at most 15 minutes.

Logical deletion does not promise forensic erasure from SQLite freelists, WAL files, filesystem snapshots, or backups. Operations documentation covers WAL checkpointing, backup retention, and secure-media requirements for deployments that need physical-erasure guarantees. After downtime, expired rows may remain physically present until the mandatory startup sweep, but handler startup ordering ensures they are inaccessible to provider execution.

The 30-day idempotency promise is provided by the durable receipt, not Jobs retention: administrators may archive or prune Jobs earlier. A terminal receipt replays its stored safe result after Jobs removal, and the unique presentation `generation_job_uuid` prevents a second commit from the same immutable job. Provider invocation is necessarily at-least-once under WorkerSDK: a process crash after the provider returns but before the Slides commit may repeat the provider call and its cost. Transport replay does not create another Jobs row, and any retry after the presentation commit follows step 1 without calling the provider. If a nonterminal receipt loses all Jobs evidence, the 15-minute confirmed-miss rule terminally resolves it rather than implicitly enqueueing. Thirty days after terminal state, cleanup may remove receipt metadata; a deliberate new generation then uses a new key. This design does not claim exactly-once external side effects.

## Standalone HTML Validation

One pure Python backend validator is authoritative after generation, before persistence, on every HTML save or restore, and before every saved standalone HTML or JSON export. The browser's non-loading safe-outline parser is only a responsiveness and display aid; it is never authoritative for persistence and never executes the source.

The validator uses directly declared and locked `html5lib` and `tinycss2` runtime dependencies. Before invoking either library, a conservative single-pass state machine scans the already byte-bounded UTF-8 source. It enforces potential HTML token, start-tag, attribute, apparent-nesting, unterminated-quote/comment, and single-token budgets and refuses over-budget input before tree construction. It is not a regular-expression sanitizer and does not bless input; false-positive refusal is acceptable. `html5lib` then runs in strict fail-on-first-parse-error mode through a counting token-stream/tree-builder boundary that refuses to create the 10,001st element, 20,001st attribute, or 129th actual nesting level. The builder aborts at the boundary rather than checking only after a complete tree exists.

Each collected `<style>` text is measured before `tinycss2` runs. A second linear lexical preflight enforces aggregate CSS bytes/tokens, block/function nesting, single-token, declaration-candidate, stylesheet-count, and unterminated string/comment budgets. Only in-budget CSS reaches `tinycss2`; its result is walked with an explicit stack, with the same depth/token/declaration ceilings and a bounded parse-error count. `tinycss2` supplies escaped-identifier, at-rule, function, nested-block, and URL-token semantics for reject-on-match policy. The validator rejects parse errors rather than sanitizing or rewriting source, makes no security decision with regular expressions, and fails generation capability closed if either dependency is unavailable. They are direct project dependencies rather than optional/transitive Bleach features, and a clean-install smoke test imports and exercises both paths.

Validation runs outside the event loop in killable subprocess workers behind bounded admission. V1 uses at most four validator subprocesses per server process, a 24-entry high-priority queue for authenticated save/restore/export work, and a separate eight-entry generation-reservation queue. A generation worker must acquire one low-priority reservation before it calls the provider and consume it when the returned document is queued, so 32 slow provider calls cannot occupy or starve the interactive queue and local saturation never causes an otherwise successful provider call to be repeated. Workers use bounded weighted scheduling so neither queue starves. Queue saturation returns `503 standalone_html_validator_busy` with bounded `Retry-After` before parsing or, for generation, before provider dispatch.

The lexical/parser counters are the deterministic rejection mechanism and return nonretryable `422 standalone_html_validation_budget_exceeded` immediately at a stated budget. Separately, a generous 60-second watchdog handles an implementation hang: the parent terminates and replaces that subprocess rather than cancelling an unkillable thread/future, discards all partial output, and returns `503 standalone_html_validator_timeout`. Generation treats that timeout as a terminal safe failure rather than automatically repeating a completed provider call; a user may deliberately retry with a new idempotency key. No partially derived record is committed. These controls apply identically to generation, save, restore, and saved export validation.

Save and restore return bounded diagnostic codes and locations when validation fails; error responses never echo source. The editor may download a malformed current draft for recovery, but cannot persist it. V1 adds no standalone draft-validation endpoint because no execution flow needs a separate validation handshake.

Delivery style is a generation-time instruction, not immutable post-generation metadata. Generation validation receives the stored options and enforces exactly one direct-child `.notes` element per slide for `speaker-led` and none for `self-guided`. The base persistence validator used by later save/restore permits zero or one `.notes` direct child per slide, rejects notes elsewhere/duplicates, and classifies every such node as speaker notes regardless of original style. Users may therefore add/remove notes while editing without the client needing deleted generation options; search always excludes them and the safe outline always isolates any present notes under its labelled disclosure.

Default hard limits are:

- 1 MiB UTF-8 document size;
- 1 through 30 `.slide` sections;
- 50,000 potential/consumed HTML tokens and 65,536 bytes per lexical token;
- 10,000 parsed HTML elements;
- 20,000 total attributes;
- maximum parsed element nesting depth of 128;
- strict abort on the first HTML parse error;
- 64 `<style>` elements and 524,288 aggregate UTF-8 CSS bytes;
- 100,000 CSS tokens, 10,000 declaration candidates/declarations, 65,536 bytes per CSS token, block/function nesting depth 64, and at most 100 CSS parse errors;
- 250,000 indexable semantic-text characters before indexing truncation;
- the existing source-input ceiling and a server-enforced generation token ceiling.

These HTML document/parser limits are immutable in V1 so every saved document remains editable and downloadable. Changing them later requires a capability-schema/version change and an explicit compatibility policy. Standalone generation computes its effective source limits as the lower of existing Slides configuration and fixed V1 maxima of 200,000 characters and 50,000 tokens; administrators may lower either value but cannot raise the V1 transport/storage exposure through configuration. Raising those maxima later requires a capability-schema/version change. The provider output-token budget remains independently configured under its own fixed server safety ceiling.

Validation rejects:

- invalid UTF-8, NUL bytes, C0/C1 controls other than HTML whitespace U+0009/U+000A/U+000D, incomplete document structure, or likely truncation;
- missing doctype, head, body, title, or slide sections;
- excess size, slide count, nodes, attributes, or nesting depth;
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
- bounded indexable semantic text excluding script, style, template, noscript, speaker notes, and generated deck chrome;
- safe diagnostic codes without echoing source or secrets into logs.

Search extraction is syntactic, not computed visibility: it never evaluates CSS. It iteratively visits `.slide` ASTs and gathers normalized text only from `h1`–`h6`, `p`, `li`, `dt`, `dd`, `blockquote`, `pre`, `code`, `caption`, `th`, `td`, and `figcaption`, while excluding any subtree rooted at the active/noncontent elements above or `.notes`, `.deck-header`, `.deck-footer`, `.slide-number`, and progress/navigation chrome. It never uses recursive `.text()`/`textContent` helpers. Truncation is deterministic and does not mutate source.

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

- `id TEXT PRIMARY KEY`, the public owner-scoped `generation_id` UUID
- `owner_user_id TEXT NOT NULL`, the canonical Jobs/Slides owner identifier
- `digest_key_id TEXT NOT NULL`, selecting the bounded installation HMAC keyring
- `idempotency_key_hmac_sha256 TEXT UNIQUE NOT NULL`
- `jobs_idempotency_key TEXT UNIQUE NOT NULL`, the safe derived `slides:v1:<hex>` key, never the raw client key
- `client_request_hmac_sha256 TEXT NOT NULL`
- `execution_hmac_sha256 TEXT NOT NULL`
- `job_id INTEGER NULL`, a nonauthoritative numeric lookup hint only
- `job_uuid TEXT NULL`, the immutable Jobs correlation identity
- `presentation_id TEXT NULL`, referencing `presentations.id`
- `receipt_status TEXT NOT NULL`, constrained to `claimed | queued | running | completed | failed | cancelled`
- bounded nullable `error_code` and `error_message`
- `created_at`, `updated_at`, and nullable `expires_at` assigned only at terminal transition

The owner-scoped `slides_generation_inputs` table contains one ephemeral row per nonterminal receipt:

- `receipt_id TEXT PRIMARY KEY`, referencing the receipt;
- `source_kind`, `source_text`, `source_hmac_sha256`, and `source_bytes`;
- bounded `provenance_json` and normalized `html_options_json`;
- canonical `provider`, `model`, `adapter_id`, and normalized `endpoint_identity` identifiers;
- bounded exact `system_prompt`, `prompt_sha256`, and `prompt_contract_version`;
- `input_expires_at TEXT NOT NULL`, exactly 24 hours after receipt creation in V1;
- `created_at`.

The receipt stores no raw client key, source material, prompt, model output, or HTML. The input row stores no credentials; its system prompt is administrator-resolved application configuration, not source material, and is deleted with the other execution input. Successful commit copies the exact bounded normalized provenance into `presentations.generation_provenance_json` before deleting the input; failed/cancelled terminal reconciliation deletes the input. The receipt and presentation binding remain for the idempotency window.

Generation provenance is a server-built closed schema with no extra fields:

```json
{
  "schema_version": 1,
  "source_kind": "prompt",
  "source_ref": null,
  "source_snapshot_hmac_sha256": "<hex>",
  "digest_key_id": "slides-generation-v1",
  "source_bytes": 1234,
  "provider": "canonical-provider",
  "model": "allowed-default-model",
  "adapter_id": "built-in-adapter",
  "endpoint_identity": "https://provider.example/v1",
  "prompt_sha256": "<hex>"
}
```

`source_ref` is null for direct material, an owner-scoped identifier capped at 256 UTF-8 bytes for a single chat/media source, or a domain-separated keyed HMAC reference for multi-note and RAG selections; it never contains prompt, note, chat, media text, or a RAG query. The summary projection exposes only `{source_kind, provider, model}`. The complete object intentionally retains the nonsecret bound adapter and endpoint identity for owner-visible audit, is immutable, is copied into version snapshots, and is validated against the 4,096-byte cap before the source input can be deleted.

The canonical invariant is:

- `structured_slides`: validated slide list; `html_document`, `html_sha256`, `html_bytes`, `html_slide_count`, `generation_job_uuid`, and `generation_provenance_json` are null.
- `standalone_html`: nonblank validated `html_document`; matching digest, byte count, and slide count; stored `slides` is `[]`; `generation_job_uuid` is nonnull; and bounded valid generation provenance is present and immutable in v1.

`slides_text` remains the canonical derived FTS source for both kinds. Clients can never supply it.

All paths that create or mutate persisted records, restore versions, or export saved content—including REST, MCP, and workers—enforce the invariant through one domain service. The bounded `draft-attachment` recovery echo is not a persistence or saved-export path and intentionally applies only its owner, kind, size, UTF-8, and NUL checks. Only the verified generation worker may create a `standalone_html` record in v1; generic presentation-create and MCP-create requests for that kind return `409 standalone_html_creation_requires_generation`. Partial updates first merge with the current record, then validate the complete candidate inside the optimistic-concurrency operation. Omitting `content_kind` preserves the current kind; it never converts an HTML project into an empty structured project. A partial unique index on nonnull `generation_job_uuid` provides worker retry deduplication. Standalone HTML provenance and generation correlation are immutable; HTML duplication is not exposed in v1.

The dedicated HTML-source save operation accepts only the complete raw UTF-8 document bytes; the server derives the record title, digest, byte count, `html_slide_count`, and `slides_text` from that candidate in the same write transaction. Generic JSON presentation PATCH rejects `html_document` and every standalone title, provenance, FTS, digest, byte/slide-count, or generation-correlation field rather than ignoring it.

Content kind cannot change in v1. A request that attempts to change it returns `409 content_kind_immutable`.

### Migration

Slides schema versioning becomes authoritative at schema version 2. New databases are created directly at v2. Existing schema 0/1 databases enter `BEGIN IMMEDIATE`, re-read the version and actual columns after acquiring the lock, apply individual migration statements without `executescript`, add/backfill the presentation fields, create both generation tables and indexes, normalize `schema_version` to exactly one row containing `2`, and commit atomically. Any statement failure rolls back the whole migration. Legacy rows become `structured_slides` with null HTML fields.

The v2 runner replaces the ad-hoc ensure-column path for fields introduced by this feature; unrelated legacy compatibility helpers may remain until separately migrated. Tests cover a legacy database, an empty version table, inconsistent multiple version rows, an already migrated database, rollback after an injected failure, and concurrent first access from separate connections/processes. New row mapping and summary queries use explicit projections rather than depending on `SELECT *` positional/dataclass compatibility. Deployment documentation requires a database backup before upgrade and treats this migration as forward-only for old binaries.

A separate Jobs schema migration adds the archive scope index and unique partial index on nonnull immutable UUID, then changes archive compression/update helpers to address exactly one row by UUID. It audits existing archive rows before index creation. Duplicate nonnull UUIDs or ambiguous legacy rows do not crash unrelated Jobs/structured Slides operation, but they keep standalone HTML generation disabled with an administrator-facing repair diagnostic; they are never auto-deduplicated by numeric ID. New `presentation.generate` jobs require a nonnull UUID before acceptance.

### Summaries and search

List and search queries use lightweight projections that exclude `html_document` and full version payloads. Summaries include `content_kind`, title, the bounded `{source_kind, provider, model}` provenance summary, timestamps, version, and either structured slide count or stored HTML slide/byte metadata.

HTML search indexes only the bounded indexable semantic text derived by the server in the authoritative create/save/restore transaction. It does not attempt computed CSS visibility. Raw markup, JavaScript, CSS, speaker notes, and deck chrome are never indexed or emitted as result snippets.

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

Snapshots serialize as compact UTF-8 JSON with `ensure_ascii=False`; they do not turn non-ASCII source into six-byte `\uXXXX` sequences. Because valid quotes, backslashes, tabs, and newlines can still require JSON escaping, the serialized snapshot has a fixed ceiling of `2 × max_document_bytes + 65,536` bytes. The global source-control policy above makes that ceiling sufficient for every valid 1 MiB document plus bounded metadata; a candidate exceeding it fails atomically before changing the entity.

To bound full-document amplification, standalone HTML retains the newest 25 entity snapshots per presentation by default; retention is configurable downward. Retention cleanup occurs in the same successful mutation transaction, never removes the current entity, and a request for a pruned version returns `404 presentation_version_not_found`. Full-document duplication inside those bounded snapshots is accepted for v1 instead of introducing delta storage.

## API Contracts

### Capabilities

`GET /api/v1/slides/capabilities` separates persistence/editor support from generation availability. A temporary provider failure must never make already-saved HTML projects inaccessible. V1 returns this exact shape:

```json
{
  "schema_version": 1,
  "content_kind_request_header": "X-Slides-Accept-Content-Kinds",
  "content_kinds": {
    "structured_slides": {
      "read": true,
      "edit": true
    },
    "standalone_html": {
      "read": true,
      "edit": true,
      "export_attachment": true,
      "draft_attachment": true,
      "reason": null,
      "limits": {
        "max_document_bytes": 1048576,
        "max_source_write_bytes": 1048576,
        "max_draft_attachment_bytes": 1048576,
        "max_slides": 30,
        "max_nesting_depth": 128
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
      "model": "allowed-default-model",
      "adapter_id": "built-in-adapter",
      "endpoint_identity": "https://provider.example/v1",
      "generation_config_revision": "sha256:<64-lowercase-hex>",
      "input_limits": {
        "max_request_bytes": 4194304,
        "max_source_chars": 200000,
        "max_source_tokens": 50000,
        "max_audience_chars": 500,
        "max_source_identifier_bytes": 256,
        "max_note_ids": 100,
        "max_rag_query_chars": 20000,
        "max_rag_top_k": 100
      },
      "output_limits": {
        "max_provider_response_bytes": 8388608,
        "max_document_bytes": 1048576
      }
    }
  }
}
```

The numeric values above show the effective values under default configuration; 200,000 source characters and 50,000 source tokens are also the fixed V1 maxima. The response always contains effective limits after downward configuration clamping, not untrusted raw configuration. The fixed 4 MiB `max_request_bytes` is enforced by the receive-time guard and accommodates worst-case escaping for every closed V1 generation request; it does not raise `max_source_chars` or `max_source_tokens`, which are both authoritative after source resolution. The 1 MiB source-write and draft limits are raw-byte transport limits. `generation_config_revision` changes whenever enabled state, any execution-target tuple component, exact prompt digest/version, static handler registration, digest-key availability, reconciler admission state, or effective input/output limits change. It is nonnull only when generation is enabled and is required for a genuinely new generation request; a same-key receipt replay uses its stored configuration without comparing the current revision.

Disabled generation returns `enabled: false`, null provider/model/adapter/endpoint/revision identifiers, the effective limits, and one safe reason: `feature_disabled`, `egress_disabled`, `default_model_not_configured`, `default_model_not_allowed`, `default_endpoint_not_allowed`, `prompt_asset_unavailable`, `digest_key_unavailable`, `generation_worker_unavailable`, `generation_reconciler_overloaded`, or `validator_unavailable`. Execution-target, prompt, key, reconciler, or worker configuration failures do not alter the standalone content-kind flags: read/edit/safe-outline/download support remains available independently.

Missing validator dependencies are different because saves, restores, and saved exports require authoritative validation. In that misdeployment, `standalone_html.read` and `draft_attachment` remain true because inert JSON reads and the bounded raw recovery echo do not trust the document; `edit` and `export_attachment` become false; `content_kinds.standalone_html.reason` and the generation reason are `validator_unavailable`. Structured behavior remains available. Capability discovery performs no live provider health check. The endpoint requires the normal Slides authentication and uses `Cache-Control: private, no-store` plus the applicable authentication `Vary` headers; the creation page fetches it on entry and on explicit Retry.

Servers advertise `standalone_html` read/edit/export support when generation alone is disabled; validator unavailability follows the narrower fail-closed behavior above. Safe outline is fixed V1 frontend behavior, not a server-configurable capability. The creation form checks `generation_modes.standalone_html.enabled`; list, detail, editor, versions, saved export, and draft download check their specific content-kind flags. Generation failure disables only generation and blocks neither structured creation nor existing HTML access. The frontend does not infer support from unrelated Slides routes.

### Content-kind negotiation and legacy clients

REST clients opt into additive presentation kinds with the closed request header `X-Slides-Accept-Content-Kinds`. The current WebUI and extension send `structured_slides,standalone_html` on every Slides list/search/detail/version/mutation/delete/restore/export/render request; the dedicated HTML save/download clients send it as well. Capabilities and the mode-explicit generation/status routes do not require it. The header is a representation-safety negotiation signal, not authorization, and HTML ownership/kind guards still run normally.

An omitted header preserves legacy behavior: list/search queries filter at the database level to `structured_slides` before pagination/counting, and any presentation-targeting request that resolves to an HTML presentation/version returns bounded `406 content_kind_not_accepted` before source loading, mutation, export, or render dispatch and without an empty `slides` representation. Supplying only `structured_slides` behaves the same. A malformed header returns bounded `400`; unsupported tokens are ignored only if at least one known requested kind remains, and never opt a client into that unknown kind. Opted-in clients receive the discriminated union below. Responses that vary by this negotiation include `Vary: X-Slides-Accept-Content-Kinds` in addition to auth/origin variation. MCP uses its explicit matrix rather than this REST header.

This prevents a genuinely old client from ignoring `content_kind` and silently rendering `standalone_html` as `slides: []`; it will either never list the row or receive an explicit unsupported-representation error on a stale/direct link. Structured records and structured-only clients keep their existing representation and mutation behavior.

### Presentation response

Detail responses form a discriminated union:

- structured detail contains `content_kind: "structured_slides"` and `slides`;
- HTML detail contains `content_kind: "standalone_html"`, `html_document`, `html_sha256`, `html_bytes`, and stored `html_slide_count`.

HTML source is present only in authenticated detail, HTML-source save/restore, version-content, and explicit export responses, not summaries, search snippets, delete responses, version lists, job summaries, error bodies, or logs. HTML DELETE returns a bounded metadata-only tombstone `{id, content_kind, deleted_at}` rather than adapting the existing full presentation response. REST and MCP version-list queries use dedicated database projections that do not select or deserialize `payload_json` or `html_document`; only the separately authorized version-content route loads a snapshot payload. Every non-attachment response containing `html_document` uses `Content-Type: application/json`, `X-Content-Type-Options: nosniff`, `Cache-Control: private, no-store`, and varies on the authentication mechanism in use (`Authorization`, `X-API-KEY`, and/or `Cookie`). No source-bearing route returns or content-negotiates `text/html`. Request/response body logging, tracing, exception representation, and error-report payload capture are disabled for every raw-source route. Unknown content kinds render as an unsupported read-only state in clients.

### Update and concurrency

HTML Save uses `PUT /api/v1/slides/presentations/{presentation_id}/html-source` with required `Content-Type: application/octet-stream`, required strong `If-Match`, and the complete UTF-8 document as the raw request body. It accepts only an existing owner-scoped `standalone_html` presentation. A wrong media type returns `415`; a structured target returns `409 operation_not_supported_for_content_kind`; receive, UTF-8/NUL, validation, and storage limits use the stable errors defined below. The server derives title and all source metadata, persists atomically, and returns the authenticated HTML detail union as JSON plus the resulting strong ETag. A complete canonical match is a no-op with the existing ETag.

Standalone detail/save/restore responses return strong tags such as `"v7"` and temporarily accept either strong or legacy weak tags when parsing `If-Match`. Existing structured endpoints continue returning and accepting their current weak `W/"v7"` contract; this feature does not change them. A stale HTML-source tag returns `412` with bounded current-version metadata, not the remote HTML body unless the client explicitly fetches it. The existing generic JSON presentation PATCH remains structured-only for mutable content: any `html_document` or HTML title/provenance/digest/correlation field returns `409 operation_not_supported_for_content_kind` and is never buffered under a larger structured-body allowance.

### Operation matrix

“Supported” in this matrix describes the authenticated, content-kind-opted-in REST/domain capability. Legacy REST clients receive the structured-only filtering/406 behavior above. Secondary surfaces may intentionally expose a narrower subset; MCP's exact V1 subset is defined below.

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

The separately authenticated `format=json` export is an API-only V1 attachment with the fixed filename `presentation.json`, exact `Content-Type: application/json`, `Content-Disposition: attachment; filename="presentation.json"`, `X-Content-Type-Options: nosniff`, `Cache-Control: private, no-store`, and the same referrer/cross-origin protections. It carries the discriminator and source as a JSON string, never as HTML. The HTML workspace does not expose or create a Blob URL for JSON export in V1; adding that browser transport later must preserve the same non-navigation boundary.

Saved-version downloads include ETag and Last-Modified. No route serves stored HTML inline as `text/html`. Before download, the UI states that the file contains executable code and that opening it locally occurs outside tldw's security boundary.

After an authenticated download response passes required status, exact `Content-Type: application/octet-stream`, and attachment-header checks, the WebUI may wrap its exact bytes in an `application/octet-stream` Blob. It assigns the object URL only to a temporary application-owned `<a download="presentation.html">` with no `target`, triggers the download, and removes the anchor in `finally`. Revocation is scheduled on the next safe task after the browser has accepted the download and no later than one second; `pagehide`/unmount cleanup revokes it sooner when applicable. This avoids premature cross-browser cancellation without retaining the URL. The URL is never persisted, logged, previewed, passed to a worker, or assigned to `window.location`, `window.open`, an iframe, image, script, object, embed, or other navigation/resource sink. No other path creates a Blob URL from HTML source.

## Safe Outline Security Contract

### Safe outline pipeline

The automatic outline:

1. sends the already 1 MiB-bounded text to a statically bundled application-owned parser Web Worker; no worker source, module name, import, or URL is derived from the document;
2. inside that worker, runs a conservative one-pass, linear-time scanner that stops after 50,000 total potential HTML/node tokens, 10,000 potential start tags, 20,000 potential attributes, 10,000 comments/declarations, 20,000 text-run transitions, 128 apparent nesting levels, or any 65,536-byte token/comment/text run; malformed quoting or an unterminated token fails preflight;
3. declines to invoke the outline parser and returns the fixed **Outline unavailable — document too complex** state when a structural preflight budget is exceeded;
4. otherwise parses the current source into a non-DOM AST with the frontend's existing lazily loaded `cheerio/slim` dependency, which performs no browser DOM construction, custom-element upgrade, or subresource load;
5. traverses every AST node with an explicit iterative stack, aborts before the 50,001st total node or 129th actual depth, visits only `.slide` semantic descendants for output, and never calls Cheerio's recursive `.text()` or `textContent` helpers;
6. groups semantic text from `h1`–`h6`, `p`, `li`, `dt`, `dd`, `blockquote`, `pre`, `code`, `caption`, `th`, `td`, and `figcaption`; whenever `.notes` nodes are present, it extracts them separately under a labelled **Speaker notes** disclosure regardless of the original generation style;
7. discards every original attribute and all CSS, scripts, SVG, MathML, templates, noscript content, links, images, fonts, forms, embeds, active objects, and generated deck chrome;
8. emits at most 30 cards, 4,096 Unicode scalar values per semantic block, 20,000 per slide, and 100,000 total. Text-cap truncation appends an application-owned marker; structural refusal shows the bounded error above. Neither path mutates editor source;
9. removes C0/C1 and bidi-formatting controls and returns only a closed, capped structured-clone DTO containing the buffer digest and plain strings—never AST nodes, markup, attributes, URLs, or error excerpts;
10. validates that DTO and digest again on the main thread, then rebuilds cards with application-owned components and text nodes using `dir="auto"` plus bidi isolation.

The scanner is a responsiveness guard, not an HTML validator or sanitizer. It does not use a catastrophic-backtracking regular expression, and false-positive refusal affects only the outline; backend validation remains authoritative. The controller permits at most one active parse plus one coalesced latest pending buffer; a newer edit replaces and releases the prior pending snapshot rather than enqueueing another 1 MiB structured clone. A ten-second worker watchdog covers implementation hangs: timeout terminates and replaces the static worker, discards all partial/stale output, and starts only the coalesced latest buffer. Latest-wins digest checks prevent a timed-out or stale response from replacing a newer outline, while the queue rule bounds retained source copies under rapid edits.

The safe-outline path creates no iframe, `srcdoc`, `DOMParser`, `innerHTML`, `insertAdjacentHTML`, source-derived Blob/data URL, popup, generated-code worker, or generated-HTML route. Application-owned editor/parser workers may process bounded source strictly as inert text; they may not evaluate it, construct browser DOM from it, load resources, or interpret any returned value as markup. The download-only Blob exception is not an outline path. The outline preserves no model DOM node and never calls a backend validation endpoint merely to repaint. No regular-expression CSS sanitizer is introduced because no model CSS reaches this view. It never claims visual fidelity; the backend remains authoritative for generation, save, restore, and export.

### No V1 execution surface

Validation and safe-outline extraction may parse `html_document` into non-executing ASTs, while the code editor and trusted application workers handle it only through value/text APIs. No WebUI component, server renderer, MCP path, extension surface, worker, iframe, new window, or attachment endpoint inserts or renders it as browser markup or executes its script. No execution context, worker code, worker URL, module, or function is created from source. The application CSP remains unchanged: this feature adds no `unsafe-inline`, `unsafe-eval`, runtime origin, or `frame-src` exception.

The only HTML-workspace handoff of an executable standalone file is an authenticated attachment using `application/octet-stream`, `nosniff`, fixed `presentation.html`, and `private, no-store`. Authenticated detail, save/restore, version-content, and API-only JSON export carry source only as inert JSON, text, or octet-stream data under `private, no-store`, never as `text/html`. The WebUI provides no inline view or **Open in new tab** action. Its warning states that opening the downloaded HTML file may execute code outside tldw's security boundary.

Adding execution later requires a new design and security review; implementation must not expose a dormant runtime flag or reuse the safe outline as an execution substrate.

## MCP And Secondary Consumers

The MCP Slides module currently duplicates REST generation, persistence, restore, and export behavior. This feature must not add another independent implementation.

The shared domain service becomes authoritative for content invariants, summary mapping, update/restore, operation guards, and export dispatch. MCP's exact V1 behavior is:

| MCP tool | `standalone_html` V1 behavior |
| --- | --- |
| `slides.presentations.list`, `slides.presentations.search` | Supported metadata summaries; never source |
| `slides.presentations.get` | Supported metadata/provenance/version summary; omits `html_document` |
| `slides.presentations.delete`, `slides.presentations.restore` | Supported soft-delete/undelete; response uses the metadata-only mapper |
| `slides.versions.list` | Supported metadata only: presentation ID, version, content kind, and timestamps; never `payload` |
| `slides.versions.get`, `slides.versions.restore` | Rejected |
| `slides.presentations.create`, `update`, `patch`, or `reorder` when targeting/requesting HTML | Rejected |
| every `slides.generate.*` request for HTML | Rejected; existing structured generation remains supported |
| `slides.export` for HTML, including `json` | Rejected for every format |
| attachment or source retrieval/mutation | Rejected |

Every rejection returns `{ "success": false, "error": { "code": "operation_not_supported_for_content_kind", "operation": "<exact MCP tool name>", "content_kind": "standalone_html" } }`. All MCP metadata, target-kind guard, get, delete/undelete, and version-list paths use dedicated owner-scoped ID/kind/version/deleted/summary database projections that never select or deserialize `html_document` or version `payload_json`. Guards run from those projections before full-row mapping, deserialization, restore, export dispatch, or base64 encoding. In particular, the current full-row mapper and full-payload version mapper are never used for supported HTML metadata operations or `slides.versions.list`.

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
- `standalone_html_endpoint_not_allowed`
- `standalone_html_egress_disabled`
- `standalone_html_provider_response_invalid`
- `standalone_html_provider_response_too_large`
- `standalone_html_output_too_large`
- `standalone_html_invalid_document`
- `standalone_html_storage_limit`
- `standalone_html_unsupported_encoding`
- `standalone_html_validator_busy`
- `standalone_html_validator_timeout`
- `standalone_html_validation_budget_exceeded`
- `standalone_html_response_invalid`
- `json_structure_too_complex`
- `validator_unavailable`
- `content_kind_immutable`
- `content_kind_not_accepted`
- `standalone_html_creation_requires_generation`
- `operation_not_supported_for_content_kind`
- `presentation_version_conflict`
- `generation_idempotency_key_required`
- `generation_idempotency_conflict`
- `generation_configuration_changed`
- `generation_receipt_unresolved`
- `generation_correlation_mismatch`
- `generation_digest_key_unavailable`
- `generation_reconciler_overloaded`
- `generation_cancelled`
- `generation_expired`
- `generation_quarantined`
- `presentation_version_not_found`

Use `400` for a missing/invalid idempotency or content-negotiation header; `406` for a valid request whose target content kind was not accepted; `413` for receive-time and semantic byte/token ceilings; `415` for unsupported request media/content encoding; `422` for malformed documents, validation-budget refusal, invalid options, or rejected Jobs payload construction; `409` for kind/operation/idempotency/configuration-revision/endpoint conflicts; `412` for stale ETags; fixed redacted `500` for source-bearing response validation/serialization failure; and `503` plus bounded `Retry-After` for validator saturation/unavailability, digest-key/reconciler unavailability, an unresolved ambiguous receipt inside its 15-minute recovery window, or a Jobs-store outage. Provider response/output failures use the stored Jobs/receipt failure representation rather than turning the original accepted POST into a synchronous `413`. `generation_expired` is a stored terminal job/receipt failure, not a transport retry instruction.

Diagnostics identify bounded fields, limits, and machine-readable codes. They do not echo source documents, prompt bodies, API keys, model output, notes, or JavaScript into logs or error responses.

## Compatibility

- Existing request omission follows the current structured path exactly.
- Existing per-source generation endpoints keep their response types and behavior.
- Legacy database rows and version snapshots default to `structured_slides`.
- Opted-in list clients receive the additive `content_kind` field; legacy clients that omit content-kind negotiation see the existing structured-only result set, and HTML source stays out of all summaries.
- Frontend normalization preserves both discriminated variants and never converts unknown/missing HTML into `slides: []` for mutation.
- Old servers are detected through explicit capabilities; the HTML form never assumes support from a generic Slides route.
- JSON export preserves the discriminator and active payload and uses attachment/no-store handling when it contains HTML. Presentation import remains out of scope for v1.
- Existing structured weak ETags remain unchanged; strong ETags are scoped to standalone HTML responses.
- Structured Reveal, PDF, and render workers retain their existing behavior, with an added kind check.
- The browser extension can list or hand off HTML projects but never requests, renders, or executes their source.

## Observability And Operations

Record only safe metadata:

- generation mode and source kind;
- provider/model/adapter and normalized endpoint identity after secret removal;
- job duration and terminal state;
- document byte and slide counts;
- validation/error code.

Never log source material, source snapshots, full HTML, JavaScript, notes, prompts, API keys, endpoint query/userinfo, receipt HMACs, download bodies, or raw-source request/response bodies. Receipt/HMAC metadata remains access-controlled internal state rather than being classified as freely loggable “safe metadata.” Existing auth, per-user isolation, rate limiting, and Jobs ownership checks apply.

Standalone HTML **generation** is disabled by default until configured. Startup validates the complete default provider/model/adapter/endpoint tuple against the standalone allowlist, rejects custom/overridden/fallback endpoints, validates the bounded HMAC keyring, and loads the packaged prompt asset. Invalid generation configuration keeps structured generation and standalone HTML read/edit/export available, advertises the HTML generation mode as disabled with a safe reason code, and logs a safe administrator-facing reason.

`presentation.generate` is registered explicitly in the Slides Jobs worker and in any job-type allowlist. In-process startup/shutdown follows the existing Jobs worker lifecycle; external-worker deployments receive an equivalent documented registration command/configuration. Startup fails the generation capability closed if the handler or required validator dependencies are unavailable.

The lightweight domain reconciler checks active Jobs owners at startup and at least once per minute and completes the bounded dormant-database sweep at least every 15 minutes. It repairs unbound receipts by stored Jobs key, applies the completed-presentation precedence rule, maps eligible terminal active/archive Jobs state into nonterminal receipts, logically deletes terminal inputs, enforces overdue 24-hour logical access deadlines, and requests the exact Slides-scoped JobManager CAS transition when Jobs is reachable. Public status invokes the same idempotent reconciliation synchronously. Cleanup removes expired terminal receipt metadata without deleting committed presentations. Archive lookup uses both the composite scope index and unique immutable-UUID index; archive mutation/compression never keys on numeric ID. Metrics contain bounded latency/error/state values without keys, HMACs, source, prompts, endpoints, or payloads.

## Testing Strategy

### Backend unit and property tests

- strict duplicate-key/nonfinite JSON decoding; request/content-kind enum mapping and cross-field invariants, including lone-surrogate rejection in every client string before UTF-8/HMAC/token work;
- structured partial-update merge-before-validation behavior plus generic rejection of HTML source fields;
- immutable kind and structured-only operation guards;
- outer-fence removal and deterministic prompt assembly;
- every presentation type, visual direction, delivery style, and boundary slide count; generation-time note-style enforcement plus save/restore acceptance and safe isolation of zero/one edited `.notes` node independent of original style;
- direct `html5lib`/`tinycss2` clean-install imports; preparse HTML token/tag/attribute/apparent-depth/single-token refusal; counting tree-builder abort before node/attribute/actual-depth overflow; strict parse-error behavior; independent CSS byte/token/declaration/block/function/error budgets before and after `tinycss2`; split priority/reservation saturation and fairness; a hung killable subprocess is terminated/replaced while a deterministic budget refusal is distinct; malformed/truncated documents; exact one-classic-script placement; all URL/CSS/data-URI/style/event-attribute rejection; namespace-aware SVG attributes; control-character and Unicode title normalization; iterative semantic search extraction; and digest calculation;
- generated source never appears in logs or error messages;
- property/fuzz cases for parser limits and invariant preservation;
- exact no-op saves, metadata-only revisions with unchanged HTML, compact `ensure_ascii=False` snapshot serialization/byte ceilings, and version retention.

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
- receipt/input atomic claim, canonical-manifest/HMAC domain separation, bounded key rotation/retirement registry, long-downtime deterministic expiry, sweep-proven key retirement, and global POST/worker fail-closed behavior when any required secret is absent; stored derived Jobs key; immutable-UUID binding with numeric IDs treated only as paired hints; exact prompt/source/adapter/endpoint snapshots; 24-hour logical input expiry; durable presentation provenance before source-snapshot cleanup; terminal replay; fenced single-leader dormant-database reconciliation with two-process/crash takeover; and terminal cleanup;
- metadata-only HTML delete/version-list projections prove source/payload columns are not selected or deserialized;
- SQLite/PostgreSQL archive scope and unique-UUID index creation, duplicate-UUID migration refusal, indexed active/archive lookup, and UUID-only archive compression/mutation behavior.

### API, Jobs, and MCP integration tests

- successful HTML generation from prompt, chat, media, notes, and retrieval-only RAG with exactly one mocked allowlisted completion call per normal worker attempt;
- complete provider/model/adapter/endpoint canonicalization and allowlisting; normal verified HTTPS; exact literal-loopback-only HTTP; rejection of remote HTTP, TLS verification disablement, `localhost`/LAN/link-local targets, and custom/overridden/proxy/fallback targets; queued/retried jobs recheck feature/egress kill controls immediately pre-call and flag-off produces zero provider calls; exact capability schema/effective limits; matching and stale `generation_config_revision`; ownership, size, timeout, and idempotency behavior;
- submission-time immutable source snapshots for all source kinds and proof that workers do not reread changed sources;
- each chat/media/notes/RAG adapter stops at `max_source_chars + 1` without materializing or persisting the remainder, and oversized fixtures never reach token counting or a provider;
- the closed RAG source profile cannot enable generation, HyDE, LLM/VLM rewriting/expansion, clarification, LLM reranking, verification, synthesis, `generated_answer` fallback, web/discussion search, URL scraping, image/video search, or any other external egress through caller/stored settings;
- the receipt-only Jobs payload uses `queue=default`, survives existing secret normalization byte-for-byte, and maps oversized/redacted/nonexact envelopes to 413/422 without a provider call, including with `JOBS_JSON_TRUNCATE=true` and a deliberately tiny limit;
- stored client-request HMACs are validated but never impossibly rederived from execution input; incoming replay recomputes from its own canonical request; execution HMACs recompute from the authoritative client HMAC plus verified source/target/prompt; key rotation/missing-key behavior and same-key replay use the stored tuple/prompt;
- same-key exact closed replay variants/status codes, same-key/different-client-HMAC conflict, cross-owner isolation, pre-admission indexed active/archive lookup, API-first and worker-first immutable-UUID bind recovery, confirmed unbound miss after 15 minutes, receipt/Jobs correlation mismatch failure, and rejection of reused numeric IDs;
- crash recovery after presentation commit but before Jobs completion returns the verified presentation without input/provider access; final Jobs state/lease/cancel recheck; cancellation-before-commit and cancellation-after-check races; immutable completed-receipt precedence; retryable failure followed by success with retained input; exhausted/nonretryable cleanup; 24-hour expiry while Jobs is unavailable; dormant-database sweep overload/startup ordering; late-worker result discard; and 30-day receipt replay/one-presentation semantics independent of Jobs pruning;
- validator capacity is reserved before provider dispatch; local validator saturation causes no provider call; provider calls may repeat after a retryable provider/transport or other precommit infrastructure failure, or a process crash before commit, while every retry after a verified presentation commit skips the provider;
- provider success and non-2xx responses request/require identity encoding, reject gzip/br/deflate/conflicting encodings before body read, and enforce the raw 8 MiB stream cap for declared, missing, chunked, and dishonest lengths; the generic eager error-body hook is bypassed; 4xx bodies never reach logs/exceptions; redirects and proxy environment variables are ignored; duplicate-key/nonfinite/lone-surrogate provider JSON fails safely; extracted HTML still enforces 1 MiB;
- queued/processing/completed/failed/cancelled/quarantined public status mapping;
- `WorkerTerminalOutcome` failed/cancelled paths use expected-state/lease/UUID terminalization and never call Jobs completion; normal bounded result remains the sole completion path;
- failed validation creates no presentation;
- exact raw `PUT /presentations/{id}/html-source` media type/body/ETag/error/response behavior; generic JSON PATCH source rejection; invalid standalone saves and restores are atomic and create no revision; corrupt/invalid stored standalone content is rejected before both `format=html` and JSON export, while an in-bounds malformed draft remains downloadable through `draft-attachment`;
- generic REST/MCP create cannot construct a standalone HTML record; the constant-memory schema-path prefilter stops before rejected HTML values and the shared sanitizer keeps sentinels out of default validation, logs, traces, and errors; legal large structured payloads (including multiple maximum-size inline images and existing MCP slide strings) retain current transport behavior; the generation provenance closed schema/4,096-byte cap is enforced;
- job result contains metadata rather than HTML;
- omitted/structured-only content-kind negotiation filters HTML before list/search pagination and returns 406 on direct HTML targets, while opted-in current clients receive the discriminated union and every varying response carries the negotiation `Vary` token;
- receive-time rejection for duplicate/conflicting/invalid/oversized declared, dishonest, missing, and chunked `Content-Length`, stopping at the first over-limit chunk; absent/identity-only `Content-Encoding`; fixed 4 MiB generation and 1 MiB raw-source routes; worst-case escaped generation JSON; and semantic decoded-source limits;
- malformed JSON, lone surrogates across source/selectors/options, attacker-controlled unknown field names, Pydantic, and custom-validator sentinel failures prove raw locations/source never appear in validation responses, application/proxy logs, client errors, or error capture;
- deliberately broken detail/save/restore/version-content/export response unions prove `ResponseValidationError`, serialization exceptions, logs, traces, and error capture never contain sentinel `html_document`;
- every non-attachment raw-source response is `application/json` plus `nosniff`, is `private, no-store`, varies for auth, never negotiates `text/html`, and is excluded from body logging/error capture;
- a genuinely separate-origin WebUI/API browser flow can preflight `Idempotency-Key`/`If-Match`, read `Content-Disposition`/`ETag`/`Retry-After`, save and adopt an ETag, and download only after header verification through both normal and drain/error CORS paths;
- saved HTML remains readable/editable/exportable while only generation is disabled; validator-unavailable mode keeps inert read/draft recovery but fails save/restore/saved export closed with matching capabilities;
- all structured-only operations reject HTML before dispatch;
- render workers independently recheck content kind;
- metadata-only HTML DELETE, database-level REST/MCP version-list projections, saved export, and exact draft-attachment route/body/error/header behavior;
- every MCP tool follows the explicit HTML matrix, including metadata-only version lists and rejection of version-content/restore and every export format including JSON;
- omission of a new mode field on the existing per-source endpoints preserves current structured behavior; the new `/slides/generations` route always requires `standalone_html`.

### Frontend tests

- capability loading, Retry, unsupported-server, generation-disabled, validator-unavailable read/recovery-only, and fully supported states;
- minimal paginated index loading/empty/error/retry states, Load more, New/open actions, and HTML kind metadata;
- form validation (including unpaired-surrogate rejection and native spellcheck/autofill/persistence opt-out), effective-limit/config-revision handling, complete server-default target display/reconfirmation, payload mapping, immediate pre-POST immutable snapshot/field lock, duplicate-submit prevention, trusted-principal-scoped 24-hour form-draft/recovery expiry, reload-before-`202` same-key replay, Resume/Stop waiting/Forget/new-request semantics, creation-page logout/account-switch/pagehide/pageshow/bfcache disposal and guarded restore, storage failures, polling failures, and job completion handoff;
- the WebUI/extension send content-kind negotiation on every applicable REST call; discriminated client normalization, legacy omission behavior, and unknown-kind fallback;
- code/outline responsive layout and absence of structured controls;
- Monaco and fallback use labelled non-form text inputs with spellcheck/autocorrect/autocapitalize/autocomplete and supported password-manager persistence disabled; Monaco uses the inert-text model with `links: false`, no HTML document-link/hover provider, and a rejecting scoped opener without changing unrelated editors;
- fixed hard-cap, U+0000, and unpaired-surrogate edit/paste/recovery rejection in Monaco and fallback before `TextEncoder`; exact UTF-8 bytes round-trip through recovery, raw save, and draft download; trusted-principal-scoped/expiring recovery records with Restore/Download/confirmed Discard; exclusion of HTML from global caches/devtools/error capture; last-keystroke synchronous pagehide flush followed by Monaco/outline disposal, plus logout/scope disposal and guarded pageshow/visibility rehydration; session quota/account-switch behavior; explicit destructive conflict labels; raw-save dirty/save/error/conflict behavior; and digest/identity-only unknown-save reconciliation that adopts the server title/ETag;
- safe-outline extraction is latest-wins under rapid edits with exactly one active plus one replaceable pending snapshot and bounded retained source copies; the static parser worker preflights total tokens/nodes, tags, attributes, comments/declarations, text transitions, depth, and single-token length before Cheerio; comment/bogus-markup/text storms refuse safely; a hung worker is terminated/replaced; only a digest-bound capped plain-text DTO crosses to the main thread; iterative traversal/output budgets, Unicode isolation, markup/CSS/script/asset discard, separate safe notes, and unchanged source are verified;
- authenticated download validates status/MIME/attachment headers, transfers exact bytes through only the fixed-name temporary octet-stream Blob anchor, and always removes/revokes it;
- keyboard access, labels, focus behavior, `aria-live`, reduced motion, 44px touch targets, and focus-preserving Code/Outline tabs at narrow viewports.

### Safe-outline security and browser tests

- sentinel JavaScript never runs during generation handoff, open, edit, outline refresh, save, restore, reload, Back/Forward, or download in Chromium, Firefox, and WebKit;
- multi-user logout → login-as-another-user → Back/Forward/bfcache and visibility-restoration scenarios cover creation, submitted-state, and workspace pages; they show an empty guarded shell and never expose the prior form source, immutable request, Monaco buffer, outline AST, response object, or recovery source before principal/age revalidation;
- no model DOM node, attribute, CSS, script, SVG, MathML, template, link, or asset survives into the component-rendered outline;
- the feature creates no iframe, `srcdoc`, source-derived data URL, popup, generated-HTML route, `dangerouslySetInnerHTML`, `DOMParser`, `innerHTML`, or `insertAdjacentHTML` path from source; no worker code/URL/module/function is sourced from or evaluates generated content, while trusted Monaco/parser workers remain permitted for inert text;
- no Blob URL is created except during an explicit successful attachment download; it contains `application/octet-stream`, reaches only the temporary fixed-name download anchor, is revoked after use, and never reaches a rendering, navigation, popup, worker, or preview sink;
- malicious `href`/`src`, CSS URL, and plain-URL source remains ordinary editor text: hover, context menu, Enter, and Cmd/Ctrl-click cause no opener call, navigation, popup, or network request;
- parsing valid, invalid, and adversarial drafts emits no subresource or navigation request and never calls a backend validation endpoint;
- direct API requests and browser navigation to every source-bearing route receive only the contracted JSON/octet-stream MIME plus `nosniff`; none renders HTML or executes a sentinel;
- the ordinary application CSP remains unchanged with no new `unsafe-inline`, `unsafe-eval`, or `frame-src` allowance;
- malformed/over-budget outline input shows bounded trusted UI, leaves source unchanged, and cannot replace a newer outline result;
- comment/bogus-token/text-run storms and a deliberately hung static parser worker never block the main thread, leak AST/source in error DTOs, or survive watchdog replacement;
- malicious titles cannot inject response headers; HTML/draft attachments use fixed `presentation.html`, while JSON export uses fixed `presentation.json` with its contracted MIME;
- a keyboard-focused desktop/mobile E2E covers generate, edit, safe outline, save, conflict recovery, reopen, and download without horizontal page overflow.
- a same-principal **type, immediately navigate, Back/Forward** E2E restores the last accepted form/editor keystroke from the bounded pagehide flush, while expired or other-principal records never restore.

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
