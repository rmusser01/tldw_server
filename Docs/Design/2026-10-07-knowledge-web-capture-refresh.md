# Explicit web capture and refresh in Research

**Status:** Approved by requester on 2026-10-07; implemented and locally verified. Controller review/publication gates remain.

**Date:** 2026-10-07

**Task:** TASK-13530.1, part of TASK-13530

**Investigated baseline:** `dev` at `26ae4fd679`

**Decision record:** [Accepted ADR-066](../ADR/066-explicit-web-capture-and-refresh-snapshots.md)

## Outcome and scope

An external-web result continued from Knowledge remains a retrieved excerpt until the user chooses **Capture article**. Research fetches public readable text through the existing scraper, shows the actual extraction, and saves only after **Save capture**. Saving reuses WebClipper: it creates a capture Note and a dedicated Media-backed workspace source. **Refresh capture** repeats preview and acceptance with a fresh capture UUID. Original excerpts, answer qualifications, capture Notes, and previously accepted source versions remain available; a refresh does not overwrite them.

The first slice captures extracted text, not a faithful copy of a whole website, authenticated browser session, PDF download, or remote publisher revision. Existing extension capture remains available for user-selected or browser-extracted content. No automatic fetch on import, selection, preview, reopen, or Ask; no new crawler, persistence subsystem, dependency, queue, or endpoint.

## Existing behavior and patterns to reuse

| Existing pattern | Reuse and necessary qualification |
| --- | --- |
| Knowledge Notes continuation in `use-research-workspace-prefill.ts` | Freeze owner/workspace request scope, verify canonical identity, checkpoint accepted media before attaching, and retain all original evidence references. External web currently uploads only the retrieved excerpt. |
| WebClipper save and extension preview | One accepted body plus a stable owner-scoped `clip_id`; canonical Note identity is separately mapped. `full_extract` is promoted into Media even when the visible Note body is shortened. Preserve the extension's actual selection/article/fallback labels. |
| Workspace source preview and Media versions | Workspace membership gates preview; Media provides exact active version reads. Current preview reads latest content and current chunks, so a new optional version pin must prevent mixed-version evidence. |

`POST /api/v1/media/ingest-web-content` with `individual` returns fresh extraction without persisting it. It is suitable for preview, but currently accepts configured scraper credentials and discards individual extraction failure reasons. `process-web-scraping` is unsuitable for refresh: persistent saves use URL deduplication with `overwrite=False`, which can skip changed pages. `TldwMedia.processUrl` also sends a body inconsistent with this extraction API; use the correct request contract.

## User flow

1. A web result shows **Retrieved excerpt**, its original URL, and **Capture article**. Activating the action starts exactly one scoped extraction. A capture records a new retrieval time; it does not strengthen the support claimed for the earlier answer.
2. Preview shows URL, extracted title, retrieval time, extracted character count, and **Extracted article snapshot**. The display can be shortened, with a clear count and expand action; the accepted extraction cannot be silently shortened. Empty text, blocked transport, extraction failure, or text above WebClipper's 1,000,000-character bound cannot be accepted. Render untrusted text as text, never execute returned HTML.
3. **Save capture** freezes the trimmed UTF-8 text, descriptor, target workspace, capture UUID, and exact WebClipper body. It saves a capture Note and promotes a workspace source, then verifies the persisted source and exact Media version. Success is shown only after that readback. Partial promotion remains visible and retryable with the same UUID and body.
4. The source preview shows capture time and **Source snapshot: Media version N**. If useful, it separately shows **Capture Note revision M**. Neither number denotes a version published by the website. Original retrieved excerpts remain available as their own evidence.
5. **Refresh capture** fetches into a new preview. Accepting creates another UUID, Note, Media identity, and source, even if the extraction is unchanged; show **Text unchanged** when hashes match. Keep the previous source available. The user's chosen active source may move to the new capture only after readback, without replacing an intervening manual selection.

Saving an extra Note is an accepted reuse cost and must be disclosed in the preview's destination text. Canceling an extraction preview does not save a Note or source. Existing storage/indexing work after acceptance stays with WebClipper and Workspace Jobs.

## Identity, version, and evidence contract

Three identities remain separate: the original retrieval reference, the WebClipper capture UUID, and the canonical Media ID/version. The WebClipper Note UUID and its independently changing revision are a fourth identity; `save.note.version` is never a Media version.

Use the existing bounded `capture_metadata` dictionary for one reserved descriptor:

```json
{
  "web_capture_v1": {
    "mode": "server_article",
    "requested_url": "https://example.org/article",
    "captured_at": "2026-10-07T18:00:00Z",
    "content_sha256": "<SHA-256 of the exact trimmed UTF-8 extraction>",
    "refresh_of": "<previous clip_id, or null>"
  }
}
```

The server validates the reserved descriptor, recomputes the content digest from the promoted `full_extract`, and rejects disagreement. The descriptor describes an accepted client capture, not attested website authorship. Retain only bounded fields; credentials and raw response headers are excluded. Do not invent a final URL or publisher version when the scraper does not supply one. Existing WebClipper copies this dictionary into version `safe_metadata.capture_metadata`; its 65,536-character metadata bound already applies.

After save, read `GET /api/v1/workspaces/{workspace_id}/sources`, locate exactly `web-clipper:{clip_id}`, and verify the workspace, URL, and Media ID. Use existing Media version list/read APIs to identify an active version whose `safe_metadata.clip_id`, descriptor, and full-content digest match the frozen acceptance. Checkpoint its actual `(media_id, version_number, version_uuid)`; do not assume version 1 or infer it from a Note response. Exact retries may encounter an already accepted version. Once pinned, use that exact version; never silently fall back to latest. Readback failure leaves a recoverable accepted-but-unconfirmed state.

The strict `notes.provenance` v1 contract forbids new descriptor fields. Original Knowledge provenance stays unchanged during capture/refresh. When a later sourced Note explicitly references a capture, existing source fields suffice: `originalId=clip_id`, `sourceType="web_capture"`, `snapshotMediaId=media_id`, `originalVersion=Media version_number`, `type="website"`, and the original URL. Define and display that version as a local snapshot version. Keep original retrieved references alongside it; no richer Notes wire schema or new Sync domain. Removed provenance is not implicitly restored by capturing a page.

## Minimal interface changes

| Interface | Proposed change |
| --- | --- |
| `IngestWebContentRequest` / existing `POST /media/ingest-web-content` | Add optional `credential_free: bool = False`. For this profile require exactly one HTTP(S) URL, `individual`, no URL userinfo, no caller cookies, analysis/translation/LLM extraction/chunking disabled. Reject conflicting options. Preserve existing quota/governance behavior, add expected-owner and shared Media-create authorization/rate-limit dependencies, and make the unused required `token` header optional/deprecated; authenticated principal remains authoritative. |
| Extraction response | Keep the existing success/results envelope. For the opt-in profile preserve bounded structured failure reasons instead of dropping them into an empty success list; produce an unambiguous unsuccessful preview for policy, transport, timeout, extraction, and size failures. Timestamp in UTC. Preview skips topic-monitoring alert scheduling; existing usage accounting remains. |
| `scrape_article` and existing article preparation/policy hooks | Add keyword-only `credential_free=False`. Apply its clean request plan before policy, preflight, or acquisition, and retain it through recommendations, retries, redirects, and browser fallback. Reuse the existing `plan_modifier` and injected guards, not a second HTTP stack. |
| Existing WebClipper save request/service | No new request fields or save endpoint. Use a fresh UUID, `destination_mode="workspace"`, existing workspace payload with `default_review_state="needs_review"`, accepted text in `full_extract`, enhancements off, and the descriptor above. Validate the reserved descriptor only when present. Preserve existing callers and ordinary clip metadata. |
| `GET /workspaces/{workspace_id}/sources/{source_id}/preview` | Add optional positive `version_number`; return optional `document_version_number`. After current workspace/source ownership checks, read that exact active version through the Media DB abstraction. A missing/deleted version returns unavailable/404, never latest. Suppress current MediaChunks for pinned preview; the exact version's bounded text excerpt remains available. Existing unpinned previews stay compatible. |
| Shared service clients | Add scoped, abortable extraction and exact-version reads; add optional `ScopedRequestOptions` to `getWebClipStatus`, `getWorkspaceSources`, and `getWorkspaceSourcePreview`. Extend preview params with `version_number`. Use frozen origin, expected owner, and abort signal at every stage. Regenerate checked OpenAPI artifacts/types for additive API changes. |

### Public credential-free transport

Currently `ArticlePlan.from_routing_plan` combines generated browser headers with configured `extra_headers`, and carries route cookies. `_prepare_article` merges caller cookies and then route cookies, so passing `None` or an empty list does not clear configured credentials. The new profile must remove all site-specific extra headers, all caller/route cookies, and browser `custom_cookies`; regenerate only canonical browser negotiation headers. It must not merely blacklist `Authorization` and miss custom credential headers.

Canonical bounded HTTP acquisition already uses fresh clients; guarded browser acquisition creates a fresh context without persisted storage. Retain those paths and clear cookies before creating either context. Keep existing limits, robots behavior, allow/deny rules, proxy/transport admission, and ADR-042 attestation. A denial cannot trigger an unguarded fallback. Reject URL userinfo and require public-address egress checks for initial targets, probes, robots fetches, each redirect, and browser subresources. Use existing Security checks and request-scoped guard injection; if a deployment cannot enforce this stricter public profile, fail closed rather than changing global policy or using the configured-local-LLM exception. Do not attach the user's API credentials to the target site.

## Ownership, retries, retirement, and stale state

Freeze the authenticated owner and workspace before extraction. Revalidate current authority and exact source membership before save, readback, attachment, and any sourced-note mutation. Server per-user DB/RLS checks remain authoritative. An account/origin/workspace switch, source deletion, lost permission, or retired component cancels pending work; late responses cannot attach to a new owner or workspace. Accepted server work is checkpointed under its original owner even if its UI retired, so recovery does not start another capture.

One acceptance has one frozen UUID/body. A lost response retries or queries that same identity; changed text/target is a new acceptance, not a mutation of the retry body. Existing WebClipper Sync receipts and inactive Notes transactions own persistence. Partial source promotion does not justify deleting the Note or creating another UUID. A deliberate refresh always gets a new UUID and leaves old evidence intact. Avoid repeat submissions while an acceptance is pending.

Existing RAG scopes by Media ID and current chunks; it has no historical-version retrieval contract. Fresh Media identities keep ordinary refreshes from changing prior retrieval material. A capture whose Media head later changes must show **Snapshot changed outside refresh** and be excluded from the captured-source Ask selection until the user accepts a fresh capture; do not claim historical preview implies historical RAG. Recheck the current owned head before sending scoped Ask, retain normal server authorization, and state that this first slice does not provide a server-atomic version-aware RAG guarantee. Full historical-version RAG would be separate approved work.

## Implementation files and verification

Expected backend scope:

- `api/v1/schemas/media_request_models.py`, `api/v1/endpoints/media/ingest_web_content.py`, and `services/web_scraping_service.py` for extraction profile/admission/errors.
- `core/Web_Scraping/orchestration/article.py`, `article_models.py`, and existing `policy/`/`preflight/` seams only where profile propagation is required. `core/Security/egress.py` remains the central evaluator.
- `core/WebClipper/schemas.py` and `service.py` for optional reserved-descriptor validation and digest verification.
- `api/v1/schemas/workspace_schemas.py`, `api/v1/endpoints/workspaces.py`, and `core/Workspaces/source_preview.py` for pinned bounded preview.

Expected shared UI scope:

- `services/tldw/TldwMedia.ts`, `domains/web-clipper.ts`, `domains/workspace-api.ts`, and existing WebClipper types for correct scoped calls.
- `components/Option/ResearchWorkspace/SourcesPane/index.tsx` and `ResearchWorkspace/index.tsx` for explicit actions, preview/acceptance, recovery, and labels.
- `utils/research-workspace-prefill.ts`, `use-research-workspace-prefill.ts`, `knowledge-note-provenance.ts`, and `ResearchWorkspace/workspace-server-restore.ts` only for capture references/checkpoints and type-aware labels. Preserve the Notes v1 schema. Share behavior between WebUI and extension; no extension menu rewrite is required.

| Behavior gate | Existing test homes to extend |
| --- | --- |
| Configured/caller cookies and arbitrary credential headers absent on every acquisition path; policy/robots/private redirect/browser denial remains enforced; analysis absent; empty/oversized failures bounded | `tests/Web_Scraping/test_phase4_article_orchestration.py`, `test_phase4_article_models.py`, `test_phase4_article_browser.py`, existing outbound-policy tests, `tests/Media/test_ingest_web_content_endpoint_sanitization.py` |
| Text beyond the visible Note budget survives in Media; descriptor/hash validation; UUID retries do not fork Notes/sources; refreshed UUID retains old capture; partial promotion recoverable; current owner and deleted workspace enforced | `tests/Notes_NEW/unit/test_web_clipper_service.py`, `integration/test_web_clipper_api.py`, `integration/test_web_clipper_sync_contract.py`, `tests/ChaChaNotesDB/test_web_clipper_db.py`, `test_web_clipper_postgres_tenancy.py` |
| Exact old version preview after newer head; deleted/missing pin unavailable; latest chunk evidence excluded; source membership and per-owner reads enforced | `tests/Workspaces/test_workspace_source_preview.py`, `test_workspace_source_preview_context_api.py`, existing Media version read tests |
| No network until explicit action; preview cancel no save; unchanged/changed refresh preserves evidence; frozen partial retry; owner/workspace/manual-selection retirement; readback distinguishes Note revision from Media version; stale capture Ask fence; reopen/export retains references | `ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx`, shared `utils/__tests__/research-workspace-import.test.tsx`, `research-workspace-prefill.test.ts`, `knowledge-note-provenance.test.ts`, plus a narrow capture workflow test |

Use real SQLite and the existing PostgreSQL tenancy fixtures for persistence boundaries; mock outbound sites. Run relevant backend/UI tests, TypeScript checks, both client builds, OpenAPI drift checks, touched-scope lint and Bandit. Implementation and local verification are recorded in [the canonical closeout](../Reviews/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.md), including mocked acquisition versus real persistence and remaining external qualifications.

## Alternatives and ADR assessment

| Alternative | Tradeoff |
| --- | --- |
| **Recommended: extraction preview → existing WebClipper save** | Small additive contracts, existing receipts and workspace promotion. Creates an extra Note and needs separate Media-version readback. |
| Use persistent enhanced web scraping directly | Reuses crawler controls but URL/content dedup can retain old content, and persistence precedes acceptance. Fixing its full ingestion contract broadens this task. |
| Add a capture/version database, endpoint, or authenticated-browser session bridge | Could model richer capture lineage but adds authority, migration, retention, transport, and lifecycle rules unnecessary for readable text. |
| Keep excerpts and direct users to extension capture only | Lowest change cost but leaves explicit in-Research public capture/refresh incomplete. |

**ADR required: yes. ADR path:** [accepted ADR-066](../ADR/066-explicit-web-capture-and-refresh-snapshots.md). Explicit fetch/accept boundaries, credential-free acquisition, and refresh identity/version semantics are durable public API, security, and persistence rules. This decision composes [ADR-007](../ADR/007-research-workspace-canonical-first-slice-shell.md), [ADR-018](../ADR/018-resource-governance-endpoint-policy-and-route-map.md), [ADR-026](../ADR/026-security-outbound-egress-and-ssrf-policy.md), [ADR-031](../ADR/031-notes-capability-sync-domains.md), [ADR-034](../ADR/034-durable-server-origin-sync-mutation-batches.md), [ADR-036](../ADR/036-web-clipper-external-identity-mapping.md), [ADR-042](../ADR/042-browser-transport-admission-and-attestation.md), and [ADR-065](../ADR/065-independent-notes-knowledge-provenance.md); no accepted rationale is rewritten or superseded.

The requester approved this behavior and ADR on 2026-10-07: readable public text only, an extra capture Note, a new source identity on every accepted refresh, preservation of old evidence, and the boundary around historical-version RAG. Local implementation acceptance is recorded in the canonical closeout; independent branch review, publication, a new human Change summary, updated-head CI and merge remain controller gates.
