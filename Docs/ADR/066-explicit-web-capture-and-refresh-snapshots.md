# ADR-066: Explicit web capture and refresh snapshots

**Status:** Accepted

**Date:** 2026-10-07

**Backfilled from:** not backfilled

**Decision owner:** Requester; design approved 2026-10-07

**Related task:** TASK-13530.1

**Related spec/plan:** [Requester-approved Research capture design (historical proposal source)](https://github.com/rmusser01/tldw_server/blob/746598afff328fd0585677003bd916b7471a106f/Docs/Design/2026-10-07-knowledge-web-capture-refresh.md)

## Decision

Require explicit preview and acceptance for public credential-free web capture, reuse WebClipper's owner-scoped Note/Media persistence, and give every accepted refresh a fresh capture identity pinned to an existing Media document version while retaining earlier evidence.

## Context

Knowledge web search supplies retrieved excerpts. Continuing them into Research currently saves excerpt copies, whereas Notes continuation reads an owned canonical Note and saves a full versioned text snapshot. Treating a URL, current web result, or subsequently fetched article as the original answer's evidence would obscure what supported that answer.

The existing individual extraction endpoint can fetch without persisting, and WebClipper can save accepted extracted text as a capture Note plus a dedicated Media-backed workspace source. However, configured route cookies and custom headers still reach the ordinary scraper even when a caller passes no cookies. Current Workspace preview reads latest Media content and chunks. Neither endpoint nor a Note revision alone supplies a trustworthy snapshot pin.

[ADR-065](065-independent-notes-knowledge-provenance.md) deliberately does not establish web capture/refresh policy. Its strict Notes capability already retains historical references and is not extended for capture descriptors. [ADR-036](036-web-clipper-external-identity-mapping.md) distinguishes public capture identity from canonical Note UUIDs. [ADR-026](026-security-outbound-egress-and-ssrf-policy.md) and [ADR-042](042-browser-transport-admission-and-attestation.md) continue to govern all acquisition paths.

## Alternatives considered

| Option | Why not selected |
| --- | --- |
| Persistent enhanced web scraping as the capture operation | URL/content dedup can retain old source content; saving before acceptance also changes the preview contract. |
| Overwrite one URL-keyed Media item on refresh | Existing answers and chunks can silently move to newer text; a latest-content preview cannot explain the original evidence. |
| New capture store, version API, or browser-session bridge | Adds persistence and authority boundaries that existing extraction, WebClipper, and Media versions can already serve for public readable text. |
| Automatically fetch when web evidence enters Research | Creates hidden network/persistence work and makes the original retrieval and later capture difficult to distinguish. |
| Extension-only capture | Remains useful for user-selected/browser content, but does not complete the requested in-Research public capture/refresh flow. |

## Consequences

No new endpoint, dependency, Sync domain, or persistence subsystem is required. A small opt-in extraction profile removes caller/route cookies and all site-specific credential headers before probes or acquisition, disables analysis, and remains subject to existing quotas, robots checks, central egress, and browser transport admission. All concrete targets must pass public-address checks; inability to enforce the profile fails closed. No policy denial permits an unguarded fallback or a global SSRF relaxation.

The user reviews actual extracted text and explicitly accepts one frozen save body. Existing WebClipper save creates an additional capture Note; that visible product cost is disclosed. A bounded descriptor lives in existing capture metadata/Media safe metadata, with the accepted text digest verified on save. It does not claim publisher authenticity, completeness of a whole website, or a remote publisher revision.

Every acceptance owns one UUID and immutable retry body. Recovery reuses that identity and existing WebClipper receipts; a deliberate refresh gets a new UUID, Note, Media identity, and workspace source. The earlier extraction and original retrieved excerpts remain intact unless separately deleted by their authorized owner. Capture does not implicitly restore removed Notes provenance or retroactively change answer support.

Readback must resolve the exact promoted source and an active matching Media version. A capture Note revision is not a source version. Pinned source preview reads exact version text and excludes current-version chunk snippets; an unavailable historical version never falls back to latest. Existing Notes provenance can reference captures with its current source fields, with a local snapshot version explicitly distinguished from a Note revision.

Every asynchronous stage freezes and revalidates owner/workspace authority. Retired work cannot attach to a different account or workspace. Partial persistence stays recoverable under the accepted identity instead of creating another capture or compensating deletion. A provenance reference conveys history, not access permission.

Current RAG targets Media IDs/current chunks rather than historical document versions. New identities isolate normal refreshes; externally changed capture heads are surfaced and excluded from captured-source Ask until recaptured. This decision does not promise server-atomic historical-version RAG. That would require separately approved work.

[ADR-007](007-research-workspace-canonical-first-slice-shell.md), [ADR-018](018-resource-governance-endpoint-policy-and-route-map.md), [ADR-031](031-notes-capability-sync-domains.md), and [ADR-034](034-durable-server-origin-sync-mutation-batches.md) continue to govern shell reuse, ingress authority, strict capability contracts, and durable persistence. No existing ADR is superseded.

## Follow-up

Requester approval was recorded on 2026-10-07. TASK-13530.1 implemented and locally verified the additive profile, descriptor, exact-version readback, owner/retry fences and retained evidence behavior. The historical immutable spec link preserves the approved proposal; the current canonical design records implementation. Local verification and remaining remote-acquisition/current-head-RAG/native-device qualifications are recorded in Docs/Reviews/KNOWLEDGE_CAPTURE_REFRESH_2026_10_07.md. Independent branch review and publication gates remain; acceptance of this durable policy does not claim those gates passed.
