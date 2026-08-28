# Agent-Native Web Research Quality and Provenance Roadmap

**Status:** Proposed after current-dev reconciliation

**Program task:** TASK-13139

**Milestone:** Agent-Native Web Research Quality and Provenance

**Source comparison:** [DonSeTch](https://github.com/dondai44423/donsetch)

**Reconciled base:** `origin/dev` at `24f79419061ba85e9273b38a05431d6fd46ca40f`

## 1. Purpose

Adopt the useful product ideas identified in the DonSeTch comparison without
copying its implementation or creating parallel tldw_server subsystems. The
program improves agent-oriented web reading, deterministic general-search
fusion, governed crawl semantics, retrieval provenance, and Chatbook export.

This roadmap replaces the earlier branch-local planning records that reused
`TASK-12125`. On current dev, `TASK-12125` is the completed Chat Macros planning
task. The branch-local `TASK-12125.*` program is invalid and must not be merged.

## 2. Current-dev baseline

The following capabilities already exist and are foundations, not backlog work
to reimplement:

| Area | Current capability | Program decision |
| --- | --- | --- |
| Browser request governance | Playwright HTTP and WebSocket routes are installed before page creation; service workers are blocked; redirects, subresources, frames, and browser destinations are policy checked. | Preserve and test. Do not create another request-interception task. |
| Browser DNS security | Browser routing validates URLs but does not pin DNS or verify the connected peer. | Add a strict capability gate. Do not claim URL checks solve DNS rebinding. |
| Article escalation | `Web_Scraping.orchestration.article` already attempts governed lightweight retrieval, detects JS-required/thin/no-extract outcomes, falls back to the guarded browser, and records metrics. | Reuse it and expose its trace. Do not add another HTTP-to-browser ladder. |
| General search contracts | Immutable `Web_Scraping.contracts.SearchResult` and `SearchResultsPayload` exist. | Put reusable fusion here, not in scholarly Research discovery models. |
| MCP web retrieval | `web.fetch` already provides bounded URL retrieval, per-hop policy checks, rate limiting, an opt-in process-local TTL/LRU cache, and citation/eval metadata. | Preserve the default contract; add reading views and revalidation incrementally. |
| Media handoff | `/api/v1/media/add`, Media Jobs, per-owner Media storage, and the Phase 2B design task already own durable web ingestion. | TASK-12964 remains authoritative. No second queue, endpoint, or Research-owned ingestion path. |
| Chatbook provenance | Chatbook v1.1 already supports typed `source_refs`, snapshot hashes, resolution status, rehydration hints, and redaction profiles. | Extend existing provenance fields additively; do not add a web-specific content type. |
| Cookie security | TASK-13100 already owns removal of the global plaintext cookie surface. | Use the minimal safe decision below; do not create a speculative cookie-vault task. |

## 3. Goals

1. Let agents read a large fetched document in bounded, deterministic views
   without repeatedly receiving the entire document.
2. Make general multi-provider web search fusion deterministic, explainable,
   versioned, and independent of scholarly source schemas.
3. Preserve the current governed HTTP-to-browser escalation and expose enough
   trace data to explain extraction decisions.
4. Add conditional revalidation for public credentialless MCP fetches without
   creating durable or cross-owner cache state.
5. Add resume and explicit stop semantics to the authoritative Phase 2B crawl
   after its Jobs and cancellation contracts are approved.
6. Persist selected safe retrieval provenance only through Media and export it
   through existing Chatbook provenance contracts.
7. Establish small deterministic quality measurements before adding larger fuzz,
   soak, PDF, or external-comparator programs.

## 4. Non-goals

- Porting DonSeTch source or presenting this comparison as a formal clean-room
  process.
- Adding another web extraction pipeline, durable queue, crawler service,
  Research ingestion endpoint, retrieval database, or Chatbook content type.
- Rebuilding Playwright request interception that current dev already provides.
- Adding authenticated browser automation, a persistent multi-user cookie jar,
  or an encrypted cookie vault in the first release.
- Treating token counts as authoritative or adding a tokenizer dependency.
- Writing exact implementation plans for every future wave before their
  dependencies stabilize.
- Making an external executable, public network, fuzz run, or soak run a default
  pull-request dependency.

## 5. Architecture and ownership

```text
general providers
      |
      v
Web_Scraping SearchResult/SearchResultsPayload
      |
      +--> versioned deterministic fusion --> MCP web.search

requested URL
      |
      v
MCP web.fetch adapter
      |
      v
existing governed Web_Scraping article orchestration
      |                  |
      | HTTP             | guarded browser when policy/capability permits
      +------------------+
      |
      v
bounded reading view + safe retrieval envelope
      |
      +--> ephemeral MCP response/cache only
      |
      `--> optional /api/v1/media/add handoff
                    |
                    v
              Media-owned durable record
                    |
                    v
          Chatbook v1.1 source_refs export
```

Ownership rules:

- `Web_Scraping` owns reusable retrieval, extraction, search-result, crawl, and
  policy primitives.
- MCP owns agent-facing tool arguments, bounded reading views, composition, and
  serializable response envelopes.
- Jobs owns user-visible asynchronous execution, leases, cancellation, retries,
  and checkpoints.
- Media is the only durable owner of selected web content and retrieval
  snapshots.
- Research discovery resolves scholarly/provider candidates and never becomes a
  general web-search schema or ingestion owner.
- Chatbooks exports/imports Media-owned provenance through existing versioned
  contracts.

## 6. Agent reading contract

`web.fetch` remains backward compatible when no new arguments are supplied. New
views are additive:

- `whole`: current extracted output, bounded.
- `toc`: stable section identifiers and bounded headings only.
- `section`: one requested stable section.
- `focus`: bounded content selected around a caller-provided focus phrase.
- `must_contain`: optional validation that rejects a misleading or incomplete
  response with a structured reason.

Budgets use characters and UTF-8 bytes as the authoritative units. A token
estimate may be returned as convenience metadata, but it must be labeled
approximate and must not change behavior.

Continuation is stateless. A cursor contains a version, content fingerprint,
and a block index or character offset. The server refetches/revalidates through
the governed path and rejects continuation when the content fingerprint no
longer matches. The cursor is not a database key and does not create server-side
session state.

The response exposes bounded, versioned quality and route metadata:

- extraction/quality algorithm version;
- attempted tiers and selected tier;
- trust classification and bounded fallback reasons;
- truncation and authoritative character/byte budgets;
- content fingerprint and continuation data when applicable.

The MCP adapter must reuse the existing article orchestration instead of adding
another extractor or fallback classifier. Classifier changes require fixture
evidence of a current gap.

## 7. General search fusion contract

Fusion lives beside the general immutable `SearchResult` and
`SearchResultsPayload` contracts. It accepts ordered provider results and
provider status, then produces a versioned deterministic result.

The first algorithm may combine:

- conservative canonical-URL deduplication;
- reciprocal-rank fusion;
- independent-provider consensus;
- domain-diversity adjustment;
- stable provider/result tie-breaking;
- explicit healthy, degraded, failed, and weak-result status.

Provider ordering, coroutine completion order, and mapping iteration order must
not change the output. MCP web search can opt into this contract. Scholarly
Research discovery may reuse pure normalization or scoring helpers, but retains
its DOI, PMID, PMCID, arXiv, open-access, and provider-specific fields.

## 8. Browser DNS-rebinding gate

Current Playwright routing is necessary but insufficient: a browser can resolve
an approved hostname independently after URL policy validation. Until a browser
transport can prove it connects through a governed proxy or equivalent pinned
and peer-attested path:

- strict multi-user and untrusted profiles disable browser escalation;
- governed HTTP extraction remains available;
- denial is returned as a structured capability reason, not misclassified as an
  extraction failure;
- authenticated browser sessions remain unavailable and separately depend on
  TASK-13100.

This is a release gate, not a commitment to build custom browser networking. A
small fail-closed capability gate is the preferred initial implementation.

## 9. Cookie decision

TASK-13100 owns this work. The first release takes the smallest safe path:

1. Remove ordinary-user raw cookie read/list/write endpoints.
2. Disable persistent web-scraping cookies in multi-user mode.
3. Permit only explicit request-scoped cookies on already-approved workflows.
4. Quarantine and retire or securely remove the legacy plaintext cookie file.
5. Prove credentialless discovery, MCP fetch, caches, logs, Jobs metadata, and
   browser gates never consult ambient cookie state.
6. Defer an encrypted owner-scoped cookie vault until a concrete authenticated
   retrieval requirement is separately approved.

## 10. Conditional revalidation and retrieval envelopes

Conditional revalidation extends the existing opt-in process-local TTL/LRU MCP
cache. The first release is intentionally narrow:

- public credentialless GET retrieval only;
- existing bounded key dimensions such as URL, format, byte limit, and robots
  mode;
- `ETag`/`If-None-Match` and `Last-Modified`/`If-Modified-Since` when supplied;
- 304 reuse without re-extraction;
- changed 200 responses replace the ephemeral entry;
- any future credential-bearing request shape bypasses the cache fail closed.

The first release does not support durable cache persistence, owner-scoped MCP
cache databases, arbitrary `Vary`, cookies, authorization, custom
credential-bearing headers, or cross-process coherence.

A retrieval envelope is serializable safe provenance, not a storage service. It
contains only bounded data such as safe source/final URLs, retrieval time,
content fingerprint, selected validators, extraction/quality versions, tier
trace, and change classification. MCP may return it; only Media may persist it.

## 11. Crawl follow-ups

TASK-12964 remains the sole owner of the Phase 2B HTML-to-Media handoff, Media
Jobs crawl, credential envelope, partial results, and cooperative cancellation
decision. The program adds no implementation task until that design is approved
and its concrete implementation task exists.

After that baseline lands, two narrow follow-ups are allowed:

1. Owner-scoped compact checkpoints, deterministic resume, and explicit stop
   reasons through the same Jobs contract.
2. Versioned URL canonicalization, bounded same-origin sitemap seeding, and
   deterministic anchor/path/locale relevance ordering.

Resume can preserve or tighten the original scope, never expand it. Stop reasons
include frontier exhaustion and each hard page, character/byte, depth, deadline,
cancellation, or policy limit. Checkpoints and results exclude raw HTML and
credentials.

## 12. Media and Chatbook provenance

When a user elects to keep a retrieval:

1. The safe retrieval envelope is handed through `/api/v1/media/add` and the
   approved Phase 2B Media Jobs path.
2. Media stores extracted content and bounded provenance using existing metadata
   and versioning seams.
3. Chatbook export maps Media provenance into v1.1 `source_refs`, snapshot hash,
   resolution status, rehydration hints, and compatible metadata.

The mapping is additive and redacted. It excludes raw HTML, cookies,
authorization, credential-bearing headers, unsafe unsanitized URLs, and MCP
cache internals. Importers preserve supported fields and degrade unsupported
optional provenance with warnings rather than inventing a new content type.

## 13. Evaluation strategy

### Shared baseline

TASK-13139.1 establishes only a versioned offline fixture format, a deterministic
runner, and fast metrics. Each feature task owns its unit/integration/property
tests. Character and UTF-8 byte efficiency are primary; token counts are
estimates.

### Parked robustness work

Fuzz and soak coverage waits until the implemented contracts and algorithm
versions stabilize. It starts informational and becomes required only after
runtime, flake-rate, triage, and ownership criteria are met.

### Parked PDF benchmark

The PDF task measures representative current-path failures before proposing any
production change. It does not port DonSeTch's PDF engine or replace the current
extractors speculatively.

### Parked external comparator

The operator supplies a DonSeTch artifact pinned by commit, release artifact, or
digest at execution time. No release is hardcoded in the roadmap, no binary is
downloaded by default, and required CI never depends on it. A suitability and
AGPL operational review precedes use. tldw_server implementation remains
independent and copies no DonSeTch source; this lightweight comparison is not
described as a formal clean-room process.

## 14. Delivery waves and backlog map

| Wave | Task | Scope | Gate |
| --- | --- | --- | --- |
| Existing prerequisite | TASK-13100 | Minimal cookie remediation | Required before authenticated retrieval/browser work |
| Existing prerequisite | TASK-12964 | Phase 2B HTML-to-Media design and cancellation decision | Required before crawl/provenance implementation |
| Wave 0 | TASK-13139.1 | Minimal deterministic quality baseline | Roadmap approval |
| Wave 0 | TASK-13139.2 | Strict browser DNS-rebinding capability gate | Roadmap approval |
| Wave 1 | TASK-13139.3 | Bounded MCP reading views and existing escalation trace | TASK-13139.1 and TASK-13139.2 |
| Wave 1 | TASK-13139.4 | General web-search fusion | TASK-13139.1 |
| Wave 2 | TASK-13139.5 | Credentialless conditional revalidation and envelopes | TASK-13139.3 |
| Wave 2 | TASK-13139.6 | Post-Phase-2B crawl checkpoints and stop reasons | TASK-13139.1 and TASK-12964; add concrete implementation dependency |
| Wave 2 | TASK-13139.7 | Crawl frontier normalization and relevance | TASK-13139.6 |
| Wave 2 | TASK-13139.8 | Media-owned provenance and Chatbook export | TASK-13139.5 and TASK-12964; add concrete implementation dependency |
| Parked | TASK-13139.9 | Post-contract fuzz and soak gates | Implemented feature contracts |
| Parked | TASK-13139.10 | PDF trust/OCR benchmark | TASK-13139.1 |
| Parked | TASK-13139.11 | Operator-supplied DonSeTch comparator | TASK-13139.1 plus suitable artifact |

## 15. Planning policy

This roadmap is the program-level plan. Detailed implementation plans are
written just in time:

1. Review and approve this roadmap.
2. Write the Wave 0 plan or plans against the then-current dev branch.
3. Implement and verify Wave 0.
4. Reinspect current dev and write Wave 1 plans using the delivered contracts.
5. Repeat for Wave 2; do not plan crawl or persistence around an unresolved
   Phase 2B implementation task.
6. Plan parked work only when its activation gate is satisfied.

Each implementation task follows repository TDD, review, verification, Bandit,
and human-authored AI-PR Change summary requirements.

## 16. Success criteria

The program succeeds when:

- agent reading is bounded, deterministic, resumable without server session
  state, and backward compatible;
- general search fusion is provider-order independent and schema-appropriate;
- strict deployments cannot silently use a browser path with unresolved DNS
  rebinding risk;
- credentialless revalidation improves freshness without durable or cross-owner
  cache state;
- crawl jobs resume deterministically and explain why they stopped;
- durable web provenance has exactly one owner, Media, and exports through
  existing Chatbook contracts;
- cookie security is remediated without speculative credential infrastructure;
- quality gates are evidence-based, fast by default, and versioned.

## 17. Explicitly rejected approaches

- Merge or rebase the earlier `922ff42cc1` planning commit.
- Reuse `TASK-12125` or any `TASK-12125.*` program child.
- Add a new Playwright request-routing layer.
- Add a second HTTP-to-browser escalation ladder.
- Route general MCP web search through scholarly `DiscoveryResult`.
- Persist MCP retrieval envelopes in a new database.
- Let Chatbooks or Research own durable snapshots.
- Build a persistent encrypted cookie vault before a real requirement exists.
- Use token budgets as the primary truncation contract.
- Hardcode any comparator release in required tooling.
- Freeze detailed plans for every wave before current dependencies land.
