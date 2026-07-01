# Standalone MCP Docs Stage 2 URL Acquisition Design

Date: 2026-06-30
Status: Draft for user review
Backlog: TASK-12076.1
Related:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md
- Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md

## Summary

Stage 2 adds optional single-page URL acquisition to the standalone MCP docs
corpus. It introduces `docs.ingest_url` for approved URLs, with source-policy
checks, no-fetch-before-approval behavior, redirect-aware SSRF defenses,
bounded response handling, static/rich extraction, and ingestion into the
existing SQLite + FTS5 document store.

This is deliberately not a crawler, browser automation layer, cookie/session
scraper, or `tldw_server` service bridge. The goal is to make the standalone
docs corpus useful for agent-directed page ingestion while preserving the Stage
1 guarantees: local docs work without web extras, `mcp_unified.docs` has no
runtime dependency on `tldw_Server_API`, and optional web acquisition can be
disabled for locked-down deployments.

## Goals

- Expose `docs.ingest_url` only when web acquisition is enabled.
- Fetch and ingest one approved `http` or `https` URL into the docs corpus.
- Enforce config-driven source profiles before any network request.
- Return `approval_required` for unknown domains or URL prefixes according to
  profile policy, without fetching content.
- Deny unsupported schemes, local file URLs, private/loopback/link-local DNS
  results, unsafe redirects, unsupported content types, and oversized bodies.
- Use rich extraction when optional libraries are installed, with a stdlib
  fallback when they are not.
- Report acquisition status, extraction method, policy decisions, redirect
  summary, and warnings in tool responses and `docs.status`.
- Test acquisition through fake resolver and transport seams, with no live
  internet dependency.

## Non-Goals

- No broad crawler, sitemap sync, recursive link discovery, or `docs.sync_source`.
- No Playwright or browser fallback in this slice.
- No cookies, credentials, session cloning, stealth behavior, or host browser
  profile integration.
- No mandatory `trafilatura`, BeautifulSoup, requests, aiohttp, httpx, or
  Playwright dependency for importing the baseline docs package.
- No direct imports from `tldw_Server_API` inside `mcp_unified.docs`.
- No Media DB, ChromaDB, RAG service, Jobs, Scheduler, or AuthNZ coupling.
- No CLI or WebUI flow for managing approval state; Stage 2 uses explicit
  configuration only.

## Approved Approach

Use a standalone-safe single-page acquisition layer inside `mcp_unified.docs`.
Existing `tldw_server` scraping code remains reference material for extraction
and policy behavior, but Stage 2 should copy or adapt only focused standalone
utilities. Host-specific scraping services, Media DB writes, cookies, browser
automation, and application lifecycle code must stay outside the standalone
package.

The first implementation should prefer boring, inspectable primitives:

- stdlib URL parsing, DNS resolution, and IP classification;
- stdlib HTTP transport by default, behind an injectable transport interface;
- manual redirect handling instead of automatic redirects;
- optional lazy imports for rich extraction only inside extractor functions;
- the existing docs store/import normalization path for chunking, collection
  membership, keywords, and FTS indexing.

## Components

### Settings

Extend `DocsSettings` with web acquisition configuration:

- `enable_web_acquisition`: advertises and enables `docs.ingest_url`.
- `web_source_profile`: one of `locked_down`, `local_first`, or
  `online_capable`.
- `preapproved_domains`: domains that can be fetched directly.
- `allowed_url_prefixes`: specific URL prefixes that can be fetched directly.
- `denied_domains`: domains that are always denied.
- `max_url_redirects`: small redirect cap.
- `max_url_body_bytes`: hard response body limit.
- `url_request_timeout_seconds`: network timeout.
- `allowed_content_types`: default to static text and HTML types.
- `url_user_agent`: configured user-agent for URL acquisition.
- `respect_robots`: default false for Stage 2 unless a standalone robots
  checker is implemented in the same plan with tests.
- `allow_arbitrary_public_domains`: default false. Only meaningful for
  `online_capable`; when false, unknown public domains still return
  approval-required.

Locked-down config can keep `enable_web_acquisition` false or enable it with
only explicit `allowed_url_prefixes`; domain-only allow rules are ignored in
this profile so locked-down deployments remain narrow. `local_first` allows
configured domains/prefixes and returns approval-required for unknown domains.
`online_capable` allows configured domains/prefixes and may also allow unknown
public domains only when `allow_arbitrary_public_domains` is explicitly true.
The repository's default config must keep this flag false.

### Source Policy

`mcp_unified.docs.acquisition.policy` owns source decisions before fetch:

- normalize URLs and reject missing scheme/host;
- allow only `http` and `https`;
- reject credentials in URLs;
- apply denied domains before allow rules;
- match configured allowed domains and URL prefixes;
- return stable decisions: `allowed`, `approval_required`, or `denied`;
- compute a safe argument hash for approval-required responses without storing
  query secrets in logs or messages.

The policy layer must not perform network I/O. That separation lets tests prove
no fetch occurs when approval is required.

Domain and prefix matching must be exact and structured:

- normalize hostnames with IDNA and lowercase comparison;
- normalize default ports before comparison;
- treat `example.com` as matching only `example.com`, not
  `badexample.com` or arbitrary subdomains;
- support subdomains only through explicit wildcard rules such as
  `*.example.com`, where the wildcard does not match the apex unless both are
  configured;
- reject raw string prefix matching for host allow/deny checks;
- parse URL-prefix rules into scheme, host, optional port, and path components;
- compare path prefixes on decoded path-segment boundaries so
  `/docs/` does not allow `/docs.evil/`;
- ignore fragments for policy decisions and include query strings only in the
  safe argument hash, never in logs.

### URL Safety And Fetching

`mcp_unified.docs.acquisition.fetcher` owns network execution after policy
allows a URL:

- resolve hostnames through an injectable resolver;
- deny loopback, private, link-local, multicast, unspecified, and reserved IPs;
- open requests through an injectable transport;
- bind transport connections to resolver-validated addresses, or otherwise
  prove the transport cannot perform an unvalidated hostname re-resolution;
- disable automatic redirects;
- validate each redirect target before following it;
- re-run source policy, DNS, and IP checks on redirect targets;
- enforce redirect count;
- validate content type before reading the full response when headers are
  available;
- request `Accept-Encoding: identity` in the baseline transport;
- read bodies with a hard byte limit and enforce limits on both transferred and
  decoded bytes if an optional transport supports compression;
- return final URL, status code, headers, redirect chain summary, and bytes.

The design avoids relying on `requests`, `httpx`, or `aiohttp` for the
baseline. If a future optional transport uses those libraries, it must remain
behind the same interface and must not be imported at package import time.
Tests must include a DNS-rebinding scenario where the first resolution is
public and a later transport resolution would be private; the fetcher must deny
or avoid that re-resolution path.

If `respect_robots` is true and no standalone robots checker is implemented,
`docs.ingest_url` must fail closed with `robots_unavailable` before fetching
page content. Robots support is not required for this Stage 2 slice, but the
setting cannot silently fail open.

### Extraction

`mcp_unified.docs.acquisition.extract` converts fetched content into a parsed
document suitable for the existing import/store path.

Extraction order:

1. `trafilatura`, if installed and enabled, imported lazily inside the extractor.
2. BeautifulSoup, if installed and enabled, imported lazily inside the extractor.
3. Existing stdlib static HTML/text extraction fallback.

Extractor output must include:

- title;
- normalized body text;
- document type;
- source URL and canonical URI;
- best-effort headings/sections;
- extraction method: `trafilatura`, `beautifulsoup`, `static_html`, or `text`;
- warnings when rich extraction is unavailable or falls back.

Adding `bs4` to import-boundary tests is part of this design. No optional rich
extractor import may appear at module top level inside `mcp_unified.docs`.

### Acquisition Service

`DocsAcquisitionService` coordinates the operation:

1. Receive URL, collections, keywords, title override, and optional profile.
2. Evaluate source policy.
3. Return `approval_required` or `denied` before network I/O when policy says so.
4. Fetch with redirect and SSRF guards.
5. Extract content.
6. Upsert a document with `canonical_uri` based on the final canonical URL.
7. Apply keyword and collection metadata through existing store helpers.
8. Return `created`, `updated`, `unchanged`, or `denied`.

The service should share chunking and store writes with Stage 1 import logic
where practical, but should not force URL content through filesystem path
objects. If the Stage 1 parsed-document model requires `source_path`, Stage 2
should generalize it to support `source_url` without pretending URLs are files.

### MCP Tool Provider

`DocsMCPToolProvider.tool_definitions()` advertises `docs.ingest_url` only when:

- `settings.enable_web_acquisition` is true; and
- the acquisition service can be constructed without missing required baseline
  pieces.

`docs.status` reports:

- `web_acquisition_enabled`;
- `web_acquisition_available`;
- `web_source_profile`;
- configured policy mode summary;
- available extractors, for example `["static_html"]` or
  `["trafilatura", "beautifulsoup", "static_html"]`;
- disabled or unavailable reason code when relevant.

`docs.ingest_url` is categorized as `ingestion` and is write-capable for MCP
policy/rate-limit purposes.

## Data Flow

### Disabled Capability

1. Client calls `docs.status` or lists tools.
2. `docs.status` reports web acquisition disabled.
3. `docs.ingest_url` is not advertised.
4. If a stale client calls `docs.ingest_url`, the provider returns
   `capability_disabled` without policy or network work.

### Approval Required

1. Client calls `docs.ingest_url` with an unknown URL under `local_first`.
2. Source policy returns `approval_required`.
3. Service returns status, reason code, domain, requested scope, canonical URL,
   and safe argument hash.
4. No resolver or transport method is called.

### Approved Single Page

1. Client calls `docs.ingest_url` with a URL under an allowed domain or prefix.
2. Source policy allows the URL.
3. Fetcher resolves DNS and blocks unsafe IP ranges.
4. Fetcher performs a request with manual redirect handling.
5. Each redirect target is validated before following.
6. Fetcher enforces content type and byte limits.
7. Extractor parses HTML/text using the best available method.
8. Acquisition service writes the document and chunks to SQLite/FTS5.
9. `docs.search` and `docs.context` can retrieve the ingested page.

## Error Handling

Responses should use stable machine-readable `status` and `reason_code` values:

- `capability_disabled` / `web_acquisition_disabled`
- `approval_required` / `source_approval_required`
- `denied` / `source_domain_denied`
- `denied` / `unsupported_url_scheme`
- `denied` / `url_credentials_denied`
- `denied` / `egress_private_address_denied`
- `denied` / `redirect_policy_denied`
- `denied` / `redirect_limit_exceeded`
- `denied` / `content_type_denied`
- `denied` / `content_too_large`
- `denied` / `robots_unavailable`
- `failed` / `fetch_timeout`
- `failed` / `fetch_error`
- `failed` / `extract_empty`
- `created`, `updated`, or `unchanged` for successful ingest outcomes

Errors should include safe diagnostics but not raw secrets from query strings,
headers, or credentials. Fetch/audit summaries may include sanitized URL host
and path, final URL, status code, byte count, content type, redirect count,
extraction method, and warnings.

## Testing Strategy

All Stage 2 tests must be deterministic and avoid live internet.

Unit tests:

- settings parsing for source profile, domains, prefixes, content types, limits;
- disabled capability status and hidden tool discovery;
- `docs.ingest_url` stale-call disabled result;
- policy decisions for locked-down, local-first, and online-capable profiles;
- no-fetch-before-approval using fake resolver/transport call counters;
- unsupported scheme and credential URL denial;
- denied-domain precedence over allowed-domain config;
- domain matching normalization, explicit wildcard handling, and URL prefix
  path-boundary matching;
- DNS private/loopback/link-local/multicast/reserved denial;
- DNS-rebinding denial or validated-address transport binding;
- redirect validation and redirect-to-private denial;
- redirect count denial;
- content-type denial;
- compressed and decoded response size denial;
- `respect_robots` fail-closed behavior when no robots checker exists;
- extraction fallback order with monkeypatched optional imports;
- stdlib extraction works without rich libraries.

Integration tests:

- approved fake HTTP URL ingests into the docs store;
- ingested page is returned by `docs.search`;
- ingested page can appear in `docs.context`;
- keywords and collections passed to `docs.ingest_url` are applied;
- status reports enabled with available extractors;
- import-boundary test verifies no `tldw_Server_API`, `requests`, `aiohttp`,
  `httpx`, `playwright`, `trafilatura`, or `bs4` top-level imports in the
  baseline package.

Security validation:

- Bandit on `mcp_unified/docs/acquisition`, modified docs package files, and
  host shim changes.
- Focused review of redirect, DNS, body-limit, and no-fetch-before-approval
  tests before implementation is called complete.

## Rollout And Profiles

Default Stage 2 rollout keeps web acquisition disabled in the repo's built-in
MCP config. Users enable it explicitly:

```yaml
settings:
  enable_web_acquisition: true
  web_source_profile: local_first
  allow_arbitrary_public_domains: false
  preapproved_domains:
    - docs.python.org
  allowed_url_prefixes:
    - https://docs.python.org/3/
```

Locked-down deployments can keep the Stage 1 behavior by setting
`enable_web_acquisition: false`, or by using `locked_down` with only specific
prefixes. The implementation plan should not flip default configs to online
capable.

## Acceptance Criteria

- `docs.ingest_url` is hidden when web acquisition is disabled.
- Stale direct calls to `docs.ingest_url` fail before policy or network work
  when disabled.
- `docs.ingest_url` returns approval-required without fetch for unknown sources
  under approval-requiring profiles.
- `online_capable` only fetches unknown public domains when
  `allow_arbitrary_public_domains` is explicitly true.
- Approved single-page URLs can be ingested, searched, and included in context
  packs.
- The fetch path validates DNS/IP policy before connect and after redirects.
- The fetch path binds the network connection to resolver-validated addresses
  or denies possible DNS-rebinding paths.
- The fetch path denies private, loopback, link-local, multicast, and reserved
  addresses.
- Redirects are manual, capped, and policy-checked per hop.
- Content type, transferred body-size, and decoded body-size limits are
  enforced.
- `respect_robots: true` fails closed with `robots_unavailable` unless a
  standalone robots checker is implemented and tested.
- Optional rich extractors are lazy and never required for package import.
- `docs.status` distinguishes disabled, enabled stdlib-only, and enabled rich
  extractor availability.
- Tests use fake resolver/transport and require no live internet.
- `mcp_unified.docs` still imports without `tldw_Server_API` or optional web
  scraping dependencies.
