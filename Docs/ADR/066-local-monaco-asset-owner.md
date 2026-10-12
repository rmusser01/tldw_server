# ADR-066: Shared local Monaco assets and loader

**Status:** Accepted
**Date:** 2026-10-11
**Decision owner:** TASK-13520 controller, Ruling47
**Related task:** TASK-13520
**Related spec:** Chatbook Docs/superpowers/specs/2026-09-27-structured-document-editing-design.md
**Related plan:** Chatbook Docs/superpowers/plans/2026-09-27-structured-document-editing.md
**Backfilled from:** N/A — controller decision recorded before implementation
**Extends:** [ADR-065](065-portable-note-language-and-negotiated-structured-editing.md)

## Decision

One fixed build-time preparer copies the installed Monaco `min/vs` distribution and its five native worker bootstraps into the Next public directory and the existing shared WXT public directory. Existing postinstall/build wrappers and advertised development commands invoke that preparer. Generated installed assets are not repository source. `next start` and standalone serving only consume prepared public assets and never import the preparer or require installed build dependencies.

One shared loader configures fixed same-origin `/monaco/vs` and packaged worker paths before any lazy Monaco consumer initializes the global loader. Notes, PresentationStudio and Watchlists use that same initialization. No caller-selected URL, CDN, new dependency, CSP weakening, bundler replacement or plugin bypass is introduced. Document processing remains the distinct E4 app-owned service with trusted Next native and WXT `?worker` construction.

## Context

The reviewed I1 candidate used test-only local installed-asset copies. Actual PresentationStudio and Watchlists consumers use the default global Monaco loader, whose ordering can start a CDN load before Notes. A per-Notes configuration cannot qualify that previous initialization or production packaging. Production integration needs one existing-consumer owner for assets and initialization.

## Alternatives considered

- Per-Notes loader configuration leaves earlier global consumers uncontrolled.
- CDN loading introduces network-dependent source-editor availability and violates local packaging admission.
- Configuration-time copying makes runtime serving depend on installed build packages and performs filesystem mutation during startup.
- A new editor dependency, custom bundler pipeline or relaxed CSP adds unnecessary boundaries.

## Consequences and qualification

The installed distribution is copied during existing preparation/build/development paths, while serving stays independent. All three consumers must be checked for initialization ordering. Actual default Next, production/standalone and installed extension loading remain TASK-13520/I5 evidence gates; the candidate smoke does not earn them. Missing assets must produce truthful unavailable editing with canonical source retained. Native worker termination remains asynchronous per ADR-065/Ruling42.

The installed0.55 AMD entry replaces MonacoEnvironment during initialization. The shared lazy loader reapplies its fixed native factory after initialization, and both prior React consumers await this loader before mounting. Actual extension probes exposed the default blob factory’s importScripts failure despite packaged files; a pre-init assignment alone is insufficient. This keeps all consumers on the same approved owner and preserves their error fallbacks.
