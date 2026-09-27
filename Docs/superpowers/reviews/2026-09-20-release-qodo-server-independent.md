# Independent read-only review: release repairs

Reviewed committed range bdeedd7282^..89cf5448b1, excluding my own scheduled-task, macro, SQLite acquisition, and Personal Context changes. No repository edits in this review.

## Findings

No concrete new regressions or missed requirements found in the reviewed repairs.

## Evidence inspected

- MCP brokered credential validation: case-insensitive reserved protocol/session/framing/header rejection runs before merging request headers; static configuration remains a separate trusted input. Shared JSON-RPC envelope validation covers ordinary JSON, matching streamable SSE responses, and legacy SSE responses. Existing ID correlation remains in the request paths.
- Public shutdown behavior regression: real registered module, public `protocol.process_request`, declared idempotency key and real server shutdown exercise operation-before-module-teardown ordering; no private finalizer creation/access remains in the rewritten test.
- Readiness: legacy ready/engine/db/time fields are restored alongside the existing readiness projection; status code and no-store behavior preserved. Database compatibility fields derive from the previously sanitized snapshot. Existing health authentication policy was intentionally not relaxed by this repair.
- Notes tombstones: the endpoint now couples include_deleted_endpoints to include_deleted_links, and the same include_deleted value remains bound into pagination cursors. Owner-scoped user database and existing dataset/cursor binding remain intact.
- Frontend cancellation: refresh JSON parsing errors reach the catch, where abort classification runs before ignored refresh failures and before falling through to the original 401; configuration-scope errors retain their special handling.
- Browser TTS: actual extension runtime identity selects extension speech APIs, while the no-op web-shim path now reaches native speechSynthesis; stop uses the same selection predicate.
- Ingest: candidate iteration considers top-level identifiers then all batch results, with invalid candidate identifiers no longer masking valid later aliases/results.
- Standalone HTML i18n: message state stores semantic keys, translation happens during render, and status/recovery/conflict labels share the existing playground namespace with matching English fallback keys.

## Bounds

Static, focused review of the stated commits plus adjacent call sites. Did not rerun other agents' suites, perform live browser QA, or assess unrelated older behavior. Parent owns aggregate validation. No claim of exhaustive security or whole-repository review.
