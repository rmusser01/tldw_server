# TASK13260.213 / UAT272 — World Book attachment hydration report

## Result

The native failure was caused by the character collection client choosing `/api/v1/characters` before the backend's canonical `/api/v1/characters/`. When OpenAPI discovery is unavailable, the client uses its first candidate; the retained native redirect then received a `429` under `character_chat.default`, leaving the Manager without candidates for reciprocal association hydration.

The repair puts the trailing-slash route first in both maintained character client implementations while retaining the no-slash alternate fallback. The World Books Manager defers the character collection request until an attachment entry point requests hydration. It preserves sequential owner-scoped `listCharacterWorldBooks` reads and mutation paths. A missing or inaccessible relationship (`404` or `403`) remains an empty association for that character; a transport or server failure rejects the attachment query, so it cannot be rendered as a false zero.

The detail panel now distinguishes an unavailable or loading attachment state from zero attachments. Its neutral retryable error has a **Try again** button that invalidates the existing character and attachment query keys. It does not claim an authentication or authorization cause; native evidence did not retain request headers.

The same Manager must retain a disabled persisted link after reload because it renders membership and edits `attachment_enabled` and priority. Its relationship read now explicitly requests `enabled_only=false`; the two maintained readers accept `includeDisabled` and preserve the ordinary endpoint-default path for every no-argument caller. This is a Manager-only opt-in, not a backend default change.

## Verification

- Causal red: 3 test files, 52 tests; 3 expected failures before the repair: canonical collection route, lazy candidate query, and error state.
- Review revision causal red: the relationship transport rejection and both retry-button controls failed before the revision. The base-client path test was corrected to call `TldwApiClientBase.prototype.listCharacters` directly.
- The retained green receipts cover 62 focused tests: client request-scope; Manager attachment, matrix, attach/detach, metadata, and quick-attach controls; and the detail panel. There were no skips.
- Disabled-link causal red: 47 focused tests ran, with 45 passing and 2 expected failures before the opt-in read change. The retained full log proves the Base/client transport omitted `enabled_only=false` and the actual Manager query function returned an empty map when the disabled relationship was hidden.
- Final green receipts cover 63 focused tests with no skips: 57 in the client/Manager/detail group and 6 adjacent Manager controls.
- `git diff --check` passed for the owned source, tests, brief, and report.
- `apps/tldw-frontend` typecheck exited 2 with the repository's established **90** unrelated diagnostics and none in changed paths. The earlier package-level compiler exhausted the Node heap and is superseded by this intended entry-point receipt.
- Root-scoped ESLint assessed eight owned paths without ignores. It reported one pre-existing Manager error at `openEntries?.id!` and 963 warnings; its baseline comparison is retained separately by root. This is not a clean lint result. Bandit exited 0 with zero findings but reported parse errors for both TypeScript client files, so it is not language-level security assurance.

## Scope and limitation

No backend routes, resource-governor policy, authorization behavior, or global proxy behavior changed. The retained native `429` remains classified in the UAT evidence. The fix prevents this client from selecting the redirecting route and avoids eager attachment catalog reads; it does not alter the policy or assert a missing authentication header.

## Source hashes

See `.tmp/uat-repairs-231-246/worldbook272/source-sha256.txt`; `pre-review-source-sha256.txt` and `pre-disabled-links-source-sha256.txt` preserve the two earlier review rounds.
