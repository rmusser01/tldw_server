# TASK13260.212 / UAT271 report

## Result

The Character editor now preserves the version from the record selected for edit and passes it through the existing update mutation. The existing client transport sends that value as `expected_version`; it already omitted the query when no value was supplied.

## Causal fix

The safe native response projection recorded an edit `PUT` with no `expected_version` and a backend `422`. The editor's `handleEdit` copied the record identifier and form fields but omitted `record.version`. Its mutation already forwarded `editVersion ?? undefined`.

`handleEdit` now stores a loaded non-negative integer version, and `CharactersManager` passes the existing setter into that hook. A missing or malformed version remains `null`; no save-time fetch or conflict bypass was added. The focused regression also asserts that saving does not call `getCharacter`.

The production files `TldwApiClient.ts` and `domains/characters.ts` remain unchanged: their conditional query behavior was already correct and is covered by a focused client test, including version `0`.

## Verification

- Causal red: the focused edit-flow test failed with `expected undefined to be 3` before the hook/state wiring.
- Causal green: the same test passed (`1 passed, 98 skipped`).
- Client conditional-query regression: `4 passed`.
- `git diff --check` passed.
- `apps/tldw-frontend`: `bun run typecheck` returned the known baseline of 90 diagnostics and reported no changed UAT271 file path. Direct UI compilation with an 8 GiB Node heap similarly found no changed-path diagnostic among 380 existing diagnostics; its default-heap attempt had exhausted memory.

The full Manager test file was interrupted after progress and is not counted as passing evidence. Bandit was run as required but cannot parse TypeScript, so it provides no TypeScript security assessment. The configured frontend ESLint invocation ignored these packages/ui paths, and Prettier reported whole-file baseline warnings; neither is claimed as a clean lint/format result.

Exact commands, status, limits, and artifact hash are retained in [verification.json](../../../.tmp/uat-repairs-231-246/character271/verification.json).

## Changed files

- `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx`
- `apps/packages/ui/src/components/Option/Characters/Manager.tsx`
- `apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx`
- `apps/packages/ui/src/services/__tests__/tldw-api-client.characters-delete.test.ts`
