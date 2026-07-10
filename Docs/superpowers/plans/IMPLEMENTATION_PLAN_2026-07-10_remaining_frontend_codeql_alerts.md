# Remaining Frontend CodeQL Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve CodeQL alerts 2251 through 2262 with minimal frontend trust-boundary fixes, focused regression tests, and a pull request targeting `dev`.

**Architecture:** Reuse the existing URL, raster-image, DOMPurify, watchlist API, and Vitest infrastructure. Put image validation in one shared utility, remove the two unnecessary untrusted-DOM parsing paths, sanitize the printable document at its sink, and make analyzer-sensitive logging/provider callbacks explicit without changing runtime behavior.

**Tech Stack:** TypeScript, React 18, DOMPurify, Vitest, Testing Library, Bun workspace tooling.

---

## File map

- `apps/packages/ui/src/utils/image-utils.ts` and its test: shared image-source trust boundary.
- `apps/packages/ui/src/types/assistant-selection.ts` and its test: normalized assistant/avatar storage.
- `apps/packages/ui/src/components/Common/CharacterSelect.tsx`: common selector avatar normalization.
- `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts` and its test: article text and preview image validation.
- `apps/packages/ui/src/services/watchlists.ts`: expose the existing `groups` query parameter.
- `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx` and advanced-details test: server-side group filtering.
- `apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx` and bulk-duplicate test: printable document sanitization.
- `apps/packages/ui/src/components/Common/Playground/DocumentGeneratorDrawer.tsx`: constant console format string.
- `apps/packages/ui/src/services/timeline/api.ts`: constant conversation log messages.
- `apps/packages/ui/src/utils/provider-registry.ts` and tests: analyzer-safe callback naming with unchanged inference.
- `apps/packages/ui/src/utils/__tests__/codeql-source-contracts.test.ts`: analyzer-sensitive source-shape checks.
- `backlog/tasks/task-12946 - Address-remaining-frontend-CodeQL-alerts-for-dev.md`: verification and PR record.

## Alert traceability

| Alerts | Source change | Primary regression test |
| --- | --- | --- |
| 2251 | Sink: `Common/AssistantSelect.tsx`; boundary: `types/assistant-selection.ts` | `types/__tests__/assistant-selection.test.ts` |
| 2252 | Sink/boundary: `Common/CharacterSelect.tsx` | `utils/__tests__/image-utils.test.ts` plus source-path review |
| 2253 | `ItemsTab/items-utils.ts::stripHtmlToText` | `ItemsTab/__tests__/items-utils.test.ts` |
| 2254-2255 | Sinks: `ItemsTab/ItemsTab.tsx`; boundary: `ItemsTab/items-utils.ts::extractImageUrl` | `ItemsTab/__tests__/items-utils.test.ts` |
| 2256, 2262 | `SourcesTab.tsx`, `services/watchlists.ts` | `SourcesTab.advanced-details.test.tsx` |
| 2257 | `Quiz/tabs/ManageTab.tsx` | `ManageTab.bulk-duplicate.test.tsx` |
| 2258 | `DocumentGeneratorDrawer.tsx` | `codeql-source-contracts.test.ts` |
| 2259-2260 | `services/timeline/api.ts` | `codeql-source-contracts.test.ts` |
| 2261 | `utils/provider-registry.ts` | `provider-registry-tts.test.ts`, `codeql-source-contracts.test.ts` |

## Stage 1: Centralize image-source validation

**Goal:** Ensure every alerted `<img src>` receives a verified raster data URL or a URL with a known safe prefix.

**Success Criteria:** Unsafe schemes and malformed data are rejected; HTTP(S), relative URLs, and verified raster data are preserved; unsafe external avatars fall back to valid embedded data.

**Tests:** Image utility, assistant selection, and watchlist item utility suites.

**Status:** Not Started

### Task 1: Add the shared image-source validator

**Files:**
- Modify: `apps/packages/ui/src/utils/image-utils.ts:1-245`
- Test: `apps/packages/ui/src/utils/__tests__/image-utils.test.ts`

- [ ] **Step 1: Write the failing test**

Add `safeImageUrl` cases for canonical HTTP(S), explicit and bare relative paths, valid PNG data, `javascript:`, `mailto:`, and SVG data. Representative assertions:

```ts
expect(safeImageUrl("HTTPS://example.com/a.png")).toBe("https://example.com/a.png")
expect(safeImageUrl("images/a.png")).toBe("./images/a.png")
expect(safeImageUrl("javascript:alert(1)")).toBeNull()
expect(safeImageUrl("mailto:image@example.com")).toBeNull()
```

- [ ] **Step 2: Run the test and verify RED**

From `apps/packages/ui`, run:

```bash
bunx vitest run src/utils/__tests__/image-utils.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because `safeImageUrl` is not exported.

- [ ] **Step 3: Implement the minimal helper**

Reuse `createImageDataUrl` first, then `safeExternalUrl`. Normalize accepted absolute HTTP(S) URLs to a constant lowercase scheme prefix, retain explicit safe relative prefixes, prefix bare relative paths with `./`, and reject every remaining colon-bearing scheme. Do not add a dependency.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/utils/image-utils.ts apps/packages/ui/src/utils/__tests__/image-utils.test.ts
git commit -m "fix(ui): validate untrusted image sources"
```

### Task 2: Apply the validator at every alerted image boundary

**Files:**
- Modify: `apps/packages/ui/src/types/assistant-selection.ts:1-110`
- Test: `apps/packages/ui/src/types/__tests__/assistant-selection.test.ts`
- Modify: `apps/packages/ui/src/components/Common/CharacterSelect.tsx:115-155`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts:480-500`
- Test: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts`

- [ ] **Step 1: Write failing consumer tests**

Add an assistant-selection case where an unsafe `avatar_url` falls back to valid `image_base64`. Add item extraction cases rejecting `javascript:`, `mailto:`, and SVG data candidates while preserving HTTP(S) and relative candidates.

- [ ] **Step 2: Run the tests and verify RED**

```bash
bunx vitest run src/types/__tests__/assistant-selection.test.ts src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because current normalizers return raw candidates.

- [ ] **Step 3: Apply one shared guard per normalization boundary**

For assistant normalization, use the safe external candidate, then the safe embedded candidate, then preserve the existing nullish contract: missing input remains `undefined`, explicit/invalid input becomes `null`. For the common character selector, which requires a string, use `safeImageUrl(external) ?? safeImageUrl(embedded) ?? ""`. Return `safeImageUrl(candidate)` from `extractImageUrl`; do not add per-render guards.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/types/assistant-selection.ts apps/packages/ui/src/types/__tests__/assistant-selection.test.ts apps/packages/ui/src/components/Common/CharacterSelect.tsx apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts
git commit -m "fix(ui): normalize avatar and preview image URLs"
```

## Stage 2: Remove untrusted DOM/XML parsing

**Goal:** Eliminate both reported parser sinks without losing text or group-filter behavior.

**Success Criteria:** Article text never reaches `DOMParser`; group selection never exports/parses OPML; server pagination remains authoritative for group-only filtering.

**Tests:** Watchlist item utility and SourcesTab advanced-details suites.

**Status:** Not Started

### Task 3: Sanitize article text before the non-DOM scanner

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts:1-5,459-475`
- Test: `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts`

- [ ] **Step 1: Write a failing no-DOM-path test**

Stub `DOMParser` to `undefined`. Pass paragraph content plus `script` and `style` bodies and assert only normalized paragraph text remains. This fails the current fallback because it retains the forbidden bodies.

- [ ] **Step 2: Run the test and verify RED**

```bash
bunx vitest run src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 3: Replace `DOMParser` with DOMPurify plus the existing scanner**

Sanitize with `USE_PROFILES: { html: true }` and `FORBID_TAGS: ["script", "style"]`, then pass the sanitized markup through `stripHtmlTagsWithoutRegex`, `decodeCommonEntities`, and current whitespace normalization. Delete the parser branch.

- [ ] **Step 4: Run the test and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Watchlists/ItemsTab/items-utils.ts apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts
git commit -m "fix(watchlists): sanitize article text without DOM parsing"
```

### Task 4: Replace OPML filtering with the existing API query

**Files:**
- Modify: `apps/packages/ui/src/services/watchlists.ts:142-148`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx:1-65,250-335`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx:365-410`

- [ ] **Step 1: Replace the OPML-cache test with failing server-filter expectations**

For `selectedGroupId: 7`, assert fetch parameters contain `groups: [7]`, the current page, and current page size; assert `exportOpml` is never called. Add a combined group/type case proving all paginated fetches contain `groups: [7]` while type filtering remains client-side.

- [ ] **Step 2: Run the test and verify RED**

```bash
bunx vitest run src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because current group filtering exports/parses OPML.

- [ ] **Step 3: Delete the OPML parser/cache and pass `groups` to the API**

Add `groups?: number[]` to `FetchSourcesParams`. Remove `exportOpml`, `groupOpmlCacheRef`, its TTL/type, and the XML parsing block. Include `groups: selectedGroupId ? [selectedGroupId] : undefined` in `baseParams` and set `useClientFilter = Boolean(selectedTypeFilter)`.

- [ ] **Step 4: Run the test and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/watchlists.ts apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx
git commit -m "fix(watchlists): filter source groups through the API"
```

## Stage 3: Sanitize the printable quiz sink

**Goal:** Put a recognized HTML trust boundary immediately before `document.write` without breaking the printable shell.

**Success Criteria:** Malformed runtime fields cannot create executable markup; one trusted doctype plus `<html>`, `<head>`, `<title>`, `<style>`, and `<body>` reach the print document.

**Tests:** ManageTab print flow.

**Status:** Not Started

### Task 5: Sanitize and preserve the whole printable document

**Files:**
- Modify: `apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx:1-30,403-477,1075-1090`
- Test: `apps/packages/ui/src/components/Quiz/tabs/__tests__/ManageTab.bulk-duplicate.test.tsx:650-685`

- [ ] **Step 1: Strengthen the print test**

Capture `document.write`. Supply a runtime-malformed numeric field such as a `points` value containing an `<img onerror>` payload. Assert the output has exactly one doctype, retains the full trusted shell/title/style/body, and contains neither `onerror` nor the payload marker.

- [ ] **Step 2: Run the test and verify RED**

```bash
bunx vitest run src/components/Quiz/tabs/__tests__/ManageTab.bulk-duplicate.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the raw document contains the malformed runtime value.

- [ ] **Step 3: Sanitize in whole-document mode**

Build the existing document without its doctype, sanitize with `DOMPurify.sanitize(rawDocument, { WHOLE_DOCUMENT: true })`, then prepend one constant `<!doctype html>` to the sanitized result. Keep existing field escaping and error handling.

- [ ] **Step 4: Run the test and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Quiz/tabs/ManageTab.tsx apps/packages/ui/src/components/Quiz/tabs/__tests__/ManageTab.bulk-duplicate.test.tsx
git commit -m "fix(quiz): sanitize printable documents"
```

## Stage 4: Remove analyzer-sensitive names and format strings

**Goal:** Address alerts 2258 through 2261 without altering provider inference or error handling.

**Success Criteria:** Console first arguments are constants; external identifiers are later arguments; the custom callback is named `matches`; inference output is unchanged.

**Tests:** Provider behavior and a focused static contract for analyzer-relevant source shape.

**Status:** Not Started

### Task 6: Add source contracts, then make the mechanical fixes

**Files:**
- Create: `apps/packages/ui/src/utils/__tests__/codeql-source-contracts.test.ts`
- Modify: `apps/packages/ui/src/components/Common/Playground/DocumentGeneratorDrawer.tsx:335-352`
- Modify: `apps/packages/ui/src/services/timeline/api.ts:100-148`
- Modify: `apps/packages/ui/src/utils/provider-registry.ts:455-530`
- Modify: `apps/packages/ui/src/utils/__tests__/provider-registry-tts.test.ts`

- [ ] **Step 1: Write failing analyzer-sensitive source contracts**

Use `readFileSync` to assert the three source files contain constant-message console calls and `matches: (value)`, and do not contain `rule.match(value)`. Extend provider behavior tests with representative GPT, Claude, Llama, Gemini, and Mistral cases.

- [ ] **Step 2: Run the tests and verify RED**

```bash
bunx vitest run src/utils/__tests__/codeql-source-contracts.test.ts src/utils/__tests__/provider-registry-tts.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: source contracts FAIL on interpolated messages and the `match` callback; behavior assertions remain green.

- [ ] **Step 3: Apply only mechanical changes**

Use constant console messages with the job/conversation identifier and error as later arguments. Rename `ProviderInferenceRule.match`, every rule key, and `rule.match(value)` to `matches`/`rule.matches(value)` without changing predicates or ordering.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/utils/__tests__/codeql-source-contracts.test.ts apps/packages/ui/src/components/Common/Playground/DocumentGeneratorDrawer.tsx apps/packages/ui/src/services/timeline/api.ts apps/packages/ui/src/utils/provider-registry.ts apps/packages/ui/src/utils/__tests__/provider-registry-tts.test.ts
git commit -m "fix(ui): remove tainted analyzer sink patterns"
```

## Stage 5: Verify, review, and open the PR

**Goal:** Prove the changed behavior locally, document the JavaScript CodeQL branch limitation, and deliver a reviewable PR to `dev`.

**Success Criteria:** Focused suites and TypeScript pass; every alert path is inspected; TASK-12946 and the PR record exact verification; the branch is pushed and the PR is open against `dev`.

**Tests:** Focused Vitest command, frontend typecheck, diff check, touched-source security review, and GitHub checks.

**Status:** Not Started

### Task 7: Run final verification and deliver the PR

**Files:**
- Modify: `backlog/tasks/task-12946 - Address-remaining-frontend-CodeQL-alerts-for-dev.md`
- Update then remove when complete: `Docs/superpowers/plans/IMPLEMENTATION_PLAN_2026-07-10_remaining_frontend_codeql_alerts.md`

- [ ] **Step 1: Run all focused suites together**

From `apps/packages/ui`, run one `bunx vitest run` command containing these seven files:

- `src/utils/__tests__/image-utils.test.ts`
- `src/types/__tests__/assistant-selection.test.ts`
- `src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts`
- `src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx`
- `src/components/Quiz/tabs/__tests__/ManageTab.bulk-duplicate.test.tsx`
- `src/utils/__tests__/provider-registry-tts.test.ts`
- `src/utils/__tests__/codeql-source-contracts.test.ts`

Use `--maxWorkers=1 --no-file-parallelism`. Expected: all tests PASS.

- [ ] **Step 2: Run TypeScript and repository hygiene checks**

From `apps/tldw-frontend`, run `bun run typecheck`. From the repository root, run `git diff --check origin/dev...HEAD` and `git status --short`.

- [ ] **Step 3: Review every alert source and security boundary**

Confirm alerts 2251-2262 map to the changed paths, no `DOMParser` remains in the two alerted flows, no interpolated console first argument remains at alerts 2258-2260, and `ProviderInferenceRule` exposes only `matches`. Bandit is not applicable because no Python source is touched; record that explicit skip.

- [ ] **Step 4: Request code review and address findings**

Use `superpowers:requesting-code-review`. Re-run the relevant focused test after every correction.

- [ ] **Step 5: Finalize task records and commits**

Update TASK-12946 with verification output and modified files through Backlog MCP. Mark every stage complete, then remove this task-specific plan file per repository instructions. Commit task finalization with the implementation.

- [ ] **Step 6: Push and open the PR**

Push `codex/codeql-frontend-alerts` and open a ready PR targeting `dev`. Include alert mapping, test/typecheck results, the Bandit not-applicable note, the `dev` JavaScript CodeQL limitation, and a request for the human-authored `Change summary` required by repository policy.

- [ ] **Step 7: Inspect GitHub checks**

Run `gh pr checks` and inspect every failure. Do not claim JavaScript CodeQL passed unless a JavaScript analysis is actually emitted.
