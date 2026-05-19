# Flashcards Global Tag Suggestions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add global flashcard tag suggestions to the shared create and edit drawers so users can select existing tags across accessible flashcards in both the WebUI and extension while still typing new tags.

**Architecture:** Add one backend `GET /api/v1/flashcards/tags` endpoint backed by flashcard-keyword aggregation instead of card-list scans. Keep the current ManageTab page-scan suggestion hook unchanged, add a dedicated create/edit suggestion service and React Query hook with abort-signal support, and route both drawers through one shared `FlashcardTagPicker` component. Preserve extension parity by keeping the behavior in shared UI and adding one explicit extension-path parity check.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL DB abstraction, React, TypeScript, Ant Design, TanStack Query, Vitest, React Testing Library, Playwright

---

## File Structure

- `tldw_Server_API/app/api/v1/schemas/flashcards.py`
  Purpose: define response models for the new tag suggestions endpoint so the API contract is explicit and typed.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  Purpose: add a dedicated database method that aggregates active flashcard tags globally for the current user without inheriting ManageTab visibility defaults.
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
  Purpose: expose `GET /api/v1/flashcards/tags` before the dynamic `/{card_uuid}` alias route and return the new typed response.
- `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`
  Purpose: lock endpoint behavior for ordering, filtering, deleted rows, route precedence, and global visibility.
- `apps/packages/ui/src/services/flashcards.ts`
  Purpose: add typed client helpers for the new endpoint, including abort-signal support for typeahead requests.
- `apps/packages/ui/src/services/__tests__/flashcards.test.ts`
  Purpose: verify the service builds the correct `/api/v1/flashcards/tags` request and forwards abort signals.
- `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`
  Purpose: add a dedicated create/edit tag-suggestions query hook while preserving the existing ManageTab scan hook.
- `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx`
  Purpose: prove the new query hook uses the dedicated backend endpoint, respects `enabled`, and forwards the React Query abort signal.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardTagPicker.tsx`
  Purpose: implement the shared hybrid tag picker used by both create and edit drawers.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardTagPicker.test.tsx`
  Purpose: verify suggestion rendering, manual entry, whitespace trimming, case-insensitive dedupe, and fetch-failure fallback.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
  Purpose: replace the raw create-drawer tag field with the shared picker and provide stable test ids.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx`
  Purpose: replace the raw edit-drawer tag field with the shared picker and provide stable test ids.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.tags.test.tsx`
  Purpose: prove the create drawer can select a suggested tag and still submit a brand-new one.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx`
  Purpose: prove the edit drawer preserves existing tags and can append a suggested tag.
- `apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts`
  Purpose: add minimal locators and helpers for the create/edit tag picker flow in the shared workspace.
- `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
  Purpose: add one real user flow that reproduces the complaint and proves selecting existing tags works in the web shell.
- `apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts`
  Purpose: add an extension-path parity check that locks the extension wrapper to the shared `FlashcardsWorkspace`.

## Task 1: Add The Backend Tag Suggestions Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`

- [ ] **Step 1: Write the failing backend integration tests**

Add focused endpoint tests to `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py` that cover:

1. global visibility across a general deck and a workspace deck
2. ordering by usage count, then alphabetically
3. case-insensitive substring filtering with `q`
4. deleted flashcards, deleted keywords, and deleted decks being excluded
5. route precedence for `/api/v1/flashcards/tags`

Use the existing `flashcards_db` fixture when you need direct row soft-deletes.

```python
def test_flashcard_tag_suggestions_are_global_and_ranked(
    client_with_flashcards_db: TestClient,
    flashcards_db: CharactersRAGDB,
):
    general = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "General Deck"},
        headers=AUTH_HEADERS,
    ).json()
    workspace = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={"name": "Workspace Deck", "workspace_id": "workspace-77"},
        headers=AUTH_HEADERS,
    ).json()

    client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": general["id"], "front": "Q1", "back": "A1", "tags": ["biology", "alpha"]},
        headers=AUTH_HEADERS,
    )
    client_with_flashcards_db.post(
        "/api/v1/flashcards",
        json={"deck_id": workspace["id"], "front": "Q2", "back": "A2", "tags": ["biology", "zeta"]},
        headers=AUTH_HEADERS,
    )

    response = client_with_flashcards_db.get(
        "/api/v1/flashcards/tags",
        params={"q": "bio", "limit": 10},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["items"][0] == {"tag": "biology", "count": 2}
```

- [ ] **Step 2: Run the new backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "flashcard_tag_suggestions" -v
```

Expected: FAIL because `/api/v1/flashcards/tags` does not exist yet.

- [ ] **Step 3: Implement the minimal backend endpoint**

Add two new schema models in `tldw_Server_API/app/api/v1/schemas/flashcards.py`.

```python
class FlashcardTagSuggestionItem(BaseModel):
    tag: str
    count: int


class FlashcardTagSuggestionsResponse(BaseModel):
    items: list[FlashcardTagSuggestionItem] = Field(default_factory=list)
    count: int
```

Add a DB method in `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` that does not reuse `list_flashcards` visibility defaults.

```python
def list_flashcard_tag_suggestions(
    self,
    q: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    keyword_table = self._map_table_for_backend("keywords")
    where_clauses = [
        "f.deleted = 0" if self.backend_type != BackendType.POSTGRESQL else "f.deleted = FALSE",
        "kw.deleted = 0" if self.backend_type != BackendType.POSTGRESQL else "kw.deleted = FALSE",
        "(d.id IS NULL OR d.deleted = 0)" if self.backend_type != BackendType.POSTGRESQL else "(d.id IS NULL OR d.deleted = FALSE)",
    ]
    params: list[Any] = []
    if q and q.strip():
        where_clauses.append("LOWER(kw.keyword) LIKE ?")
        params.append(f"%{q.strip().lower()}%")

    query = f'''
        SELECT kw.keyword AS tag, COUNT(DISTINCT f.id) AS count
          FROM flashcards f
          LEFT JOIN decks d ON d.id = f.deck_id
          JOIN flashcard_keywords fk ON fk.card_id = f.id
          JOIN {keyword_table} kw ON kw.id = fk.keyword_id
         WHERE {" AND ".join(where_clauses)}
         GROUP BY kw.keyword
         ORDER BY count DESC, kw.keyword COLLATE NOCASE ASC
         LIMIT ?
    '''
    params.append(limit)
```

Then expose it in `tldw_Server_API/app/api/v1/endpoints/flashcards.py` before the later `@router.get("/{card_uuid}")` alias route.

```python
@router.get("/tags", response_model=FlashcardTagSuggestionsResponse)
def list_flashcard_tag_suggestions(
    q: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
):
    items = db.list_flashcard_tag_suggestions(q=q, limit=limit)
    return {"items": items, "count": len(items)}
```

- [ ] **Step 4: Re-run the backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "flashcard_tag_suggestions or get_flashcard_alias_path_returns_card" -v
```

Expected: PASS, proving the static `/tags` route works and the existing `/{card_uuid}` alias route still works.

- [ ] **Step 5: Run Bandit on the touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_flashcards_tag_suggestions.json
```

Expected: no new actionable findings in the touched code. If Bandit reports anything new in these edits, fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/flashcards.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py
git commit -m "feat: add flashcard tag suggestions endpoint"
```

## Task 2: Add The Shared Service And Dedicated Create/Edit Query Hook

**Files:**
- Modify: `apps/packages/ui/src/services/flashcards.ts`
- Modify: `apps/packages/ui/src/services/__tests__/flashcards.test.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`
- Create: `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx`

- [ ] **Step 1: Write the failing service and hook tests**

Extend `apps/packages/ui/src/services/__tests__/flashcards.test.ts` to cover the new request path and abort signal.

```ts
import { buildQuery } from "@/services/resource-client"

it("requests flashcard tag suggestions with query params and abort signal", async () => {
  const signal = new AbortController().signal
  vi.mocked(buildQuery).mockReturnValue("?q=bio&limit=15")

  await listFlashcardTagSuggestions({ q: "bio", limit: 15, signal })

  expect(mockBgRequest).toHaveBeenCalledWith(
    expect.objectContaining({
      path: "/api/v1/flashcards/tags?q=bio&limit=15",
      method: "GET",
      abortSignal: signal
    })
  )
})
```

Create `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx` and assert the new hook:

1. calls the dedicated service, not `listFlashcards`
2. forwards the React Query abort signal
3. stays disabled when `enabled: false`

```tsx
const { result } = renderHook(
  () => useGlobalFlashcardTagSuggestionsQuery("bio", { enabled: true, limit: 15 }),
  { wrapper: buildWrapper() }
)

await waitFor(() => {
  expect(result.current.data?.items[0]?.tag).toBe("biology")
})

expect(listFlashcardTagSuggestions).toHaveBeenCalledWith(
  expect.objectContaining({
    q: "bio",
    limit: 15,
    signal: expect.any(AbortSignal)
  })
)
```

- [ ] **Step 2: Run the new frontend tests to verify they fail**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/services/__tests__/flashcards.test.ts \
  ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx \
  --reporter=verbose
```

Expected: FAIL because the service helper and hook do not exist yet.

- [ ] **Step 3: Implement the minimal service and hook**

Add typed client helpers in `apps/packages/ui/src/services/flashcards.ts`.

```ts
export type FlashcardTagSuggestion = {
  tag: string
  count: number
}

export type FlashcardTagSuggestionsResponse = {
  items: FlashcardTagSuggestion[]
  count: number
}

export async function listFlashcardTagSuggestions(params?: {
  q?: string | null
  limit?: number
  signal?: AbortSignal
}): Promise<FlashcardTagSuggestionsResponse> {
  const query = buildQuery({
    q: params?.q,
    limit: params?.limit
  })
  const path = `/api/v1/flashcards/tags${query}` as AllowedPath
  return await bgRequest<FlashcardTagSuggestionsResponse, AllowedPath, "GET">({
    path,
    method: "GET",
    abortSignal: params?.signal
  })
}
```

Then add a dedicated create/edit hook in `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts` and leave the existing `useTagSuggestionsQuery` for ManageTab untouched.

```ts
export function useGlobalFlashcardTagSuggestionsQuery(
  search: string,
  options?: { enabled?: boolean; limit?: number }
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()
  const normalizedQuery = search.trim()
  const limit = options?.limit ?? 20

  return useQuery({
    queryKey: ["flashcards:tags:suggestions:global", normalizedQuery, limit],
    queryFn: ({ signal }) =>
      listFlashcardTagSuggestions({
        q: normalizedQuery || undefined,
        limit,
        signal
      }),
    enabled: (options?.enabled ?? flashcardsEnabled),
    staleTime: 30_000
  })
}
```

- [ ] **Step 4: Re-run the service and hook tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/services/__tests__/flashcards.test.ts \
  ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx \
  --reporter=verbose
```

Expected: PASS, with the hook proving that the new path is used and abort signals are forwarded.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services/flashcards.ts \
  apps/packages/ui/src/services/__tests__/flashcards.test.ts \
  apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx
git commit -m "feat: add shared flashcard tag suggestion query"
```

## Task 3: Build The Shared Hybrid `FlashcardTagPicker`

**Files:**
- Create: `apps/packages/ui/src/components/Flashcards/components/FlashcardTagPicker.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardTagPicker.test.tsx`

- [ ] **Step 1: Write the failing picker tests**

Create `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardTagPicker.test.tsx` and cover:

1. opening the picker shows backend suggestions
2. selecting an existing suggestion emits the normalized tag list
3. typing a new tag and pressing `Enter` still works
4. whitespace-only values are ignored
5. duplicate values collapse case-insensitively
6. fetch failure falls back to free typing

```tsx
render(
  <FlashcardTagPicker
    value={["Existing"]}
    onChange={onChange}
    active
    dataTestId="flashcards-create-tag-picker"
  />
)

fireEvent.mouseDown(screen.getByLabelText("Tags"))
expect(await screen.findByRole("option", { name: "biology" })).toBeInTheDocument()

fireEvent.click(screen.getByRole("option", { name: "biology" }))
expect(onChange).toHaveBeenLastCalledWith(["Existing", "biology"])
```

- [ ] **Step 2: Run the picker test to verify it fails**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardTagPicker.test.tsx --reporter=verbose
```

Expected: FAIL because `FlashcardTagPicker` does not exist yet.

- [ ] **Step 3: Implement the minimal shared picker**

Create `apps/packages/ui/src/components/Flashcards/components/FlashcardTagPicker.tsx` and keep the behavior narrow:

- own the dropdown-open state
- debounce the search text
- only enable the query when `active && dropdownOpen`
- normalize `onChange` values by trimming and deduping case-insensitively
- keep free typing intact when the query errors or returns nothing
- expose stable `data-testid` values for the wrapper and search input

```tsx
const normalizeTagValues = (values: string[]): string[] => {
  const seen = new Set<string>()
  const next: string[] = []
  for (const raw of values) {
    const trimmed = String(raw || "").trim()
    const key = trimmed.toLowerCase()
    if (!trimmed || seen.has(key)) continue
    seen.add(key)
    next.push(trimmed)
  }
  return next
}

export const FlashcardTagPicker = ({ value, onChange, active, dataTestId, ...props }) => {
  const [open, setOpen] = React.useState(false)
  const [search, setSearch] = React.useState("")
  const debouncedSearch = useDebouncedValue(search, 150)
  const suggestionsQuery = useGlobalFlashcardTagSuggestionsQuery(debouncedSearch, {
    enabled: active && open,
    limit: 20
  })

  return (
    <Select
      mode="tags"
      showSearch
      filterOption={false}
      open={open}
      onOpenChange={setOpen}
      searchValue={search}
      onSearch={setSearch}
      value={value}
      onChange={(next) => onChange?.(normalizeTagValues(next as string[]))}
      options={(suggestionsQuery.data?.items ?? []).map((item) => ({ value: item.tag, label: item.tag }))}
    />
  )
}
```

If no shared `useDebouncedValue` helper exists in this area, keep the debounce local inside this component with `useEffect` and `setTimeout`.

- [ ] **Step 4: Re-run the picker test**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardTagPicker.test.tsx --reporter=verbose
```

Expected: PASS, including the fallback path when the suggestion query fails.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Flashcards/components/FlashcardTagPicker.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardTagPicker.test.tsx
git commit -m "feat: add shared flashcard tag picker"
```

## Task 4: Wire The Create And Edit Drawers To The Shared Picker

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.tags.test.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx`

- [ ] **Step 1: Write the failing drawer tests**

Create one focused test per drawer.

Create drawer test expectations:

1. opening the create drawer and choosing a suggested tag submits that tag
2. typing a new tag still submits successfully

Edit drawer test expectations:

1. existing tags are shown on open
2. selecting a suggested tag appends it
3. whitespace-only edits are dropped before `onSave`

```tsx
render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

fireEvent.click(screen.getByText("Advanced options"))
fireEvent.mouseDown(screen.getByLabelText("Tags"))
fireEvent.click(await screen.findByRole("option", { name: "biology" }))
fireEvent.click(screen.getByRole("button", { name: "Create" }))

expect(mutateAsync).toHaveBeenCalledWith(
  expect.objectContaining({ tags: ["biology"] })
)
```

- [ ] **Step 2: Run the new drawer tests to verify they fail**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.tags.test.tsx \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx \
  --reporter=verbose
```

Expected: FAIL because both drawers still use raw `Select mode="tags"` fields with `open={false}`.

- [ ] **Step 3: Replace the raw tag fields with `FlashcardTagPicker`**

Update both drawer files to use the new component and pass stable test ids.

```tsx
<Form.Item
  name="tags"
  label={t("option:flashcards.tags", { defaultValue: "Tags" })}
  className="!mb-0"
>
  <FlashcardTagPicker
    active={open}
    dataTestId="flashcards-create-tag-picker"
    placeholder={t("option:flashcards.tagsPlaceholder", {
      defaultValue: "tag1, tag2"
    })}
  />
</Form.Item>
```

Use the edit-drawer equivalent:

```tsx
<FlashcardTagPicker
  active={open}
  dataTestId="flashcards-edit-tag-picker"
  placeholder={t("option:flashcards.tagsPlaceholder", {
    defaultValue: "Add tags..."
  })}
/>
```

Do not change the rest of the drawer submission flow. The existing create and update mutations already invalidate `flashcards:` queries globally.

- [ ] **Step 4: Re-run the drawer tests plus one existing safety test per drawer**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.tags.test.tsx \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.cloze-help.test.tsx \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.save.test.tsx \
  --reporter=verbose
```

Expected: PASS, proving the tag-field change did not regress basic create/edit validation.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx \
  apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.tags.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx
git commit -m "feat: wire flashcard drawers to shared tag picker"
```

## Task 5: Add Real User Coverage And Extension Parity Checks

**Files:**
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
- Create: `apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts`

- [ ] **Step 1: Write the failing web flow and extension parity tests**

Add a Playwright test in `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts` that:

1. uses `fetchWithApiKey` to create a deck plus two cards sharing an existing tag
2. navigates to Manage
3. opens the create drawer and selects the existing tag from suggestions
4. opens the edit drawer for one seeded card and adds another existing tag from suggestions

Also add a small source-level extension parity test in `apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts` that locks the extension route wrapper to the shared workspace component.

```ts
expect(optionFlashcardsSource).toMatch(/FlashcardsWorkspace/)
expect(optionFlashcardsSource).toMatch(/RouteErrorBoundary/)
```

- [ ] **Step 2: Run the new parity tests to verify they fail**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/extension/option-flashcards.shared-workspace.test.ts --reporter=verbose
npx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --reporter=line
```

Expected: the Vitest parity test may pass immediately if the wrapper is already correct; the new Playwright tag-selection flow should FAIL until the page object and UI locators are in place.

- [ ] **Step 3: Add the minimal page-object helpers and spec flow**

Extend `apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts` with only the helpers needed for this scenario:

- create drawer tag picker locator
- edit button locator by card UUID
- edit drawer tag picker locator
- helper to select a tag suggestion by visible option name

```ts
get createTagPicker(): Locator {
  return this.page.locator('[data-testid="flashcards-create-tag-picker"]')
}

get editTagPicker(): Locator {
  return this.page.locator('[data-testid="flashcards-edit-tag-picker"]')
}

getEditButton(cardUuid: string): Locator {
  return this.page.locator(`[data-testid="flashcard-edit-${cardUuid}"]`)
}
```

In the Playwright spec, use `fetchWithApiKey` for setup instead of clicking through unrelated UI.

```ts
skipIfServerUnavailable(serverInfo)

const deckResponse = await fetchWithApiKey(`${TEST_CONFIG.serverUrl}/api/v1/flashcards/decks`, TEST_CONFIG.apiKey, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ name: `Tag Deck ${Date.now()}` })
})
```

Then assert the create and edit flows finish without console errors and that the saved payload shows the expected tags via a backend list/read call.

- [ ] **Step 4: Run the targeted parity suite**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/extension/option-flashcards.shared-workspace.test.ts --reporter=verbose
npx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --reporter=line
```

Expected: PASS, with the web flow proving the reported bug is fixed and the extension wrapper still anchored to the shared workspace.

- [ ] **Step 5: Run the combined targeted verification**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "flashcard_tag_suggestions" -v
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/services/__tests__/flashcards.test.ts \
  ../packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.tag-suggestions.test.tsx \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardTagPicker.test.tsx \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.tags.test.tsx \
  ../packages/ui/src/components/Flashcards/components/__tests__/FlashcardEditDrawer.tags.test.tsx \
  __tests__/extension/option-flashcards.shared-workspace.test.ts \
  --reporter=verbose
npx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts --reporter=line
```

Expected: PASS across backend, shared UI, extension parity, and the user-facing web flow.

- [ ] **Step 6: Commit**

```bash
git add apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts \
  apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts \
  apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts
git commit -m "test: cover flashcard tag suggestion parity"
```

## Self-Review Checklist

- [ ] `GET /api/v1/flashcards/tags` is declared before the dynamic `/{card_uuid}` alias route.
- [ ] The backend query includes general-scope and workspace-scoped flashcards, but excludes deleted flashcards, deleted keywords, and cards attached to deleted decks.
- [ ] The create/edit hook does not replace the existing `useTagSuggestionsQuery` used by ManageTab.
- [ ] The picker only enables its suggestion query while the drawer is active and the picker is open.
- [ ] The service helper forwards `AbortSignal` so stale typeahead requests can be cancelled.
- [ ] The picker keeps free typing when suggestion loading fails.
- [ ] The create and edit drawers expose stable tag-picker test ids.
- [ ] The extension parity check still proves the extension route uses the shared workspace component.
