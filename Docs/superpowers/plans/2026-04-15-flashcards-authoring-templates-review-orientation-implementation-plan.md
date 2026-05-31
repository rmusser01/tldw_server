# Flashcards Authoring Templates And Review Orientation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add user-level flashcard authoring templates plus deck-level/back-first review orientation without changing flashcard scheduling semantics or splitting behavior between the WebUI and extension.

**Architecture:** Implement the feature in five slices. First extend deck contracts with a new study-orientation field and expose it everywhere decks are created or edited. Next add a first-class flashcard templates backend resource plus typed shared client hooks. Then add a dedicated Templates tab and reusable template-management components, followed by create-drawer apply/save flows that materialize a normal flashcard draft. Finish by wiring review rendering to the deck default plus session override and covering the shared UI, web E2E, and extension parity paths.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL DB abstraction in `ChaChaNotes_DB`, React, TypeScript, TanStack Query, Ant Design, Vitest, React Testing Library, Playwright, Bandit

---

## File Structure

- `tldw_Server_API/app/api/v1/schemas/flashcards.py`
  Purpose: extend `DeckCreate` / `DeckUpdate` / `Deck` with `review_prompt_side`, add typed flashcard-template request/response models, and keep API contracts explicit.
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
  Purpose: carry `review_prompt_side` through deck create/update routes and add flashcard-template CRUD endpoints under `/api/v1/flashcards/templates`.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  Purpose: persist `review_prompt_side` on decks, ensure the `flashcard_templates` table exists, and expose DB CRUD helpers for templates by stable `id`.
- `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`
  Purpose: extend deck API integration coverage for the new orientation field and keep existing flashcards endpoint behavior stable.
- `tldw_Server_API/tests/Flashcards/test_flashcards_templates_api.py`
  Purpose: add focused template-endpoint integration coverage for create/list/get/update/delete, rename, validation, and stable-id routing.
- `tldw_Server_API/tests/ChaChaNotesDB/test_flashcard_templates_db.py`
  Purpose: unit-test template persistence, rename uniqueness, soft delete, and placeholder validation at the DB abstraction layer.
- `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json`
  Purpose: update only if route-registry tests fail after adding template endpoints to the flashcards router.
- `apps/packages/ui/src/services/flashcards.ts`
  Purpose: add shared `FlashcardTemplate*` types plus service helpers for template CRUD and the new deck `review_prompt_side` contract.
- `apps/packages/ui/src/services/__tests__/flashcards.test.ts`
  Purpose: verify the shared client builds the correct template CRUD requests and deck payloads.
- `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`
  Purpose: add typed template queries/mutations and forward `review_prompt_side` through deck mutations.
- `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.templates.test.tsx`
  Purpose: prove the new template query/mutation hooks call the right services, use stable query keys, and invalidate caches correctly.
- `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx`
  Purpose: ensure create-deck flows can submit `review_prompt_side`.
- `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useDeckSchedulerMutation.test.tsx`
  Purpose: ensure deck-update flows preserve and update `review_prompt_side` in cached deck records.
- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
  Purpose: add `templates` tab parsing, keep Templates visible even with zero decks, and mount the new Templates tab.
- `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
  Purpose: lock tab labels, `?tab=templates` routing, and zero-deck behavior.
- `apps/packages/ui/src/components/Flashcards/tabs/index.ts`
  Purpose: export the new Templates tab.
- `apps/packages/ui/src/components/Flashcards/tabs/TemplatesTab.tsx`
  Purpose: implement the primary template-management surface inside Flashcards.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/TemplatesTab.test.tsx`
  Purpose: cover library listing, create/edit/delete flows, and empty-state behavior.
- `apps/packages/ui/src/components/Flashcards/components/DeckStudyDefaultsFields.tsx`
  Purpose: render the reusable `review_prompt_side` control once and share it across deck-creation and deck-editing flows.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/DeckStudyDefaultsFields.test.tsx`
  Purpose: verify the reusable orientation control renders and updates correctly.
- `apps/packages/ui/src/components/Flashcards/components/NewDeckConfigurationFields.tsx`
  Purpose: surface `review_prompt_side` in every shared deck-creation flow.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateForm.tsx`
  Purpose: provide the reusable template editor used by TemplatesTab.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateValueModal.tsx`
  Purpose: collect placeholder inputs when applying a template from the create drawer.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardSaveTemplateModal.tsx`
  Purpose: capture template metadata and placeholder configuration when saving a draft as a template.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
  Purpose: rename `Card template` to `Card model`, add apply/save-template actions, and materialize resolved drafts into the existing form.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx`
  Purpose: apply the terminology cleanup so edit surfaces say `Card model` rather than `Card template`.
- `apps/packages/ui/src/components/Flashcards/components/index.ts`
  Purpose: export any new reusable flashcard-template or deck-study-default components.
- `apps/packages/ui/src/components/Flashcards/utils/flashcard-template-resolution.ts`
  Purpose: centralize placeholder substitution, dead-token validation, and template-to-draft materialization logic.
- `apps/packages/ui/src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts`
  Purpose: lock placeholder substitution, missing-default fallback, and validation edge cases.
- `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.templates.test.tsx`
  Purpose: prove applying a template fills the draft correctly and saving a draft as a template uses the expected payload.
- `apps/packages/ui/src/components/Flashcards/tabs/SchedulerTab.tsx`
  Purpose: edit deck-level `review_prompt_side` alongside existing scheduler settings without pushing the field into scheduler JSON.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.test.tsx`
  Purpose: cover the new study-default control in the deck-policy workspace.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.editor.test.tsx`
  Purpose: ensure saving deck edits forwards `review_prompt_side` and preserves dirty-state handling.
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
  Purpose: derive the effective prompt side from deck default plus session override and render prompt/answer content accordingly.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx`
  Purpose: cover front-first/back-first rendering, cloze fallback, and session-override precedence.
- `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx`
  Purpose: ensure cram-mode queue progression is unchanged when orientation flips.
- `apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts`
  Purpose: add stable locators for the Templates tab and review-orientation controls.
- `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
  Purpose: add one real web flow for the Templates tab plus one review-orientation smoke path.
- `apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts`
  Purpose: keep the extension wrapper pointed at the shared workspace after the new Templates tab lands.
- `apps/extension/tests/e2e/flashcards-ux.spec.ts`
  Purpose: add one extension-path smoke test that exercises the shared flashcards workspace with the new template/review-orientation affordances.

## Task 1: Extend The Deck Study-Defaults Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `apps/packages/ui/src/services/flashcards.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useDeckSchedulerMutation.test.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/DeckStudyDefaultsFields.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/DeckStudyDefaultsFields.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/NewDeckConfigurationFields.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/SchedulerTab.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.editor.test.tsx`
- Modify: `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`

- [ ] **Step 1: Write the failing backend and shared-hook tests**

Add focused coverage for `review_prompt_side` in the existing deck API integration test file and the deck mutation hook tests.

```python
def test_create_deck_returns_review_prompt_side(client_with_flashcards_db: TestClient):
    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/decks",
        json={
            "name": "Biology Reverse Recall",
            "review_prompt_side": "back",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["review_prompt_side"] == "back"
```

```tsx
it("forwards review_prompt_side when creating a deck", async () => {
  await result.current.mutateAsync({
    name: "Biology Basics",
    review_prompt_side: "back",
    scheduler_settings: schedulerEnvelope,
  })

  expect(createDeck).toHaveBeenCalledWith(
    expect.objectContaining({ review_prompt_side: "back" })
  )
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "review_prompt_side" -v
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useDeckSchedulerMutation.test.tsx
```

Expected: FAIL because deck schemas, DB rows, and shared deck types do not yet carry `review_prompt_side`.

- [ ] **Step 3: Add the minimal backend and shared-type contract**

In `tldw_Server_API/app/api/v1/schemas/flashcards.py`, add a reusable literal and carry it through `DeckCreate`, `DeckUpdate`, and `Deck`.

```python
DeckReviewPromptSide = Literal["front", "back"]

class DeckCreate(BaseModel):
    ...
    review_prompt_side: DeckReviewPromptSide = "front"

class DeckUpdate(BaseModel):
    ...
    review_prompt_side: Optional[DeckReviewPromptSide] = None

class Deck(BaseModel):
    ...
    review_prompt_side: DeckReviewPromptSide = "front"
```

In `ChaChaNotes_DB.py`, extend the `decks` table and deck CRUD/select SQL so the field is stored directly on decks rather than inside scheduler JSON.

```python
def add_deck(..., review_prompt_side: str = "front") -> int:
    normalized_prompt_side = "back" if review_prompt_side == "back" else "front"
    ...
    "name, description, workspace_id, review_prompt_side, scheduler_settings_json, scheduler_type, ..."
```

Mirror the field in `apps/packages/ui/src/services/flashcards.ts`:

```ts
export type DeckReviewPromptSide = "front" | "back"

export type Deck = {
  ...
  review_prompt_side: DeckReviewPromptSide
}
```

- [ ] **Step 4: Surface the field in shared deck-edit/create controls**

Create a small `DeckStudyDefaultsFields.tsx` select component and reuse it in:

- `NewDeckConfigurationFields.tsx`
- `SchedulerTab.tsx`

Keep it intentionally narrow:

```tsx
<Select
  value={reviewPromptSide}
  onChange={(value) => onReviewPromptSideChange(value)}
  options={[
    { value: "front", label: "Front first" },
    { value: "back", label: "Back first" },
  ]}
  data-testid="deck-study-defaults-review-prompt-side"
/>
```

Do not put the field into `scheduler_settings`; keep it as a first-class deck property.

- [ ] **Step 5: Re-run tests and Bandit**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "review_prompt_side" -v
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useDeckSchedulerMutation.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/DeckStudyDefaultsFields.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.editor.test.tsx
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/schemas/flashcards.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  -f json -o /tmp/bandit_flashcards_review_prompt_side.json
```

Expected: PASS, with no new actionable Bandit findings.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/flashcards.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py \
  apps/packages/ui/src/services/flashcards.ts \
  apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useCreateDeckMutation.test.tsx \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useDeckSchedulerMutation.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/DeckStudyDefaultsFields.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/DeckStudyDefaultsFields.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/NewDeckConfigurationFields.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/SchedulerTab.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/SchedulerTab.editor.test.tsx
git commit -m "feat: add flashcard deck review orientation defaults"
```

## Task 2: Add The Flashcard Templates Backend And Shared Client Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Create: `tldw_Server_API/tests/Flashcards/test_flashcards_templates_api.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_flashcard_templates_db.py`
- Modify: `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json`
- Modify: `apps/packages/ui/src/services/flashcards.ts`
- Modify: `apps/packages/ui/src/services/__tests__/flashcards.test.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`
- Create: `apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.templates.test.tsx`

- [ ] **Step 1: Write the failing DB, API, and shared-client tests**

Create focused DB and API tests for the template contract.

```python
def test_flashcard_template_create_update_delete_round_trip(db_instance: CharactersRAGDB):
    template_id = db_instance.add_flashcard_template(
        name="Vocabulary Definition",
        model_type="basic",
        front_template="What does {{term}} mean?",
        back_template="{{definition}}",
        placeholder_definitions=[
            {"key": "term", "label": "Term", "required": True, "targets": ["front_template"]},
            {
                "key": "definition",
                "label": "Definition",
                "required": True,
                "targets": ["back_template"],
            },
        ],
    )

    template = db_instance.get_flashcard_template(template_id)
    assert template["name"] == "Vocabulary Definition"
```

```python
def test_flashcard_template_routes_use_stable_id(client_with_flashcards_db: TestClient):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards/templates",
        json={...},
        headers=AUTH_HEADERS,
    ).json()

    renamed = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/templates/{created['id']}",
        json={"name": "Renamed", "expected_version": created["version"]},
        headers=AUTH_HEADERS,
    )

    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Renamed"
```

```tsx
it("calls the flashcard templates list endpoint", async () => {
  await listFlashcardTemplates()
  expect(listSpy).toHaveBeenCalledWith({}, { abortSignal: undefined })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_flashcard_templates_db.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_templates_api.py -v
bunx vitest run \
  apps/packages/ui/src/services/__tests__/flashcards.test.ts \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.templates.test.tsx
```

Expected: FAIL because no flashcard template schemas, DB methods, endpoints, or shared client methods exist yet.

- [ ] **Step 3: Implement the backend template resource**

Add Pydantic models for:

- `FlashcardTemplatePlaceholderDefinition`
- `FlashcardTemplateCreate`
- `FlashcardTemplateUpdate`
- `FlashcardTemplate`
- `FlashcardTemplateListResponse`

Use a narrow field-target enum so dead/unsupported targets are rejected early.

```python
FlashcardTemplateFieldTarget = Literal["front_template", "back_template", "notes_template", "extra_template"]

class FlashcardTemplatePlaceholderDefinition(BaseModel):
    key: str
    label: str
    help_text: str | None = None
    default_value: str | None = None
    required: bool = False
    targets: list[FlashcardTemplateFieldTarget]
```

In `ChaChaNotes_DB.py`, add:

- table-ensure logic for `flashcard_templates`
- `add_flashcard_template`
- `list_flashcard_templates`
- `get_flashcard_template`
- `update_flashcard_template`
- `soft_delete_flashcard_template`

Persist templates by stable integer `id`, keep `name` unique among non-deleted templates, and validate token/definition consistency during create/update:

- every `{{token}}` in scaffold text must have a matching placeholder definition
- every definition must target at least one supported field
- every definition must be referenced somewhere in its targeted fields

In `flashcards.py`, expose:

- `POST /api/v1/flashcards/templates`
- `GET /api/v1/flashcards/templates`
- `GET /api/v1/flashcards/templates/{template_id}`
- `PATCH /api/v1/flashcards/templates/{template_id}`
- `DELETE /api/v1/flashcards/templates/{template_id}`

Register the new `/templates` routes before the existing catch-all `GET /api/v1/flashcards/{card_uuid}` route so template listing is not shadowed by the one-segment card lookup path.

- [ ] **Step 4: Implement the shared service and React Query contract**

In `apps/packages/ui/src/services/flashcards.ts`, add:

- `FlashcardTemplate` types
- `listFlashcardTemplates`
- `getFlashcardTemplate`
- `createFlashcardTemplate`
- `updateFlashcardTemplate`
- `deleteFlashcardTemplate`

Then add matching React Query helpers in `useFlashcardQueries.ts` with stable keys such as:

```ts
["flashcards:templates"]
["flashcards:templates", templateId]
["flashcards:template:create"]
["flashcards:template:update"]
["flashcards:template:delete"]
```

Invalidate the list/detail caches on create/update/delete.

- [ ] **Step 5: Re-run tests, update route snapshot only if needed, and run Bandit**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/ChaChaNotesDB/test_flashcard_templates_db.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_templates_api.py -v
bunx vitest run \
  apps/packages/ui/src/services/__tests__/flashcards.test.ts \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.templates.test.tsx
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/schemas/flashcards.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  -f json -o /tmp/bandit_flashcards_templates.json
```

If route-registry tests fail because the new flashcards template endpoints are discovered, update `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json` in the same commit after reviewing the diff.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/flashcards.py \
  tldw_Server_API/app/api/v1/endpoints/flashcards.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_templates_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_flashcard_templates_db.py \
  tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json \
  apps/packages/ui/src/services/flashcards.ts \
  apps/packages/ui/src/services/__tests__/flashcards.test.ts \
  apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts \
  apps/packages/ui/src/components/Flashcards/hooks/__tests__/useFlashcardQueries.templates.test.tsx
git commit -m "feat: add flashcard templates backend contract"
```

## Task 3: Add The Templates Tab And Management Surface

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/index.ts`
- Create: `apps/packages/ui/src/components/Flashcards/tabs/TemplatesTab.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/TemplatesTab.test.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateForm.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/index.ts`

- [ ] **Step 1: Write the failing routing and tab-surface tests**

Extend `FlashcardsManager.consistency.test.tsx` so it locks:

1. `?tab=templates` deep-link routing
2. Templates tab visibility when there are zero decks
3. Study / Manage / Import / Export / Templates labels

Create `TemplatesTab.test.tsx` for:

1. empty state
2. listing existing templates
3. opening a create form
4. editing a selected template
5. deleting a template via the delete mutation

```tsx
it("keeps the Templates tab visible when no decks exist", () => {
  mocks.decks = []
  window.history.replaceState({}, "", "/flashcards?tab=templates")

  render(<FlashcardsManager />)

  expect(screen.getByText("Templates")).toBeInTheDocument()
  expect(screen.getByTestId("mock-templates-tab")).toBeInTheDocument()
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/TemplatesTab.test.tsx
```

Expected: FAIL because the manager only knows `review`, `cards`, `importExport`, and `scheduler`.

- [ ] **Step 3: Implement the tab routing and library UI**

In `FlashcardsManager.tsx`:

- teach `parseInitialFlashcardsTab` about `templates`
- add the new tab item
- keep Templates visible even when there are zero decks
- do not change the existing no-deck startup default away from Import / Export

In `TemplatesTab.tsx`, build a focused CRUD surface backed by the template hooks. Start boring:

- searchable list on the left or top
- create button
- editor pane using `FlashcardTemplateForm`
- delete action

Keep it functional rather than ornamental.

```tsx
const templatesQuery = useFlashcardTemplatesQuery()
const createMutation = useCreateFlashcardTemplateMutation()
const updateMutation = useUpdateFlashcardTemplateMutation()
const deleteMutation = useDeleteFlashcardTemplateMutation()
```

- [ ] **Step 4: Re-run the manager and tab tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/TemplatesTab.test.tsx
```

Expected: PASS, with the Templates tab reachable directly and visible in the no-deck state.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx \
  apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/index.ts \
  apps/packages/ui/src/components/Flashcards/tabs/TemplatesTab.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/TemplatesTab.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateForm.tsx \
  apps/packages/ui/src/components/Flashcards/components/index.ts
git commit -m "feat: add flashcards templates tab"
```

## Task 4: Add Create-Drawer Template Apply And Save Flows

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateValueModal.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/components/FlashcardSaveTemplateModal.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/utils/flashcard-template-resolution.ts`
- Create: `apps/packages/ui/src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts`
- Create: `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.templates.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.cloze-help.test.tsx`

- [ ] **Step 1: Write the failing resolution and drawer tests**

Create a pure utility test for placeholder substitution and a drawer-level test that proves:

1. applying a template opens the value modal
2. submitting values fills deck, tags, model, and field content
3. saving a draft as a template calls the create-template mutation
4. edit drawer copy now says `Card model`

```ts
it("materializes template defaults and placeholder values into a flashcard draft", () => {
  expect(
    materializeFlashcardTemplateDraft(template, {
      term: "ATP",
      definition: "The cell's energy currency",
    })
  ).toEqual(
    expect.objectContaining({
      front: "What does ATP mean?",
      back: "The cell's energy currency",
      model_type: "basic",
    })
  )
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.templates.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.cloze-help.test.tsx
```

Expected: FAIL because the modals and resolution utility do not exist yet.

- [ ] **Step 3: Implement the resolution utility and modals**

In `flashcard-template-resolution.ts`, centralize:

- extracting required placeholder inputs from a template
- applying defaults
- substituting values into `front_template`, `back_template`, `notes_template`, `extra_template`
- returning a normal draft object suitable for `form.setFieldsValue(...)`

```ts
export function materializeFlashcardTemplateDraft(
  template: FlashcardTemplate,
  values: Record<string, string>
): Pick<FlashcardCreate, "deck_id" | "tags" | "model_type" | "front" | "back" | "notes" | "extra"> {
  ...
}
```

Keep the create drawer flow simple:

- `Apply template` opens the template picker / value modal
- successful apply writes into the existing form
- `Save as template` opens the save modal and calls the create mutation

Also rename any remaining `Card template` copy in create/edit drawers to `Card model`.

- [ ] **Step 4: Re-run the drawer and utility tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.templates.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.cloze-help.test.tsx
```

Expected: PASS, with the create drawer still submitting normal cards after a template is applied.

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx \
  apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx \
  apps/packages/ui/src/components/Flashcards/components/FlashcardTemplateValueModal.tsx \
  apps/packages/ui/src/components/Flashcards/components/FlashcardSaveTemplateModal.tsx \
  apps/packages/ui/src/components/Flashcards/utils/flashcard-template-resolution.ts \
  apps/packages/ui/src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.templates.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.cloze-help.test.tsx
git commit -m "feat: add flashcard template apply and save flows"
```

## Task 5: Add Review Orientation Rendering And Final Regression Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx`
- Create: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx`
- Modify: `apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts`
- Modify: `apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts`
- Modify: `apps/extension/tests/e2e/flashcards-ux.spec.ts`

- [ ] **Step 1: Write the failing review and end-to-end coverage**

Create `ReviewTab.orientation.test.tsx` to cover:

1. deck default `front` renders current behavior
2. deck default `back` swaps prompt/answer labels for non-cloze cards
3. cloze cards ignore `back`
4. session override takes precedence over the deck default
5. scope changes reset the override

```tsx
it("shows the back as the prompt when the deck default is back-first", async () => {
  renderReviewTab({ deckReviewPromptSide: "back", card: makeCard({ front: "ATP", back: "Energy currency" }) })

  expect(screen.getByText("Back")).toBeInTheDocument()
  expect(screen.getByText("Energy currency")).toBeInTheDocument()
  await user.click(screen.getByTestId("flashcards-review-show-answer"))
  expect(screen.getByText("ATP")).toBeInTheDocument()
})
```

Extend the web and extension smoke tests to verify:

- Templates tab is visible
- one review surface can flip prompt side

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx \
  apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend && npx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/extension && npx playwright test tests/e2e/flashcards-ux.spec.ts
```

Expected: FAIL because ReviewTab still hardcodes `front` as the prompt and the page object has no Templates/orientation locators.

- [ ] **Step 3: Implement effective prompt-side rendering**

In `ReviewTab.tsx`, derive an `effectivePromptSide` from:

1. session override
2. selected deck `review_prompt_side`
3. `front`

Then map prompt/answer rendering through that value while keeping the existing review mutation path untouched.

Add a small session-scoped control near the existing study-mode controls so users can temporarily switch between `Front first` and `Back first` without editing the deck. Reset that local override inside the existing `reviewScopeKey` lifecycle so deck changes, cram-tag changes, due/cram mode changes, and review-override-card entry all start from the deck default again.

```tsx
const effectivePromptSide =
  activeCard?.model_type === "cloze"
    ? "front"
    : sessionReviewPromptSide ?? selectedDeck?.review_prompt_side ?? "front"

const promptContent = effectivePromptSide === "back" ? activeCard.back : activeCard.front
const answerContent = effectivePromptSide === "back" ? activeCard.front : activeCard.back
```

Use the existing `reviewScopeKey` effect to reset the session override when the scope changes. Do not mutate the card record or send orientation in the review payload.

- [ ] **Step 4: Re-run the orientation, web, and extension tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx \
  apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend && npx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/extension && npx playwright test tests/e2e/flashcards-ux.spec.ts
```

Expected: PASS, with no queue-progression regressions and the shared workspace still rendering correctly in the extension wrapper.

- [ ] **Step 5: Run the final focused verification sweep**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py \
  tldw_Server_API/tests/Flashcards/test_flashcards_templates_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_flashcard_templates_db.py -v
bunx vitest run \
  apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/TemplatesTab.test.tsx \
  apps/packages/ui/src/components/Flashcards/components/__tests__/FlashcardCreateDrawer.templates.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx
```

Expected: PASS. If any unrelated failures appear, stop and separate genuine regressions from pre-existing breakage before proceeding.

- [ ] **Step 6: Commit**

```bash
git add \
  apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx \
  apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.cram-mode.test.tsx \
  apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts \
  apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts \
  apps/tldw-frontend/__tests__/extension/option-flashcards.shared-workspace.test.ts \
  apps/extension/tests/e2e/flashcards-ux.spec.ts
git commit -m "feat: add flashcard review orientation"
```

## Self-Review Notes

- Keep `review_prompt_side` out of `scheduler_settings_json`. If you find yourself needing to thread it through scheduler helpers, you are coupling the wrong abstractions.
- Keep template identity stable by `id`; do not route UI updates or deletes through `name`.
- Do not add template apply/save behavior to the edit drawer in v1. Only rename terminology there.
- If the Templates tab starts to sprawl, split list/editor subcomponents before widening behavior.
- Preserve shared UI first. Avoid web-only shortcuts that bypass `apps/packages/ui`.
