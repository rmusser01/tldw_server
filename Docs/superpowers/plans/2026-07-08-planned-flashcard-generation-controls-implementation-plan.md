# Planned Flashcard Generation Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact-count planned flashcard generation for basic, reverse, cloze, and true/false-style drafts while preserving the existing single-type generation API.

**Architecture:** Treat `card_plan` as an API-boundary feature. Backend validation normalizes either legacy single-type requests or planned requests, the workflow adapter prompts for `generation_type`, and the `/flashcards` Generate panel exposes an Advanced mix toggle that sends derived totals. Storage and scheduling remain unchanged.

**Tech Stack:** FastAPI, Pydantic v2, pytest, React, TypeScript, Ant Design, Vitest.

---

## References

- Spec: `Docs/superpowers/specs/2026-07-08-planned-flashcard-generation-controls-design.md`
- Backlog task: `backlog/tasks/task-12170 - Add-planned-flashcard-generation-types-and-counts.md`
- Note: `TASK-12170` has stale duplicate files on `origin/dev`; update the exact planned-flashcard task path above.

## File Structure

- Modify `tldw_Server_API/app/api/v1/schemas/flashcards.py`: request/response schema, plan row model, response-only `generation_type`.
- Modify `tldw_Server_API/app/api/v1/endpoints/flashcards.py`: pass `card_plan`, validate generated counts, normalize true/false to stored basic cards, test-mode output.
- Modify `tldw_Server_API/app/core/Workflows/adapters/content/_config.py`: align workflow config with actual `card_type`/`card_plan` inputs.
- Modify `tldw_Server_API/app/core/Workflows/adapters/content/generation.py`: build legacy/planned prompt instructions and preserve `generation_type`.
- Modify `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`: endpoint validation and mixed-plan behavior tests.
- Modify `tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py`: adapter prompt/output tests.
- Modify `apps/packages/ui/src/services/flashcards.ts`: typed plan request and response-only `generation_type`.
- Modify `apps/packages/ui/src/services/__tests__/flashcards.test.ts`: request body test for `card_plan`.
- Modify `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`: camel-case hook params to snake-case service payload.
- Modify `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/shared.ts`: generated draft `generation_type` normalization.
- Modify `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`: Advanced mix toggle, derived total, true/false preview label.
- Modify `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx`: advanced mix payload and preview/save assertions.
- Modify `backlog/tasks/task-12170 - Add-planned-flashcard-generation-types-and-counts.md`: record plan, verification, and final summary as work progresses.

### Task 1: Backend Schema Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- Test: `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`

- [x] **Step 1: Write failing schema/endpoint validation tests**

Add focused tests near the existing generate endpoint tests:

```python
def test_generate_flashcards_accepts_mixed_card_plan(client_with_flashcards_db, monkeypatch):
    async def fake_generate_adapter(config, context):
        assert config["num_cards"] == 4
        assert config["card_plan"] == [
            {"card_type": "basic", "count": 1},
            {"card_type": "basic_reverse", "count": 1},
            {"card_type": "cloze", "count": 1},
            {"card_type": "true_false", "count": 1},
        ]
        return {
            "flashcards": [
                {"front": "Q1", "back": "A1", "generation_type": "basic", "model_type": "basic"},
                {"front": "Q2", "back": "A2", "generation_type": "basic_reverse", "model_type": "basic_reverse"},
                {"front": "{{c1::Q3}}", "back": "Q3", "generation_type": "cloze", "model_type": "cloze"},
                {"front": "True or false: Q4", "back": "False. A4", "generation_type": "true_false", "model_type": "basic"},
            ],
            "count": 4,
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.run_flashcard_generate_adapter",
        fake_generate_adapter,
    )

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/generate",
        json={
            "text": "Cell biology",
            "num_cards": 4,
            "card_plan": [
                {"card_type": "basic", "count": 1},
                {"card_type": "basic_reverse", "count": 1},
                {"card_type": "cloze", "count": 1},
                {"card_type": "true_false", "count": 1},
            ],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    cards = response.json()["flashcards"]
    assert [card["generation_type"] for card in cards] == [
        "basic",
        "basic_reverse",
        "cloze",
        "true_false",
    ]
    assert cards[-1]["model_type"] == "basic"
```

Add parameterized invalid request tests:

```python
@pytest.mark.parametrize(
    "body",
    [
        {"text": "x", "num_cards": 2, "card_type": "basic", "card_plan": [{"card_type": "basic", "count": 2}]},
        {"text": "x", "card_plan": [{"card_type": "basic", "count": 1}]},
        {"text": "x", "num_cards": 2, "card_plan": [{"card_type": "basic", "count": 1}]},
        {"text": "x", "num_cards": 1, "card_plan": []},
        {"text": "x", "num_cards": 2, "card_plan": [{"card_type": "basic", "count": 1}, {"card_type": "basic", "count": 1}]},
        {"text": "x", "num_cards": 1, "card_plan": [{"card_type": "basic", "count": 0}]},
        {"text": "x", "num_cards": 1, "card_plan": [{"card_type": "made_up", "count": 1}]},
        {"text": "x", "num_cards": 1, "card_plan": [{"card_type": "basic", "count": 1, "extra": True}]},
        {"text": "x", "num_cards": 1, "card_type": "true_false"},
    ],
)
def test_generate_flashcards_rejects_invalid_card_plan_requests(client_with_flashcards_db, body):
    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/generate",
        json=body,
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 422
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "generate_flashcards" -v
```

Expected: new plan tests fail because `card_plan` and `generation_type` do not exist yet.

- [x] **Step 3: Add schema models and validators**

In `flashcards.py`, import `ConfigDict` and `model_validator` if not already present. Add:

```python
FlashcardCardType = Literal["basic", "basic_reverse", "cloze"]
FlashcardGenerationType = Literal["basic", "basic_reverse", "cloze", "true_false"]


class FlashcardPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_type: FlashcardGenerationType
    count: int = Field(..., ge=1, le=100)
```

Update `FlashcardGenerateRequest`:

```python
class FlashcardGenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Source text to generate flashcards from")
    num_cards: Optional[int] = Field(None, ge=1, le=100, description="Requested number of generated cards")
    card_type: Optional[FlashcardCardType] = Field(None)
    card_plan: Optional[list[FlashcardPlanItem]] = None
    difficulty: Literal["easy", "medium", "hard", "mixed"] = Field("mixed")
    focus_topics: list[str] = Field(default_factory=list)
    provider: Optional[str] = Field(None, description="Optional LLM provider override")
    model: Optional[str] = Field(None, description="Optional LLM model override")

    @model_validator(mode="before")
    @classmethod
    def _validate_raw_generation_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        has_plan = value.get("card_plan") is not None
        if has_plan and value.get("card_type") is not None:
            raise ValueError("card_plan and card_type are mutually exclusive")
        if has_plan and "num_cards" not in value:
            raise ValueError("num_cards is required when card_plan is present")
        return value

    @model_validator(mode="after")
    def _validate_card_plan(self) -> "FlashcardGenerateRequest":
        if self.card_plan is not None:
            if len(self.card_plan) == 0:
                raise ValueError("card_plan cannot be empty")
            seen: set[str] = set()
            for row in self.card_plan:
                if row.card_type in seen:
                    raise ValueError("card_plan cannot contain duplicate card_type rows")
                seen.add(row.card_type)
            total = sum(row.count for row in self.card_plan)
            if self.num_cards != total:
                raise ValueError("num_cards must equal the sum of card_plan counts")
        else:
            self.num_cards = self.num_cards or 10
            self.card_type = self.card_type or "basic"
        return self
```

Update `GeneratedFlashcard`:

```python
class GeneratedFlashcard(BaseModel):
    front: str
    back: str
    tags: list[str] = Field(default_factory=list)
    model_type: FlashcardCardType = Field("basic")
    generation_type: Optional[FlashcardGenerationType] = None
    notes: Optional[str] = None
    extra: Optional[str] = None
```

- [x] **Step 4: Run schema tests**

Run the same pytest command. Expected: validation tests pass or fail only because endpoint normalization has not yet been updated.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/flashcards.py tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py
git commit -m "feat: add flashcard generation plan schema"
```

### Task 2: Endpoint Plan Validation And Test Mode

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- Test: `tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py`

- [x] **Step 1: Write failing endpoint behavior tests**

Add exact-count failure coverage:

```python
def test_generate_flashcards_rejects_planned_output_count_mismatch(client_with_flashcards_db, monkeypatch):
    async def fake_generate_adapter(config, context):
        return {
            "flashcards": [
                {"front": "Q1", "back": "A1", "generation_type": "basic", "model_type": "basic"},
            ],
            "count": 1,
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.run_flashcard_generate_adapter",
        fake_generate_adapter,
    )

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/generate",
        json={
            "text": "Cell biology",
            "num_cards": 2,
            "card_plan": [
                {"card_type": "basic", "count": 1},
                {"card_type": "cloze", "count": 1},
            ],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert "Generated flashcards did not satisfy requested card plan" in response.json()["detail"]
```

Add missing/invalid `generation_type` coverage:

```python
@pytest.mark.parametrize(
    "raw_card",
    [
        {"front": "Q1", "back": "A1", "model_type": "basic"},
        {"front": "Q1", "back": "A1", "generation_type": "made_up", "model_type": "basic"},
    ],
)
def test_generate_flashcards_rejects_planned_output_without_valid_generation_type(
    client_with_flashcards_db,
    monkeypatch,
    raw_card,
):
    async def fake_generate_adapter(config, context):
        return {"flashcards": [raw_card], "count": 1}

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.run_flashcard_generate_adapter",
        fake_generate_adapter,
    )

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/generate",
        json={
            "text": "Cell biology",
            "num_cards": 1,
            "card_plan": [{"card_type": "basic", "count": 1}],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert "Generated flashcards did not satisfy requested card plan" in response.json()["detail"]
```

Add test-mode coverage:

```python
def test_generate_flashcards_test_mode_supports_card_plan(client_with_flashcards_db, monkeypatch):
    async def fake_generate_adapter(config, context):
        return {"error": "LLM provider is required", "flashcards": [], "count": 0}

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.flashcards.run_flashcard_generate_adapter",
        fake_generate_adapter,
    )

    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/generate",
        json={
            "text": "Cell biology",
            "num_cards": 2,
            "card_plan": [
                {"card_type": "basic", "count": 1},
                {"card_type": "true_false", "count": 1},
            ],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    cards = response.json()["flashcards"]
    assert [card["generation_type"] for card in cards] == ["basic", "true_false"]
    assert cards[1]["model_type"] == "basic"
```

- [x] **Step 2: Run endpoint tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "generate_flashcards" -v
```

Expected: new tests fail until endpoint passes and validates plans.

- [x] **Step 3: Add small endpoint helpers**

Near `_build_test_mode_flashcards`, add:

```python
def _get_flashcard_generation_plan(payload: FlashcardGenerateRequest) -> list[dict[str, Any]]:
    if payload.card_plan is not None:
        return [{"card_type": item.card_type, "count": item.count} for item in payload.card_plan]
    return [{"card_type": payload.card_type or "basic", "count": int(payload.num_cards or 10)}]


def _expected_flashcard_plan_counts(payload: FlashcardGenerateRequest) -> dict[str, int]:
    return {row["card_type"]: int(row["count"]) for row in _get_flashcard_generation_plan(payload)}


def _storage_model_for_generation_type(generation_type: str) -> str:
    return "basic" if generation_type == "true_false" else generation_type
```

Update `_build_test_mode_flashcards` to iterate over `_get_flashcard_generation_plan(payload)` and emit `generation_type`. For `true_false`, use:

```python
front = f"True or false: Study point {index + 1} is covered by this source."
back = f"True. {normalized_text}"
model_type = "basic"
```

Add one normalizer helper and use it for both adapter output and test-mode fallback:

```python
def _normalize_generated_flashcards(
    raw_flashcards: Any,
    payload: FlashcardGenerateRequest,
) -> list[dict[str, Any]]:
    ...
```

For planned requests, this helper must reject missing or invalid `generation_type` with HTTP 400. It must not coerce malformed planned output to `basic`. For legacy requests only, it may fall back to `payload.card_type or "basic"` to preserve current compatibility.

- [x] **Step 4: Pass plan data into the adapter**

In `generate_flashcards`, call the adapter with:

```python
"num_cards": int(payload.num_cards or 10),
"card_type": payload.card_type or "basic",
"card_plan": _get_flashcard_generation_plan(payload) if payload.card_plan else None,
```

- [x] **Step 5: Route test-mode fallback through the same normalizer**

Change the current test-mode error branch from early return:

```python
if is_test_mode() and _should_return_test_mode_flashcards(error):
    generated_cards = _build_test_mode_flashcards(payload)
    return {"flashcards": generated_cards, "count": len(generated_cards)}
```

to setting `raw_flashcards` and continuing through `_normalize_generated_flashcards(...)` and planned-count validation:

```python
raw_flashcards = result.get("flashcards") if isinstance(result, dict) else []
if error:
    if is_test_mode() and _should_return_test_mode_flashcards(error):
        raw_flashcards = _build_test_mode_flashcards(payload)
    else:
        raise HTTPException(status_code=400, detail=str(error))

generated_cards = _normalize_generated_flashcards(raw_flashcards, payload)
```

This keeps deterministic test mode on the same strict path as real planned generation.

- [x] **Step 6: Preserve `generation_type` and validate planned output**

While normalizing `raw_flashcards`, derive:

```python
valid_generation_types = ("basic", "basic_reverse", "cloze", "true_false")
if payload.card_plan:
    raw_generation_type = str(raw.get("generation_type") or "").lower()
    if raw_generation_type not in valid_generation_types:
        raise HTTPException(
            status_code=400,
            detail="Generated flashcards did not satisfy requested card plan: missing or invalid generation_type",
        )
else:
    raw_generation_type = str(raw.get("generation_type") or raw.get("model_type") or payload.card_type or "basic").lower()
    if raw_generation_type not in valid_generation_types:
        raw_generation_type = payload.card_type or "basic"
model_type = _storage_model_for_generation_type(raw_generation_type)
if model_type not in ("basic", "basic_reverse", "cloze"):
    model_type = "basic"
```

Include `generation_type` in the response card.

After normalizing all cards, if `payload.card_plan` is present, compare actual counts by `generation_type` to `_expected_flashcard_plan_counts(payload)`. On mismatch:

```python
raise HTTPException(
    status_code=400,
    detail=(
        "Generated flashcards did not satisfy requested card plan: "
        f"{card_type} expected {expected}, got {actual}"
    ),
)
```

- [x] **Step 7: Run endpoint tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "generate_flashcards" -v
```

Expected: all generate endpoint tests pass.

- [x] **Step 8: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py
git commit -m "feat: validate planned flashcard generation"
```

### Task 3: Workflow Adapter Planned Generation

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/adapters/content/_config.py`
- Modify: `tldw_Server_API/app/core/Workflows/adapters/content/generation.py`
- Test: `tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py`

- [ ] **Step 1: Write failing adapter tests**

Add to `TestFlashcardGenerateAdapter`:

```python
@pytest.mark.asyncio
async def test_flashcard_generate_planned_prompt_and_generation_types(self, base_context, sample_long_text):
    from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter

    mock_flashcards = json.dumps([
        {"front": "Q1", "back": "A1", "generation_type": "basic", "tags": []},
        {"front": "True or false: Q2", "back": "False. A2", "generation_type": "true_false", "tags": []},
    ])
    mock_response = mock_chat_response(mock_flashcards)

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.content.generation.perform_chat_api_call_async",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_call:
        result = await run_flashcard_generate_adapter(
            {
                "text": sample_long_text,
                "num_cards": 2,
                "card_plan": [
                    {"card_type": "basic", "count": 1},
                    {"card_type": "true_false", "count": 1},
                ],
            },
            base_context,
        )

    system_message = mock_call.call_args.kwargs["system_message"]
    assert "basic: 1" in system_message
    assert "true_false: 1" in system_message
    assert "generation_type" in system_message
    assert result["flashcards"][0]["generation_type"] == "basic"
    assert result["flashcards"][1]["generation_type"] == "true_false"
    assert result["flashcards"][1]["model_type"] == "basic"
```

- [ ] **Step 2: Run adapter tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py -k "FlashcardGenerateAdapter" -v
```

Expected: planned prompt test fails because adapter ignores `card_plan`.

- [ ] **Step 3: Update workflow config**

In `FlashcardGenerateConfig`, replace or supplement `format` with:

```python
card_type: Literal["basic", "basic_reverse", "cloze"] = Field("basic", description="Legacy single card type")
card_plan: list[dict[str, Any]] | None = Field(None, description="Exact flashcard generation plan")
```

Do not add a new dependency or separate config module.

- [ ] **Step 4: Update adapter prompt construction**

In `run_flashcard_generate_adapter`, normalize:

```python
raw_plan = config.get("card_plan")
if isinstance(raw_plan, list) and raw_plan:
    plan = [
        {"card_type": str(item.get("card_type")), "count": int(item.get("count", 0))}
        for item in raw_plan
        if isinstance(item, dict)
    ]
else:
    plan = [{"card_type": card_type, "count": num_cards}]
```

For planned requests, add exact instructions:

```python
plan_lines = "\n".join(f"- {row['card_type']}: {row['count']}" for row in plan)
system_prompt = (
    f"Generate {num_cards} flashcards with these exact counts:\n"
    f"{plan_lines}\n"
    "Each JSON object must include front, back, tags, and generation_type.\n"
    "generation_type must be one of basic, basic_reverse, cloze, true_false.\n"
    "For true_false, write the front as 'True or false: ...' and the back as the answer plus a short explanation.\n"
    f"{difficulty_hints.get(difficulty, difficulty_hints['medium'])}{topics_hint}\n"
    'Return JSON array: [{"front": "Q", "back": "A", "tags": [], "generation_type": "basic"}]'
)
```

For legacy requests, keep current behavior but include `generation_type` in normalized output.

- [ ] **Step 5: Preserve generation type and storage model**

After parsing JSON:

```python
planned_request = bool(config.get("card_plan"))
for card in flashcards:
    generation_type = str(card.get("generation_type") or "").lower()
    if planned_request:
        if generation_type in ("basic", "basic_reverse", "cloze", "true_false"):
            card["generation_type"] = generation_type
            card["model_type"] = "basic" if generation_type == "true_false" else generation_type
        else:
            card.pop("generation_type", None)
    else:
        if generation_type not in ("basic", "basic_reverse", "cloze", "true_false"):
            generation_type = card_type
        card["generation_type"] = generation_type
        card["model_type"] = "basic" if generation_type == "true_false" else generation_type
```

Do not coerce missing or invalid planned `generation_type` to `basic`; the endpoint validator must be able to reject malformed planned output.

- [ ] **Step 6: Run adapter tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py -k "FlashcardGenerateAdapter" -v
```

Expected: all flashcard adapter tests pass.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Workflows/adapters/content/_config.py tldw_Server_API/app/core/Workflows/adapters/content/generation.py tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py
git commit -m "feat: prompt planned flashcard generation"
```

### Task 4: Frontend Service And Draft Plumbing

**Files:**
- Modify: `apps/packages/ui/src/services/flashcards.ts`
- Modify: `apps/packages/ui/src/services/__tests__/flashcards.test.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts`
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/shared.ts`

- [ ] **Step 1: Write failing service and normalization tests**

In `apps/packages/ui/src/services/__tests__/flashcards.test.ts`, add:

```ts
it("sends planned flashcard generation payloads unchanged", async () => {
  await generateFlashcards({
    text: "ATP powers the cell.",
    num_cards: 3,
    card_plan: [
      { card_type: "basic", count: 2 },
      { card_type: "true_false", count: 1 }
    ]
  })

  expect(mockBgRequest).toHaveBeenCalledWith(
    expect.objectContaining({
      body: expect.objectContaining({
        num_cards: 3,
        card_plan: [
          { card_type: "basic", count: 2 },
          { card_type: "true_false", count: 1 }
        ]
      })
    })
  )
})
```

If no direct test exists for `normalizeGeneratedCards`, add one in the nearest existing test file or export-free UI test in Task 5. Keep it to one assertion: `generation_type: "true_false"` survives normalization.

- [ ] **Step 2: Run frontend service test to verify failure**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/flashcards.test.ts
```

Expected: TypeScript fails or test fails because `card_plan` is not typed yet.

- [ ] **Step 3: Add shared TypeScript types**

In `flashcards.ts`, add:

```ts
export type FlashcardGenerationType = "basic" | "basic_reverse" | "cloze" | "true_false"

export type FlashcardPlanItem = {
  card_type: FlashcardGenerationType
  count: number
}
```

Update:

```ts
export type FlashcardGeneratedDraft = {
  front: string
  back: string
  tags?: string[] | null
  model_type?: "basic" | "basic_reverse" | "cloze"
  generation_type?: FlashcardGenerationType | null
  notes?: string | null
  extra?: string | null
}

export type FlashcardsGenerateRequest = {
  text: string
  num_cards?: number
  card_type?: "basic" | "basic_reverse" | "cloze"
  card_plan?: FlashcardPlanItem[] | null
  difficulty?: "easy" | "medium" | "hard" | "mixed"
  focus_topics?: string[] | null
  provider?: string | null
  model?: string | null
}
```

- [ ] **Step 4: Update generate hook params**

In `useFlashcardQueries.ts`, import `type FlashcardPlanItem` and add `cardPlan?: FlashcardPlanItem[]` to mutation params. Pass `card_plan: params.cardPlan`.

- [ ] **Step 5: Preserve response-only generation type in drafts**

In `shared.ts`, import `type FlashcardGenerationType`, add `generation_type?: FlashcardGenerationType | null` to `GeneratedCardDraft`, and normalize:

```ts
const generationTypeRaw = String(item.generation_type || model_type).toLowerCase()
const generation_type: FlashcardGenerationType =
  generationTypeRaw === "true_false"
    ? "true_false"
    : generationTypeRaw === "cloze"
      ? "cloze"
      : generationTypeRaw === "basic_reverse"
        ? "basic_reverse"
        : "basic"
```

Set `generation_type` on the draft.

- [ ] **Step 6: Run service test**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/flashcards.test.ts
```

Expected: service test passes.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/services/flashcards.ts apps/packages/ui/src/services/__tests__/flashcards.test.ts apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts apps/packages/ui/src/components/Flashcards/tabs/ImportExport/shared.ts
git commit -m "feat: plumb flashcard generation plans in ui"
```

### Task 5: `/flashcards` Advanced Mix UI

**Files:**
- Modify: `apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx`
- Test: `apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx`

- [ ] **Step 1: Write failing UI tests**

Add tests near the existing generate preview test:

```ts
it("sends an advanced flashcard mix with derived total", async () => {
  const generateMutateAsync = vi.fn().mockResolvedValue({
    flashcards: [{ front: "TF front", back: "False. Because.", tags: [], model_type: "basic", generation_type: "true_false" }],
    count: 1
  })
  vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
    mutateAsync: generateMutateAsync,
    isPending: false
  } as any)

  render(<ImportExportTab />)

  fireEvent.change(screen.getByTestId("flashcards-generate-text"), {
    target: { value: "Advanced mix source" }
  })
  fireEvent.click(screen.getByTestId("flashcards-generate-advanced-toggle"))
  fireEvent.change(screen.getByTestId("flashcards-generate-plan-basic-count"), {
    target: { value: "2" }
  })
  fireEvent.change(screen.getByTestId("flashcards-generate-plan-basic-reverse-count"), {
    target: { value: "1" }
  })
  fireEvent.change(screen.getByTestId("flashcards-generate-plan-cloze-count"), {
    target: { value: "1" }
  })
  fireEvent.change(screen.getByTestId("flashcards-generate-plan-true-false-count"), {
    target: { value: "1" }
  })
  fireEvent.click(screen.getByTestId("flashcards-generate-button"))

  await waitFor(() => expect(generateMutateAsync).toHaveBeenCalledTimes(1))
  expect(generateMutateAsync).toHaveBeenCalledWith(
    expect.objectContaining({
      numCards: 5,
      cardPlan: [
        { card_type: "basic", count: 2 },
        { card_type: "basic_reverse", count: 1 },
        { card_type: "cloze", count: 1 },
        { card_type: "true_false", count: 1 }
      ]
    })
  )
  expect(screen.getByText("True/False")).toBeInTheDocument()
})
```

Add save stripping test by extending the existing save-flow test with a generated true/false draft:

```ts
expect(createCardMutateAsync.mock.calls[0][0]).not.toHaveProperty("generation_type")
```

Add total and disabled-state coverage:

```ts
it("updates advanced mix total and disables generation outside allowed totals", async () => {
  const generateMutateAsync = vi.fn()
  vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
    mutateAsync: generateMutateAsync,
    isPending: false
  } as any)

  render(<ImportExportTab />)

  fireEvent.change(screen.getByTestId("flashcards-generate-text"), {
    target: { value: "Advanced mix source" }
  })
  fireEvent.click(screen.getByTestId("flashcards-generate-advanced-toggle"))

  expect(screen.getByTestId("flashcards-generate-plan-total")).toHaveTextContent("10")

  for (const testId of [
    "flashcards-generate-plan-basic-count",
    "flashcards-generate-plan-basic-reverse-count",
    "flashcards-generate-plan-cloze-count",
    "flashcards-generate-plan-true-false-count"
  ]) {
    fireEvent.change(screen.getByTestId(testId), { target: { value: "0" } })
  }
  expect(screen.getByTestId("flashcards-generate-plan-total")).toHaveTextContent("0")
  expect(screen.getByTestId("flashcards-generate-button")).toBeDisabled()

  fireEvent.change(screen.getByTestId("flashcards-generate-plan-basic-count"), {
    target: { value: "101" }
  })
  expect(screen.getByTestId("flashcards-generate-plan-total")).toHaveTextContent("101")
  expect(screen.getByTestId("flashcards-generate-button")).toBeDisabled()
})
```

- [ ] **Step 2: Run UI tests to verify failure**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
```

Expected: tests fail because the Advanced mix toggle and plan rows do not exist.

- [ ] **Step 3: Add minimal advanced mix state**

In `GeneratePanel.tsx`, import `Switch` from `antd` and `type FlashcardPlanItem` from `@/services/flashcards`.

Add:

```ts
type FlashcardPlanDraft = FlashcardPlanItem & {
  enabled: boolean
}

const DEFAULT_CARD_PLAN: FlashcardPlanDraft[] = [
  { card_type: "basic", count: 5, enabled: true },
  { card_type: "basic_reverse", count: 2, enabled: true },
  { card_type: "cloze", count: 2, enabled: true },
  { card_type: "true_false", count: 1, enabled: true }
]
```

Inside the component:

```ts
const [advancedMixEnabled, setAdvancedMixEnabled] = React.useState(false)
const [cardPlanDraft, setCardPlanDraft] = React.useState<FlashcardPlanDraft[]>(DEFAULT_CARD_PLAN)
const activeCardPlan = React.useMemo(
  () => cardPlanDraft
    .filter((row) => row.enabled && row.count > 0)
    .map(({ card_type, count }) => ({ card_type, count })),
  [cardPlanDraft]
)
const plannedCardTotal = React.useMemo(
  () => activeCardPlan.reduce((sum, row) => sum + row.count, 0),
  [activeCardPlan]
)
```

- [ ] **Step 4: Render toggle and fixed rows**

Add the toggle next to the simple count/type controls:

```tsx
<Switch
  checked={advancedMixEnabled}
  onChange={setAdvancedMixEnabled}
  data-testid="flashcards-generate-advanced-toggle"
/>
```

When enabled, render four fixed rows. Use existing `Input type="number"` to avoid a new numeric component. Test IDs:

- `flashcards-generate-plan-basic-count`
- `flashcards-generate-plan-basic-reverse-count`
- `flashcards-generate-plan-cloze-count`
- `flashcards-generate-plan-true-false-count`
- `flashcards-generate-plan-total`

Plan row counts should allow `0-100` per row so users can omit a type and so the aggregate can exceed `100`, which disables generation instead of silently clamping the requested mix.

Disable simple `numCards`/`cardType` controls while advanced mode is enabled or hide them if the layout is cleaner. Keep the source text, difficulty, deck, provider/model, and focus topics unchanged.

- [ ] **Step 5: Send plan from `handleGenerate`**

Build mutation params:

```ts
const generationParams = advancedMixEnabled
  ? {
      text: sourceText,
      numCards: plannedCardTotal,
      cardPlan: activeCardPlan,
      difficulty,
      focusTopics,
      provider: provider.trim() || undefined,
      model: model.trim() || undefined
    }
  : {
      text: sourceText,
      numCards,
      cardType,
      difficulty,
      focusTopics,
      provider: provider.trim() || undefined,
      model: model.trim() || undefined
    }
```

Disable Generate when advanced mode is enabled and `plannedCardTotal < 1 || plannedCardTotal > 100`.

- [ ] **Step 6: Show preview labels**

Add a tiny label in each preview card title or body:

```ts
const generationTypeLabel = {
  basic: "Basic",
  basic_reverse: "Reverse",
  cloze: "Cloze",
  true_false: "True/False"
}[card.generation_type || card.model_type]
```

Do not pass `generation_type` to `createMutation.mutateAsync`; the existing enumerated create payload already strips it. Keep the explicit save-stripping assertion.

- [ ] **Step 7: Run UI tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
```

Expected: import-results tests pass.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
git commit -m "feat: add flashcard advanced mix controls"
```

### Task 6: Verification, Security Scan, And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-12170 - Add-planned-flashcard-generation-types-and-counts.md`

- [ ] **Step 1: Run backend focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k "generate_flashcards" -v
```

Expected: pass.

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py -k "FlashcardGenerateAdapter" -v
```

Expected: pass.

- [ ] **Step 2: Run frontend focused tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/flashcards.test.ts ../packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
```

Expected: pass.

- [ ] **Step 3: Run typecheck if practical**

Run:

```bash
cd apps/tldw-frontend && bun run typecheck
```

Expected: pass. If the repo has unrelated baseline failures, document the exact failure and focused tests above.

- [ ] **Step 4: Run Bandit on touched backend Python**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/schemas/flashcards.py tldw_Server_API/app/api/v1/endpoints/flashcards.py tldw_Server_API/app/core/Workflows/adapters/content/_config.py tldw_Server_API/app/core/Workflows/adapters/content/generation.py -f json -o /tmp/bandit_task_12170_flashcard_plan.json
```

Expected: no new findings in touched code. If Bandit is unavailable in the venv, install nothing; document the missing tool.

- [ ] **Step 5: Run diff checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional files changed.

- [ ] **Step 6: Update Backlog task**

In `backlog/tasks/task-12170 - Add-planned-flashcard-generation-types-and-counts.md`, record:

- Implementation plan path.
- Commits made.
- Test and Bandit results.
- Final summary.
- Known skips: no visual flashcards, no stored generation metadata, no scheduler changes, no hidden caller default changes.

- [ ] **Step 7: Commit final bookkeeping**

```bash
git add "backlog/tasks/task-12170 - Add-planned-flashcard-generation-types-and-counts.md"
git commit -m "docs: close planned flashcard generation task"
```

- [ ] **Step 8: Request code review**

Use `superpowers:requesting-code-review` after implementation and verification complete.
