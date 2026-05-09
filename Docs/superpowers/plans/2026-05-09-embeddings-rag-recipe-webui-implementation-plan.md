# Embeddings RAG Recipe WebUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `embeddings_model_selection` recipe easy to run from `/evaluations?tab=recipes` for media-backed RAG embedding model selection, including light source labels, candidate readiness, recommendation-first results, and server-normalized apply preview.

**Architecture:** Keep the current recipe framework and embeddings A/B worker bridge as the execution path. Add additive backend contract metadata and recipe-specific helper endpoints, then replace the inline embeddings editor in `RecipesTab` with a focused shared UI component that serializes the same dataset/run config shape. Ship server-owned apply preview/copy guidance first; add live config mutation only as a final gated task after the preview contract is tested.

**Tech Stack:** FastAPI, Pydantic, existing Evaluations recipe services, existing embeddings provider config helpers, Next.js shared UI package, React, Ant Design, TanStack Query, Vitest, pytest, Bandit.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-08-embeddings-rag-recipe-webui-design.md`
- Tracking task: `TASK-145`
- Planning task: `TASK-145.1`

## Execution Status

- Tasks 1-6 completed for the preview/copy-config V1.
- Task 7 is deferred behind explicit live config mutation approval and tracked as `TASK-145.8`.
- Shipped behavior: guided embeddings recipe setup, server-owned candidate readiness, recommendation-first results, and server-normalized apply preview with copy-config fallback. Live config mutation is not included.

## Pre-Implementation Rules

- Do all code work in a clean feature worktree, not the dirty main checkout.
- Before editing implementation files, create or update a Backlog.md task for the implementation stage being executed.
- Keep V1 media-scoped. Guided expected sources must serialize only integer media IDs into `expected_ids`.
- Do not mutate live RAG or embeddings config during recipe execution.
- Server owns candidate readiness and apply eligibility. The frontend may render warnings, but must not invent the core allow/block decision.
- If live config mutation is not fully safe, stop at preview/copy-config and create a follow-up task instead of adding a broad config write.
- Do not call FastAPI endpoint functions from helper code. Backend helpers should use core config/policy functions so tests can exercise them without request objects.
- Any candidate/apply payload returned to the UI must be secret-free. Never include API keys, auth headers, full environment values, or unredacted config snapshots.

## File Map

### Backend

- Modify: `tldw_Server_API/app/core/Evaluations/recipes/embeddings_retrieval.py`
  - Add UI-friendly manifest capabilities/defaults.
  - Add validation warnings for unlabeled/light-label cases.
  - Add recommendation slot metadata needed by apply preview.
- Create: `tldw_Server_API/app/core/Evaluations/recipes/embeddings_recipe_hints.py`
  - Normalize current embedding config and candidate hints.
  - Classify candidate runnable status with the same policy inputs used by embeddings A/B execution.
  - Resolve a recipe run recommendation slot into a provider/model apply preview.
- Modify: `tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py`
  - Add candidate hint and apply preview Pydantic schemas.
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py`
  - Add recipe-specific candidate discovery endpoint.
  - Add recommendation apply-preview endpoint.
  - Add live apply endpoint only in the final gated task.
- Optional final task only: `tldw_Server_API/app/core/Setup/setup_manager.py`
  - Do not edit unless live apply needs a small, focused helper around existing `update_config`.

### Backend Tests

- Modify: `tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py`
- Create: `tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py`
- Modify: `tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py`

### Frontend

- Modify: `apps/packages/ui/src/services/evaluations.ts`
  - Add typed service functions for candidate hints and apply preview.
- Modify: `apps/packages/ui/src/components/Option/Evaluations/hooks/useRecipes.ts`
  - Add query/mutation hooks for the new recipe helper APIs.
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
  - Add any new API paths called by shared UI.
- Create: `apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/EmbeddingsModelSelectionConfig.tsx`
  - Guided corpus, query, expected media source, and candidate model controls.
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx`
  - Delegate embeddings recipe editing to the new component.
  - Render recommendation-first cards for embeddings reports.

### Frontend Tests

- Create: `apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/__tests__/EmbeddingsModelSelectionConfig.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx`

---

## Task 1: Backend Manifest, Validation, and Report Metadata

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/recipes/embeddings_retrieval.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py`

- [ ] **Step 1: Write failing manifest capability tests**

Add tests:

```python
def test_manifest_advertises_guided_media_scoped_rag_flow() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    manifest = recipe.manifest

    assert manifest.capabilities["guided_ui"] is True
    assert manifest.capabilities["source_labeling"]["source_id_contract"] == {
        "kind": "media_id",
        "type": "integer",
    }
    assert manifest.capabilities["candidate_discovery"]["endpoint"].endswith(
        "/recipes/embeddings_model_selection/candidates"
    )
    assert manifest.capabilities["apply_target"]["preview_supported"] is True
    assert manifest.capabilities["apply_target"]["live_apply_supported"] is False
    assert manifest.default_run_config["comparison_mode"] == "embedding_only"
    assert manifest.default_run_config["top_k"] == 10
```

- [ ] **Step 2: Write failing validation warning tests**

Add tests:

```python
def test_validation_warns_when_guided_queries_have_no_expected_sources() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    result = recipe.validate_dataset(
        [{"query_id": "q-1", "input": "Where are chunking defaults documented?"}],
        run_config={"guided_source_labeling": True},
    )

    assert result["valid"] is True
    assert result["dataset_mode"] == "unlabeled"
    assert "warnings" in result
    assert any("expected source" in warning.lower() for warning in result["warnings"])


def test_validation_rejects_non_integer_expected_ids_for_media_scoped_guided_flow() -> None:
    recipe = EmbeddingsRetrievalRecipe()

    result = recipe.validate_dataset(
        [
            {
                "query_id": "q-1",
                "input": "Find alpha",
                "expected_ids": ["chunk-123"],
            }
        ],
        run_config={"source_id_contract": "media_id"},
    )

    assert result["valid"] is False
    assert any("media id" in error.lower() for error in result["errors"])
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py -q
```

Expected: the new tests fail because the manifest metadata and warning fields do not exist yet.

- [ ] **Step 4: Implement manifest metadata and validation warnings**

In `EmbeddingsRetrievalRecipe.manifest`, add only additive fields:

```python
capabilities={
    "guided_ui": True,
    "rag_embedding_selection": True,
    "media_scoped_execution": True,
    "source_labeling": {
        "supported": True,
        "source_id_contract": {"kind": "media_id", "type": "integer"},
        "advanced_manual_ids": True,
    },
    "candidate_discovery": {
        "supported": True,
        "endpoint": "/api/v1/evaluations/recipes/embeddings_model_selection/candidates",
        "runnable_statuses": [
            "ready",
            "missing_key",
            "disallowed_provider",
            "disallowed_model",
            "quota_risk",
            "unknown",
        ],
    },
    "apply_target": {
        "preview_supported": True,
        "live_apply_supported": False,
        "config_section": "Embeddings",
        "config_keys": ["embedding_provider", "embedding_model"],
    },
},
default_run_config={
    "comparison_mode": "embedding_only",
    "top_k": 10,
    "hybrid_alpha": 0.7,
    "guided_source_labeling": True,
    "source_id_contract": "media_id",
    "candidates": [],
}
```

In `validate_dataset`, keep unlabeled datasets valid but add `warnings: list[str]` when guided labeling is enabled and no sample has `expected_ids`.

- [ ] **Step 5: Add recommendation metadata test**

Extend `test_build_report_emits_recommendation_slots_and_confidence_inputs`:

```python
slot = report["recommendation_slots"]["best_overall"]
assert slot["metadata"]["provider"] == "openai"
assert slot["metadata"]["model"] == "text-embedding-3-small"
assert slot["metadata"]["apply_eligible"] is True
assert "apply_warnings" in slot["metadata"]
```

- [ ] **Step 6: Implement report metadata**

Update `_build_slot` so concrete slots include:

```python
metadata={
    "candidate_id": candidate_id,
    "provider": provider,
    "model": model,
    "is_local": is_local,
    "apply_eligible": bool(provider and model and candidate_run_id),
    "apply_warnings": apply_warnings,
    "confidence": confidence,
}
```

Use warnings for low sample count or close margin only from already computed report inputs.

- [ ] **Step 7: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/recipes/embeddings_retrieval.py tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py
git commit -m "feat: expose embeddings recipe guided contract"
```

---

## Task 2: Backend Candidate Hints and Apply Preview

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/recipes/embeddings_recipe_hints.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py`
- Create: `tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py`
- Modify: `tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py`

- [ ] **Step 1: Write failing candidate hint unit tests**

Add `tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py`:

```python
from __future__ import annotations

from tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints import (
    build_embedding_recipe_candidate_hints,
)


def test_candidate_hints_include_current_model_and_policy_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "text-embedding-3-small",
            "providers": [{"name": "openai", "models": ["text-embedding-3-small"]}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: ["openai"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: ["text-embedding-3-*"],
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["current"]["provider"] == "openai"
    assert result["current"]["model"] == "text-embedding-3-small"
    assert result["candidates"][0]["status"] == "ready"
    assert result["candidates"][0]["default"] is True
```

- [ ] **Step 2: Write failing disallowed candidate test**

```python
def test_candidate_hints_mark_disallowed_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "text-embedding-3-small",
            "providers": [{"name": "huggingface", "models": ["BAAI/bge-small-en-v1.5"]}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: ["openai"],
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: None,
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["candidates"][0]["status"] == "disallowed_provider"
    assert "not allowed" in result["candidates"][0]["status_reason"].lower()
```

- [ ] **Step 3: Write failing missing-key and apply-preview tests**

Add a missing-key readiness test for remote configured providers:

```python
def test_candidate_hints_mark_missing_key_for_remote_provider(monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_simplified_embeddings_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "text-embedding-3-small",
            "providers": [{"name": "openai", "models": ["text-embedding-3-small"], "api_key": None}],
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_providers",
        lambda: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_allowed_embedding_models",
        lambda: None,
    )

    result = build_embedding_recipe_candidate_hints(user=None)

    assert result["candidates"][0]["status"] == "missing_key"
```

```python
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints import (
    build_embedding_recipe_apply_preview,
)


def test_apply_preview_resolves_slot_to_copy_config(monkeypatch) -> None:
    class FakeService:
        def get_report(self, _run_id):
            return {
                "run": {
                    "run_id": "recipe-run-1",
                    "recipe_id": "embeddings_model_selection",
                    "status": RunStatus.COMPLETED,
                    "metadata": {},
                },
                "recommendation_slots": {
                    "best_overall": {
                        "candidate_run_id": "arm-1",
                        "metadata": {
                            "provider": "openai",
                            "model": "text-embedding-3-small",
                            "apply_eligible": True,
                            "apply_warnings": ["Existing indexes may need rebuild."],
                        },
                    }
                },
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Evaluations.recipes.embeddings_recipe_hints.get_current_embedding_config",
        lambda: {"provider": "huggingface", "model": "Qwen/Qwen3-Embedding-0.6B"},
    )

    preview = build_embedding_recipe_apply_preview(
        FakeService(),
        run_id="recipe-run-1",
        slot_name="best_overall",
        live_apply_supported=False,
    )

    assert preview["apply_eligible"] is True
    assert preview["apply_available"] is False
    assert preview["proposed"]["provider"] == "openai"
    assert preview["copy_config"]["Embeddings"]["embedding_model"] == "text-embedding-3-small"
```

- [ ] **Step 4: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py -q
```

Expected: import failure because the helper module does not exist yet.

- [ ] **Step 5: Add Pydantic schemas**

In `evaluation_recipe_schemas.py`, add:

```python
EmbeddingCandidateStatus = Literal[
    "ready",
    "missing_key",
    "disallowed_provider",
    "disallowed_model",
    "quota_risk",
    "unknown",
]


class EmbeddingRecipeCandidateHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    is_local: bool = False
    default: bool = False
    status: EmbeddingCandidateStatus = "unknown"
    status_reason: str | None = None
    dimensions: int | None = Field(default=None, ge=1)
    revision: str | None = None
    cost_hint: str | None = None


class EmbeddingRecipeCandidatesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipe_id: Literal["embeddings_model_selection"] = "embeddings_model_selection"
    current: EmbeddingRecipeCandidateHint | None = None
    candidates: list[EmbeddingRecipeCandidateHint] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RecipeRecommendationApplyPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_name: str = "best_overall"
    candidate_run_id: str | None = None


class RecipeRecommendationApplyPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    recipe_id: str
    slot_name: str
    candidate_run_id: str | None = None
    apply_eligible: bool
    apply_available: bool = False
    blocked_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    current: dict[str, str | None] = Field(default_factory=dict)
    proposed: dict[str, str | None] = Field(default_factory=dict)
    affected_config: dict[str, str] = Field(default_factory=dict)
    copy_config: dict[str, dict[str, str]] = Field(default_factory=dict)
    reindex_required: bool = True
```

- [ ] **Step 6: Implement helper module**

Create `embeddings_recipe_hints.py`.

Implementation notes:

- Use `tldw_Server_API.app.core.Embeddings.simplified_config.get_config()` through a tiny wrapper named `get_simplified_embeddings_config`.
- Use the same policy inputs as `validate_abtest_policy` by wrapping `_get_allowed_providers`, `_get_allowed_models`, and `_should_enforce_policy` from `embeddings_v5_production_enhanced`. Do not duplicate policy semantics.
- Normalize both dict-like report payloads and the existing `RecipeRunReport` Pydantic model. Unit tests may use dictionaries, but production receives `RecipeRunReport`.
- Treat providers named `local`, `huggingface`, `onnx`, `llamacpp`, and `sentence-transformers` as local-ish for `is_local`.
- Return `apply_available=False` until Task 7 deliberately enables live mutation.
- `missing_key` should only apply to remote providers that require a key and have no configured key in the simplified provider config or provider-specific environment.

Core shape:

```python
def build_embedding_recipe_candidate_hints(*, user: object | None) -> dict[str, object]:
    config = get_simplified_embeddings_config()
    current = _candidate_from_provider_model(
        str(config.get("default_provider") or ""),
        str(config.get("default_model") or ""),
        default=True,
        user=user,
    )
    candidates = _collect_configured_candidates(config, user=user)
    return {
        "recipe_id": "embeddings_model_selection",
        "current": current,
        "candidates": _dedupe_with_current_first(current, candidates),
        "warnings": [],
    }
```

- [ ] **Step 7: Add API endpoint tests**

In `integration/test_recipe_runs_api.py`, add tests for:

```python
def test_embeddings_recipe_candidates_endpoint_returns_current_model(...)
def test_embeddings_recipe_apply_preview_returns_copy_config_for_completed_run(...)
def test_embeddings_recipe_apply_preview_rejects_non_embeddings_recipe_run(...)
```

Use existing TestClient/auth fixtures in the file. If the file uses a different fixture pattern, follow that local pattern rather than introducing new app setup.

- [ ] **Step 8: Implement endpoints**

In `evaluations_recipes.py`, add:

```python
@recipes_router.get(
    "/recipes/embeddings_model_selection/candidates",
    response_model=EmbeddingRecipeCandidatesResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def get_embeddings_recipe_candidates(...):
    ...


@recipes_router.post(
    "/recipe-runs/{run_id}/apply-preview",
    response_model=RecipeRecommendationApplyPreviewResponse,
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def preview_recipe_recommendation_apply(...):
    ...
```

Guard the preview endpoint:

- run must exist for the current user
- run must be `embeddings_model_selection`
- run must be completed
- slot must exist
- slot metadata must include provider/model

- [ ] **Step 9: Run focused backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py -q
```

Expected: all touched backend tests pass.

- [ ] **Step 10: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/recipes/embeddings_recipe_hints.py tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py
git commit -m "feat: add embeddings recipe candidate and apply preview APIs"
```

---

## Task 3: Frontend Services and Recipe Hooks

**Files:**
- Modify: `apps/packages/ui/src/services/evaluations.ts`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/hooks/useRecipes.ts`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`

- [ ] **Step 1: Write failing service/hook tests**

In `useRecipes.test.tsx`, extend the evaluations service mock to include:

```ts
getEmbeddingRecipeCandidates: vi.fn(),
previewRecipeRecommendationApply: vi.fn()
```

Add tests:

```ts
it("loads embeddings recipe candidates with a stable query key", async () => {
  vi.mocked(getEmbeddingRecipeCandidates).mockResolvedValue({
    ok: true,
    data: {
      recipe_id: "embeddings_model_selection",
      current: {
        provider: "openai",
        model: "text-embedding-3-small",
        status: "ready",
        default: true,
        is_local: false
      },
      candidates: [],
      warnings: []
    }
  } as any)

  const { result } = renderHook(() => useEmbeddingRecipeCandidates(true), {
    wrapper: buildWrapper(queryClient)
  })

  await waitFor(() => expect(result.current.data?.data?.current?.provider).toBe("openai"))
})

it("posts apply preview requests through the hook", async () => {
  vi.mocked(previewRecipeRecommendationApply).mockResolvedValue({
    ok: true,
    data: {
      run_id: "recipe-run-1",
      recipe_id: "embeddings_model_selection",
      slot_name: "best_overall",
      apply_eligible: true,
      apply_available: false,
      current: {},
      proposed: {},
      affected_config: {},
      copy_config: {},
      warnings: []
    }
  } as any)

  const { result } = renderHook(() => usePreviewRecipeRecommendationApply(), {
    wrapper: buildWrapper(queryClient)
  })

  await act(async () => {
    await result.current.mutateAsync({ runId: "recipe-run-1", slotName: "best_overall" })
  })

  expect(previewRecipeRecommendationApply).toHaveBeenCalledWith("recipe-run-1", {
    slot_name: "best_overall"
  })
})
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx
```

Expected: imports fail because hooks/service functions do not exist.

- [ ] **Step 3: Add service types and functions**

In `evaluations.ts`, add TypeScript types mirroring the Pydantic schemas:

```ts
export type EmbeddingRecipeCandidateStatus =
  | "ready"
  | "missing_key"
  | "disallowed_provider"
  | "disallowed_model"
  | "quota_risk"
  | "unknown"

export type EmbeddingRecipeCandidateHint = {
  provider: string
  model: string
  is_local: boolean
  default: boolean
  status: EmbeddingRecipeCandidateStatus
  status_reason?: string | null
  dimensions?: number | null
  revision?: string | null
  cost_hint?: string | null
}

export type EmbeddingRecipeCandidatesResponse = {
  recipe_id: "embeddings_model_selection"
  current?: EmbeddingRecipeCandidateHint | null
  candidates: EmbeddingRecipeCandidateHint[]
  warnings: string[]
}

export async function getEmbeddingRecipeCandidates() {
  return await apiSend<EmbeddingRecipeCandidatesResponse>({
    path: "/api/v1/evaluations/recipes/embeddings_model_selection/candidates" as any,
    method: "GET"
  })
}

export async function previewRecipeRecommendationApply(
  runId: string,
  payload: { slot_name: string; candidate_run_id?: string | null }
) {
  return await apiSend<RecipeRecommendationApplyPreviewResponse>({
    path: `/api/v1/evaluations/recipe-runs/${encodeURIComponent(runId)}/apply-preview` as any,
    method: "POST",
    body: payload
  })
}
```

- [ ] **Step 4: Add hooks**

In `useRecipes.ts`, add:

```ts
export function useEmbeddingRecipeCandidates(enabled: boolean) {
  return useQuery({
    queryKey: ["evaluations", "recipes", "embeddings_model_selection", "candidates"],
    queryFn: getEmbeddingRecipeCandidates,
    enabled,
    staleTime: 60 * 1000
  })
}

export function usePreviewRecipeRecommendationApply() {
  return useMutation({
    mutationFn: ({ runId, slotName, candidateRunId }: {
      runId: string
      slotName: string
      candidateRunId?: string | null
    }) =>
      previewRecipeRecommendationApply(runId, {
        slot_name: slotName,
        candidate_run_id: candidateRunId ?? null
      })
  })
}
```

- [ ] **Step 5: Update OpenAPI guard path union**

Add:

```ts
| "/api/v1/evaluations/recipes/embeddings_model_selection/candidates"
| "/api/v1/evaluations/recipe-runs/{run_id}/apply-preview"
```

- [ ] **Step 6: Run frontend hook tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/services/evaluations.ts apps/packages/ui/src/components/Option/Evaluations/hooks/useRecipes.ts apps/packages/ui/src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx apps/packages/ui/src/services/tldw/openapi-guard.ts
git commit -m "feat: add embeddings recipe frontend API hooks"
```

---

## Task 4: Dedicated Guided Embeddings Recipe Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/EmbeddingsModelSelectionConfig.tsx`
- Create: `apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/__tests__/EmbeddingsModelSelectionConfig.test.tsx`

- [ ] **Step 1: Write failing component tests for guided serialization**

Create tests that render the component with controlled `dataset` and `runConfig`.

Core test:

```tsx
it("serializes query rows and selected media ids into recipe payload shape", async () => {
  const manifest = {
    recipe_id: "embeddings_model_selection",
    recipe_version: "1",
    name: "Embeddings Model Selection",
    description: "Pick an embedding model for RAG",
    launchable: true,
    supported_modes: ["labeled", "unlabeled"],
    tags: ["rag", "embeddings"],
    capabilities: {
      source_labeling: {
        source_id_contract: { kind: "media_id", type: "integer" }
      }
    },
    default_run_config: { comparison_mode: "embedding_only", top_k: 10 }
  } as RecipeManifest
  const onDatasetChange = vi.fn()
  const onRunConfigChange = vi.fn()

  render(
    <EmbeddingsModelSelectionConfig
      datasetSource="inline"
      dataset={[{ query_id: "q-1", input: "", expected_ids: [] }]}
      runConfig={{ comparison_mode: "embedding_only", candidates: [] }}
      manifest={manifest}
      onDatasetChange={onDatasetChange}
      onRunConfigChange={onRunConfigChange}
    />
  )

  fireEvent.change(screen.getByLabelText("Query text 1"), {
    target: { value: "find the beta launch notes" }
  })
  fireEvent.change(screen.getByLabelText("Expected media IDs 1"), {
    target: { value: "7, 9" }
  })

  expect(onDatasetChange).toHaveBeenLastCalledWith([
    {
      query_id: "q-1",
      input: "find the beta launch notes",
      expected_ids: ["7", "9"]
    }
  ])
})
```

- [ ] **Step 2: Write failing candidate readiness tests**

Mock `useEmbeddingRecipeCandidates` to return:

```ts
{
  data: {
    ok: true,
    data: {
      candidates: [
        { provider: "openai", model: "text-embedding-3-small", status: "ready" },
        { provider: "anthropic", model: "not-embedding", status: "disallowed_provider" }
      ]
    }
  }
}
```

Assert that ready candidates can be selected and disallowed candidates render a status reason and are not auto-added to run config.

- [ ] **Step 3: Write failing media search test**

Mock `tldwClient.searchMedia` and assert that search-and-select stores only integer media IDs:

```tsx
vi.mocked(tldwClient.searchMedia).mockResolvedValue({
  media: [{ id: 42, title: "Launch notes", url: "file.md" }]
})

fireEvent.change(screen.getByLabelText("Find expected sources for query 1"), {
  target: { value: "launch" }
})
fireEvent.click(await screen.findByRole("checkbox", { name: /Launch notes/i }))

expect(onDatasetChange).toHaveBeenLastCalledWith([
  expect.objectContaining({ expected_ids: ["42"] })
])
```

- [ ] **Step 4: Run component tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/tabs/recipe-configs/__tests__/EmbeddingsModelSelectionConfig.test.tsx
```

Expected: import failure because the component does not exist.

- [ ] **Step 5: Implement component**

Component requirements:

- Props match the existing recipe config pattern:

```ts
type Props = {
  datasetSource: "inline" | "saved"
  dataset: DatasetSample[]
  runConfig: Record<string, any>
  manifest?: RecipeManifest | null
  onDatasetChange: (next: DatasetSample[]) => void
  onRunConfigChange: (next: Record<string, any>) => void
}
```

- Use compact stacked sections: Corpus, Queries, Expected sources, Models, Run review.
- Use media IDs in guided expected sources. Do not accept chunk IDs or note IDs in guided mode.
- Keep manual IDs available through an "Advanced media IDs" input.
- Use `tldwClient.searchMedia({ query }, { page: 1, results_per_page: 8 })` for source search.
- Normalize media search payloads defensively because existing callers receive `media`, `results`, or `items` depending on endpoint path.
- Candidate rows come from `useEmbeddingRecipeCandidates`; only `ready` candidates are auto-prefilled.
- Preserve user-edited candidates and do not overwrite them on every candidate-hint refetch.

- [ ] **Step 6: Run component tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/tabs/recipe-configs/__tests__/EmbeddingsModelSelectionConfig.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/EmbeddingsModelSelectionConfig.tsx apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/__tests__/EmbeddingsModelSelectionConfig.test.tsx
git commit -m "feat: add guided embeddings recipe config"
```

---

## Task 5: Integrate Guided Config and Recommendation-First Results

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx`

- [ ] **Step 1: Update existing launch test expectations**

In `RecipesTab.launch.test.tsx`, replace the current inline embeddings field labels with the new component labels while keeping the payload assertion:

```tsx
fireEvent.click(screen.getByRole("button", { name: "Use Embeddings Model Selection" }))
fireEvent.change(screen.getByLabelText("Query text 1"), {
  target: { value: "find the beta launch notes" }
})
fireEvent.change(screen.getByLabelText("Expected media IDs 1"), {
  target: { value: "7, 9" }
})
fireEvent.change(screen.getByLabelText("Media IDs"), {
  target: { value: "7, 9, 12" }
})
```

Keep the expected create payload:

```ts
expect(createSpy).toHaveBeenCalledWith(
  expect.objectContaining({
    recipeId: "embeddings_model_selection",
    dataset: [
      {
        query_id: "q-1",
        input: "find the beta launch notes",
        expected_ids: ["7", "9"]
      }
    ],
    runConfig: expect.objectContaining({
      media_ids: [7, 9, 12],
      candidates: expect.arrayContaining([
        expect.objectContaining({ provider: "local", model: "bge-large" })
      ])
    })
  })
)
```

- [ ] **Step 2: Add recommendation result tests**

Add a report fixture where `recipe_report` includes `best_overall`, candidate metrics, and slot metadata:

```ts
expect(screen.getByText("Best overall")).toBeInTheDocument()
expect(screen.getByText("text-embedding-3-small")).toBeInTheDocument()
expect(screen.getByText(/Recall/i)).toBeInTheDocument()
expect(screen.getByRole("button", { name: /Preview RAG config change/i })).toBeInTheDocument()
```

Also add a negative test where `apply_eligible=false` and assert the preview/apply button is hidden or disabled with the server blocked reason.

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx
```

Expected: tests fail because `RecipesTab` still uses inline embeddings editors and generic report rendering.

- [ ] **Step 4: Delegate embeddings config**

In `RecipesTab.tsx`:

- import `EmbeddingsModelSelectionConfig`
- replace `renderEmbeddingsDatasetEditor` and `renderEmbeddingsRunConfigEditor` usage with the new component for `selectedRecipeId === "embeddings_model_selection"`
- leave raw JSON advanced editor behavior unchanged
- do not remove generic recipe support for other recipes

- [ ] **Step 5: Render embeddings recommendation cards**

Add a small rendering branch for `embeddings_model_selection` reports:

- best overall
- best local
- best cheap
- confidence/low sample warnings from server metadata
- metrics from `run.metadata.recipe_report.candidates`
- preview action only when slot metadata says `apply_eligible`

Keep raw report JSON/details available after the cards.

- [ ] **Step 6: Run focused RecipesTab tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx
git commit -m "feat: integrate guided embeddings recipe flow"
```

---

## Task 6: Apply Preview UI and Copy-Config Fallback

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx`

- [ ] **Step 1: Write failing preview modal tests**

Add tests:

```tsx
it("renders server-normalized apply preview and copy fallback", async () => {
  previewApplySpy.mockResolvedValue({
    ok: true,
    data: {
      run_id: "recipe-run-1",
      recipe_id: "embeddings_model_selection",
      slot_name: "best_overall",
      candidate_run_id: "arm-1",
      apply_eligible: true,
      apply_available: false,
      current: { provider: "huggingface", model: "Qwen/Qwen3-Embedding-0.6B" },
      proposed: { provider: "openai", model: "text-embedding-3-small" },
      affected_config: {
        section: "Embeddings",
        provider_key: "embedding_provider",
        model_key: "embedding_model"
      },
      copy_config: {
        Embeddings: {
          embedding_provider: "openai",
          embedding_model: "text-embedding-3-small"
        }
      },
      reindex_required: true,
      warnings: ["Existing indexes may need rebuild."]
    }
  })

  fireEvent.click(screen.getByRole("button", { name: /Preview RAG config change/i }))

  expect(await screen.findByText("Current embedding model")).toBeInTheDocument()
  expect(screen.getByText("Qwen/Qwen3-Embedding-0.6B")).toBeInTheDocument()
  expect(screen.getByText("text-embedding-3-small")).toBeInTheDocument()
  expect(screen.getByRole("button", { name: /Copy config change/i })).toBeInTheDocument()
})
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx
```

Expected: preview modal/action does not exist.

- [ ] **Step 3: Implement preview modal**

Use `usePreviewRecipeRecommendationApply`.

Behavior:

- Button label: `Preview RAG config change`.
- On click, call the backend preview endpoint with `runId`, `slotName`, and optional `candidateRunId`.
- Modal shows current provider/model, proposed provider/model, affected config keys, run ID, warnings, and reindex required.
- If `apply_available=false`, show copy-config fallback and no live apply button.
- If `apply_available=true` later, the modal can show a disabled placeholder until Task 7 adds mutation.

- [ ] **Step 4: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx
git commit -m "feat: add embeddings recipe apply preview UI"
```

---

## Task 7: Gated Live Apply Endpoint

Only execute this task if Task 2 preview is stable, the mutation boundary is confirmed narrow enough, and the user explicitly approves live config mutation work. If not, create a follow-up Backlog task and stop with preview/copy-config as the shipped behavior.

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/recipes/embeddings_recipe_hints.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py`
- Modify: `tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py`
- Modify: `apps/packages/ui/src/services/evaluations.ts`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/hooks/useRecipes.ts`
- Modify: `apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx`
- Modify: relevant frontend tests from Tasks 3 and 6.

- [ ] **Step 1: Write failing backend apply tests**

Backend requirements:

- POST `/api/v1/evaluations/recipe-runs/{run_id}/apply`
- requires `EVALS_MANAGE`
- only `embeddings_model_selection`
- completed run only
- slot candidate must match preview
- writes only `[Embeddings] embedding_provider` and `[Embeddings] embedding_model`
- records audit metadata on the recipe run
- returns previous and new values plus backup path if available
- refuses to apply when environment variables override `[Embeddings]` provider/model, because editing `config.txt` would not change the effective runtime value

Test should monkeypatch `setup_manager.update_config` and assert exact payload:

```python
assert update_config_spy.call_args.args[0] == {
    "Embeddings": {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
    }
}
```

- [ ] **Step 2: Implement apply schema and endpoint**

Use one request schema:

```python
class RecipeRecommendationApplyRequest(RecipeRecommendationApplyPreviewRequest):
    confirmed_provider: str
    confirmed_model: str
```

Use one response schema extending preview with:

```python
applied: bool
backup_path: str | None = None
audit_ref: str | None = None
```

- [ ] **Step 3: Implement mutation using setup manager**

In helper:

- call preview first
- compare confirmed provider/model against preview proposed values
- call `setup_manager.update_config({"Embeddings": {...}}, create_backup=True)`
- update recipe run metadata with `embedding_recipe_apply_audit`
- log with Loguru, never log secrets

- [ ] **Step 4: Add frontend service/hook and modal apply button**

Add:

```ts
applyRecipeRecommendation(runId, payload)
useApplyRecipeRecommendation()
```

In the modal, only show `Apply to RAG config` when preview returns `apply_available=true`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py -q
cd apps/packages/ui
bunx vitest run src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/recipes/embeddings_recipe_hints.py tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py apps/packages/ui/src/services/evaluations.ts apps/packages/ui/src/components/Option/Evaluations/hooks/useRecipes.ts apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx
git commit -m "feat: apply embeddings recipe recommendation to rag config"
```

---

## Final Verification

- [ ] **Backend focused tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py \
  tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py \
  tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py \
  -q
```

- [ ] **Frontend focused tests**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx \
  src/components/Option/Evaluations/tabs/recipe-configs/__tests__/EmbeddingsModelSelectionConfig.test.tsx \
  src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx \
  src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx
```

- [ ] **OpenAPI guard if new frontend-called paths are added**

```bash
cd apps/packages/ui
bun run verify:openapi
```

- [ ] **Bandit on touched backend source**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Evaluations/recipes/embeddings_retrieval.py \
  tldw_Server_API/app/core/Evaluations/recipes/embeddings_recipe_hints.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py \
  tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py \
  -f json -o /tmp/bandit_embeddings_rag_recipe.json
```

- [ ] **Diff hygiene**

```bash
git diff --check
git status --short
```

## Completion Notes

- If Task 7 is not executed, completion summary must explicitly say the shipped behavior is server preview/copy-config fallback, not live apply.
- If Task 7 is executed, completion summary must include the exact config keys changed by live apply and the audit location.
- Update Backlog task status, notes, verification evidence, and final summary before the final implementation commit or PR.
