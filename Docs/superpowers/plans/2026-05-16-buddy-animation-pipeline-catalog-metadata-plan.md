# Buddy Animation Pipeline Catalog Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the bundled Buddy starter catalog metadata enforce the neutral-anchor-first animation pipeline contract without claiming final default animation art exists.

**Architecture:** Keep the current Persona Visual pack/runtime contract intact. The first implementation slice only tightens starter catalog recipe metadata, schema validation, documentation, and tests so static talking/reaction sheets remain separate from timed animation outputs. It does not add final art assets, new renderers, or activation behavior.

**Tech Stack:** Python dataclasses and Pydantic schemas in `tldw_Server_API`, pytest unit tests, existing Persona Visual starter catalog service, existing Markdown docs.

---

## Scope Check

The design spec covers several independent subsystems: catalog metadata, production packet fixtures, editor UX, runtime/MCP trigger hardening, and final asset production. This plan intentionally covers only the first concrete slice:

- catalog production metadata alignment,
- static-sheet versus animation-output separation,
- recipe/schema tests,
- documentation update,
- Backlog/task metadata update.

Do not generate final Buddy images in this slice. Do not change runtime rendering behavior. Do not change activation behavior.

## Files And Responsibilities

- Create: `tldw_Server_API/app/core/Persona/visual_starter_recipe_taxonomy.py`
  - Owns shared taxonomy constants for starter expected asset groups, static
    source groups, and timed animation output IDs.
  - Keeps fixture validation and API schema validation from drifting.

- Modify: `tldw_Server_API/app/core/Persona/visual_starter_fixtures.py`
  - Owns immutable bundled starter catalog fixture definitions and production recipes.
  - Remove `static_talking_reaction_sheet` from `animation_outputs`.
  - Keep `static_talking_reaction_sheet` in `expected_asset_groups` where appropriate.

- Modify: `tldw_Server_API/app/core/Persona/visual_starter_catalog.py`
  - Owns service-level starter fixture validation.
  - Reject recipe `animation_outputs` that are static/source asset groups before
    invalid fixture metadata reaches API response construction.
  - Reject recipe `animation_outputs` that are not declared as expected asset
    groups for the same starter, so a fixture cannot advertise a production
    output that the catalog does not claim to collect.

- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
  - Owns API response validation for starter production recipes.
  - Reuse the shared taxonomy to validate that recipe `animation_outputs` use
    supported timed-output IDs and do not include static asset groups.
  - Reuse the shared taxonomy to validate starter `expected_asset_groups` at the
    API response boundary as a backstop against catalog drift.

- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py`
  - Owns focused starter catalog fixture/service validation.
  - Add tests for the pipeline taxonomy and static/animation separation.
  - Update existing expectations that currently treat `static_talking_reaction_sheet` as an animation output.

- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`
  - Owns API-level starter catalog and generation enqueue coverage.
  - Add or adjust assertions so the API exposes static sheet guidance separately from `animation_outputs`.

- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_jobs.py`
  - Owns generated-candidate job and provenance coverage.
  - Update any invalid recipe-output fixtures that use static sheet outputs.

- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_candidate_provenance.py`
  - Owns generated-candidate provenance sanitization coverage.
  - Keep `static_sheet` as guidance metadata, but do not use static sheet labels
    as recipe output IDs.

- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py`
  - Owns DB-level candidate provenance persistence coverage.
  - Keep historical/static-sheet text sanitization coverage while using a valid
    timed recipe output ID.

- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
  - Owns durable technical documentation for Persona/Buddy visual packs.
  - Clarify that static talking/reaction sheets are expected asset groups and source material, not timed animation outputs.

- Modify: implementation Backlog task for this slice.
  - Use `TASK-411` if present: `backlog/tasks/task-411 - Separate-Buddy-static-source-sheets-from-animation-outputs.md`.
  - Fall back to `TASK-410` only if `TASK-411` does not exist in the worktree.
  - Record the plan path, touched files, verification, and known Bandit/doc-only details.

## Task 1: Add Pipeline Taxonomy Tests

**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_starter_recipe_taxonomy.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py`

- [ ] **Step 1: Write failing taxonomy tests**

Add imports from the new taxonomy module for the constants that will be implemented in Step 3:

```python
from tldw_Server_API.app.core.Persona.visual_starter_recipe_taxonomy import (
    BUDDY_VISUAL_ANIMATION_OUTPUT_IDS,
    BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS,
    BUDDY_VISUAL_STATIC_SOURCE_ASSET_GROUP_IDS,
)
```

Add this test near the production-readiness tests:

```python
def test_default_starter_production_recipes_use_pipeline_taxonomy(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    for detail in (
        service.get_starter_pack(starter_id)
        for starter_id in DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS
    ):
        expected_groups = set(detail["expected_asset_groups"])
        animation_outputs = set(detail["production_recipe"]["animation_outputs"])

        assert expected_groups <= BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS
        assert animation_outputs <= BUDDY_VISUAL_ANIMATION_OUTPUT_IDS
        assert not (animation_outputs & BUDDY_VISUAL_STATIC_SOURCE_ASSET_GROUP_IDS)
        assert animation_outputs <= expected_groups
```

Add this test to prove intermediate/intricate starters still ask for static source material, but not as animation outputs:

```python
@pytest.mark.parametrize(
    "starter_pack_id",
    (
        "study-desk-intermediate",
        "tool-helper-intermediate",
        "object-creature-intermediate",
        "lofi-study-intricate",
        "action-guide-intricate",
        "elaborate-persona-intricate",
    ),
)
def test_static_talking_sheet_is_source_material_not_animation_output(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    detail = service.get_starter_pack(starter_pack_id)

    assert "static_talking_reaction_sheet" in detail["expected_asset_groups"]
    assert "static" in detail["production_recipe"]["static_sheet"].lower()
    assert (
        "static_talking_reaction_sheet"
        not in detail["production_recipe"]["animation_outputs"]
    )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_default_starter_production_recipes_use_pipeline_taxonomy \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_static_talking_sheet_is_source_material_not_animation_output \
  -v
```

Expected: fail because the taxonomy module/constants do not exist and the current recipes include `static_talking_reaction_sheet` in some `animation_outputs`.

- [ ] **Step 3: Add the shared taxonomy module**

Create `tldw_Server_API/app/core/Persona/visual_starter_recipe_taxonomy.py`:

```python
"""Shared taxonomy for Persona Visual starter production recipes.

The starter catalog exposes source asset groups and timed runtime animation
outputs as separate concepts. Keeping the taxonomy in this small module avoids
drift between immutable fixture validation and public API schema validation.
"""

BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS = frozenset(
    {
        "identity_brief",
        "neutral_anchor",
        "preview_image",
        "model_sheet",
        "static_talking_reaction_sheet",
        "required_state_loops",
        "animation_strips",
        "animation_atlas",
        "custom_state_variants",
    }
)
BUDDY_VISUAL_STATIC_SOURCE_ASSET_GROUP_IDS = frozenset(
    {
        "identity_brief",
        "neutral_anchor",
        "preview_image",
        "model_sheet",
        "static_talking_reaction_sheet",
    }
)
BUDDY_VISUAL_ANIMATION_OUTPUT_IDS = frozenset(
    {
        "required_state_loops",
        "animation_strips",
        "animation_atlas",
        "custom_state_variants",
    }
)

__all__ = [
    "BUDDY_VISUAL_ANIMATION_OUTPUT_IDS",
    "BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS",
    "BUDDY_VISUAL_STATIC_SOURCE_ASSET_GROUP_IDS",
]
```

`custom_state_variants` remains in `BUDDY_VISUAL_ANIMATION_OUTPUT_IDS` because
it means timed runtime loops or frame mappings for declared custom states, not
static source-sheet cells.

- [ ] **Step 4: Run tests and verify the remaining semantic failure**

Run the same pytest command from Step 2.

Expected: constants import, but the static-sheet separation test still fails until Task 2 updates recipe outputs.

## Task 2: Separate Static Sheets From Animation Outputs

**Files:**
- Modify: `tldw_Server_API/app/core/Persona/visual_starter_fixtures.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py`

- [ ] **Step 1: Update recipe outputs in fixtures**

In `_multi_asset_pack()`, change `animation_outputs` so it no longer includes `static_talking_reaction_sheet`:

```python
animation_outputs=(
    (
        "required_state_loops",
        "animation_strips",
        "animation_atlas",
        "custom_state_variants",
    )
    if complexity_tier == "intricate"
    else (
        "required_state_loops",
        "custom_state_variants",
    )
),
```

In `_atlas_pack()`, remove `static_talking_reaction_sheet` from `animation_outputs`:

```python
animation_outputs=(
    "required_state_loops",
    "animation_strips",
    "animation_atlas",
    "custom_state_variants",
),
```

Do not remove `static_talking_reaction_sheet` from `_INTERMEDIATE_EXPECTED_ASSET_GROUPS` or `_INTRICATE_EXPECTED_ASSET_GROUPS`.
Add `required_state_loops` to `_INTRICATE_EXPECTED_ASSET_GROUPS` so every
recipe `animation_outputs` value is also declared as an expected asset group.

- [ ] **Step 2: Update existing catalog expectations**

In `test_starter_pack_reports_production_readiness_metadata`, replace the current single `required_group` assertion with separate expected asset group and expected recipe output values.

Use this parametrization:

```python
@pytest.mark.parametrize(
    ("starter_pack_id", "complexity_tier", "required_group", "expected_output"),
    (
        ("research-buddy-basic", "basic", "required_state_loops", "required_state_loops"),
        (
            "study-desk-intermediate",
            "intermediate",
            "static_talking_reaction_sheet",
            "required_state_loops",
        ),
        ("lofi-study-intricate", "intricate", "animation_atlas", "animation_atlas"),
    ),
)
```

Then assert:

```python
assert required_group in detail["expected_asset_groups"]
_assert_recipe_shape(detail["production_recipe"], expected_output=expected_output)
```

- [ ] **Step 3: Run focused starter catalog tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py \
  -v
```

Expected: pass.

- [ ] **Step 4: Commit the fixture/test separation**

Run:

```bash
git add tldw_Server_API/app/core/Persona/visual_starter_recipe_taxonomy.py \
  tldw_Server_API/app/core/Persona/visual_starter_fixtures.py \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py
git commit -m "test: separate buddy static sheets from animation outputs"
```

## Task 3: Enforce Recipe Output Semantics At Catalog And API Boundaries

**Files:**
- Modify: `tldw_Server_API/app/core/Persona/visual_starter_catalog.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py`

- [ ] **Step 1: Write failing service and schema validation tests**

Add this service-level test near the malformed production metadata tests:

```python
def test_list_starter_packs_rejects_static_source_animation_outputs(
    db_instance: CharactersRAGDB,
) -> None:
    malformed = replace(
        DEFAULT_PERSONA_VISUAL_STARTER_PACKS[0],
        production_recipe=PersonaVisualStarterProductionRecipe(
            identity_brief="Identity",
            neutral_anchor="Neutral anchor",
            static_sheet="Static sheet",
            animation_outputs=("static_talking_reaction_sheet",),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_fixture"
    assert exc_info.value.details["field_name"] == "production_recipe.animation_outputs"
```

Add a sibling service-level test proving a timed output is still rejected when
the starter does not declare the corresponding expected asset group:

```python
def test_list_starter_packs_rejects_recipe_outputs_missing_expected_groups(
    db_instance: CharactersRAGDB,
) -> None:
    malformed = replace(
        DEFAULT_PERSONA_VISUAL_STARTER_PACKS[0],
        production_recipe=PersonaVisualStarterProductionRecipe(
            identity_brief="Identity",
            neutral_anchor="Neutral anchor",
            static_sheet="Static sheet",
            animation_outputs=("custom_state_variants",),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_fixture"
    assert exc_info.value.details["field_name"] == "production_recipe.animation_outputs"
    assert exc_info.value.details["invalid_outputs"] == ["custom_state_variants"]
```

Add a test near `test_production_recipe_response_enforces_catalog_bounds`:

```python
@pytest.mark.parametrize(
    "animation_outputs",
    (
        ["static_talking_reaction_sheet"],
        ["identity_brief"],
        ["neutral_anchor"],
        ["model_sheet"],
        ["unknown_output"],
    ),
)
def test_production_recipe_response_rejects_non_animation_outputs(
    animation_outputs: list[str],
) -> None:
    payload = _valid_recipe_payload()
    payload["animation_outputs"] = animation_outputs

    with pytest.raises(ValidationError):
        PersonaVisualStarterProductionRecipeResponse.model_validate(payload)
```

Also import `PersonaVisualStarterPackResponse` and add a response-boundary test
for unknown expected asset groups:

```python
def test_starter_pack_response_rejects_unknown_expected_asset_groups() -> None:
    payload = {
        "id": "starter-with-bad-group",
        "title": "Starter With Bad Group",
        "description": "Invalid starter metadata.",
        "renderer_type": "sprite_frames",
        "expected_asset_groups": ["neutral_anchor", "unknown_group"],
        "production_recipe": _valid_recipe_payload(),
    }

    with pytest.raises(ValidationError):
        PersonaVisualStarterPackResponse.model_validate(payload)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_list_starter_packs_rejects_static_source_animation_outputs \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_production_recipe_response_rejects_non_animation_outputs \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_list_starter_packs_rejects_recipe_outputs_missing_expected_groups \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py::test_starter_pack_response_rejects_unknown_expected_asset_groups \
  -v
```

Expected: fail because the catalog service and schema currently accept any
bounded non-empty recipe item.

- [ ] **Step 3: Add catalog service validation**

In `tldw_Server_API/app/core/Persona/visual_starter_catalog.py`, import:

```python
from tldw_Server_API.app.core.Persona.visual_starter_recipe_taxonomy import (
    BUDDY_VISUAL_ANIMATION_OUTPUT_IDS,
    BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS,
)
```

In `_validate_starter_fixture()`, after `expected_asset_groups` is normalized,
reject unknown expected asset groups:

```python
        invalid_expected_groups = sorted(
            group
            for group in expected_asset_groups
            if group not in BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS
        )
        if invalid_expected_groups:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter expected_asset_groups must use supported pipeline group ids.",
                details={
                    "starter_pack_id": starter.id,
                    "field_name": "expected_asset_groups",
                    "invalid_groups": invalid_expected_groups,
                },
            )
```

After `animation_outputs` is normalized in `_starter_production_recipe()`, reject
unknown or static/source outputs:

```python
        invalid_outputs = sorted(
            output
            for output in animation_outputs
            if output not in BUDDY_VISUAL_ANIMATION_OUTPUT_IDS
        )
        if invalid_outputs:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter production recipe animation_outputs must be timed output ids.",
                details={
                    "starter_pack_id": starter_id,
                    "field_name": "production_recipe.animation_outputs",
                    "invalid_outputs": invalid_outputs,
                },
            )
```

Then, back in `_validate_starter_fixture()`, after normalizing the production
recipe, reject timed outputs that are not also declared in `expected_asset_groups`:

```python
        production_recipe = PersonaVisualStarterCatalogService._starter_production_recipe(
            starter.production_recipe,
            starter_id=starter.id,
        )
        missing_expected_outputs = sorted(
            output
            for output in production_recipe["animation_outputs"]
            if output not in expected_asset_groups
        )
        if missing_expected_outputs:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter production recipe outputs must be declared expected asset groups.",
                details={
                    "starter_pack_id": starter.id,
                    "field_name": "production_recipe.animation_outputs",
                    "invalid_outputs": missing_expected_outputs,
                },
            )
```

Avoid calling `_starter_production_recipe()` twice in the same validation path;
reuse the normalized value for the consistency check.

- [ ] **Step 4: Add schema validator using the shared taxonomy**

In `tldw_Server_API/app/api/v1/schemas/persona.py`, import:

```python
from tldw_Server_API.app.core.Persona.visual_starter_recipe_taxonomy import (
    BUDDY_VISUAL_ANIMATION_OUTPUT_IDS,
    BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS,
)
```

Add a validator to `PersonaVisualStarterProductionRecipeResponse`:

```python
    @field_validator("animation_outputs")
    @classmethod
    def validate_animation_outputs(cls, value: list[str]) -> list[str]:
        invalid = sorted(
            output for output in value if output not in BUDDY_VISUAL_ANIMATION_OUTPUT_IDS
        )
        if invalid:
            raise ValueError(
                "animation_outputs must contain timed animation output ids only"
            )
        return value
```

Add a validator to `PersonaVisualStarterPackResponse`:

```python
    @field_validator("expected_asset_groups")
    @classmethod
    def validate_expected_asset_groups(cls, value: list[str]) -> list[str]:
        invalid = sorted(
            group for group in value if group not in BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS
        )
        if invalid:
            raise ValueError("expected_asset_groups must contain supported pipeline group ids")
        return value
```

Do not import from `visual_starter_fixtures.py`; the shared taxonomy module is
the stable dependency.

- [ ] **Step 5: Run focused schema/catalog tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py \
  -v
```

Expected: pass.

- [ ] **Step 6: Commit catalog and schema validation**

Run:

```bash
git add tldw_Server_API/app/core/Persona/visual_starter_recipe_taxonomy.py \
  tldw_Server_API/app/core/Persona/visual_starter_catalog.py \
  tldw_Server_API/app/api/v1/schemas/persona.py \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py
git commit -m "fix: bound buddy starter recipe animation outputs"
```

## Task 4: Update API, Job, And Provenance Tests For Valid Recipe Outputs

**Files:**
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_jobs.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_candidate_provenance.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py`

- [ ] **Step 1: Find stale static recipe-output assumptions**

Run:

```bash
rg -n '"static_sheet"|"static_talking_reaction_sheet"|recipe_output' \
  tldw_Server_API/tests/Persona/test_persona_visuals_api.py \
  tldw_Server_API/tests/Persona/test_persona_visual_jobs.py \
  tldw_Server_API/tests/Persona/test_persona_visual_candidate_provenance.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py
```

Expected: identify any test fixtures that use `recipe_output: "static_sheet"` or expect `static_talking_reaction_sheet` as an animation output.

- [ ] **Step 2: Update invalid job/API fixtures**

Replace invalid recipe outputs with a valid timed output such as `required_state_loops` unless the test specifically checks invalid-output rejection.

If a test intentionally validates invalid recipe output handling, prefer `not_a_recipe_output` so the failure reason remains clearly invalid, not a now-static source group.

For any API assertion that verifies static sheet availability, assert it through:

```python
assert "static_talking_reaction_sheet" in starter["expected_asset_groups"]
assert "static" in starter["production_recipe"]["static_sheet"].lower()
assert "static_talking_reaction_sheet" not in starter["production_recipe"]["animation_outputs"]
```

- [ ] **Step 3: Run focused API/job tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visuals_api.py \
  tldw_Server_API/tests/Persona/test_persona_visual_jobs.py \
  tldw_Server_API/tests/Persona/test_persona_visual_candidate_provenance.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py \
  -v
```

Expected: pass. If unrelated existing failures appear, record them in the Backlog task and rerun the narrow tests touched by this slice.

- [ ] **Step 4: Commit API/job test alignment**

Run:

```bash
git add tldw_Server_API/tests/Persona/test_persona_visuals_api.py \
  tldw_Server_API/tests/Persona/test_persona_visual_jobs.py \
  tldw_Server_API/tests/Persona/test_persona_visual_candidate_provenance.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py
git commit -m "test: align buddy recipe output consumers"
```

## Task 5: Update Documentation

**Files:**
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Modify: implementation Backlog task for this slice (`TASK-411` when present,
  otherwise `TASK-410`)

- [ ] **Step 1: Patch Persona Visual Packs documentation**

In the "Bundled Starter Catalog Scaffolds" section, update the production recipe explanation to state:

- `expected_asset_groups` may include source groups such as `static_talking_reaction_sheet`.
- `production_recipe.animation_outputs` only names timed runtime outputs.
- Static talking/reaction sheets become animation only when cells are explicitly mapped into manifest `animations` frames.

Keep wording explicit that current PNGs are scaffolds and not final animation packs.

- [ ] **Step 2: Add documentation assertions if an existing docs test covers this file**

Run:

```bash
rg -n "Persona_Visual_Packs|Bundled Starter Catalog|static_talking_reaction_sheet" tldw_Server_API/tests Docs
```

If there is no existing docs test for this exact file, do not add a new broad docs-test harness in this slice. The source and API tests are enough.

- [ ] **Step 3: Update Backlog task metadata**

Use the Backlog MCP task edit tool to add to the implementation task:

- plan path: `Docs/superpowers/plans/2026-05-16-buddy-animation-pipeline-catalog-metadata-plan.md`
- touched files from this implementation slice,
- verification commands and results,
- Bandit result or skip reason.

- [ ] **Step 4: Commit docs and Backlog updates**

Run:

```bash
git add Docs/Code_Documentation/Persona_Visual_Packs.md \
  "backlog/tasks/task-411 - Separate-Buddy-static-source-sheets-from-animation-outputs.md"
git commit -m "docs: clarify buddy static sheet recipe semantics"
```

If `TASK-411` is not present in the worktree, use the `TASK-410` design task
path instead and note the fallback in the task update.

## Task 6: Final Verification

**Files:**
- No new source files; validates the full touched scope.

- [ ] **Step 1: Run focused pytest suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Persona/test_persona_visual_starter_catalog.py \
  tldw_Server_API/tests/Persona/test_persona_visuals_api.py \
  tldw_Server_API/tests/Persona/test_persona_visual_jobs.py \
  tldw_Server_API/tests/Persona/test_persona_visual_candidate_provenance.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py \
  -v
```

Expected: pass.

- [ ] **Step 2: Run py_compile on touched Python modules**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile \
  tldw_Server_API/app/core/Persona/visual_starter_recipe_taxonomy.py \
  tldw_Server_API/app/core/Persona/visual_starter_catalog.py \
  tldw_Server_API/app/core/Persona/visual_starter_fixtures.py \
  tldw_Server_API/app/api/v1/schemas/persona.py
```

Expected: no output, exit 0.

- [ ] **Step 3: Run Bandit on touched Python modules**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r tldw_Server_API/app/core/Persona/visual_starter_recipe_taxonomy.py \
     tldw_Server_API/app/core/Persona/visual_starter_catalog.py \
     tldw_Server_API/app/core/Persona/visual_starter_fixtures.py \
     tldw_Server_API/app/api/v1/schemas/persona.py \
  -f json -o /tmp/bandit_buddy_animation_catalog_metadata.json
```

Expected: no new findings in touched code. If Bandit is not installed in the venv, record the environment skip in the Backlog task.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 5: Inspect git history and status**

Run:

```bash
git log --oneline origin/dev..HEAD
git status --short --branch
```

Expected: branch contains the design commit and the implementation commits; working tree is clean.

- [ ] **Step 6: Update issue #1787**

Comment on GitHub issue #1787 with:

- the committed implementation summary,
- verification results,
- any skipped checks and why,
- remaining next slices.

- [ ] **Step 7: Final commit if verification notes changed Backlog**

If the Backlog task changed during final verification, commit it:

```bash
git add "backlog/tasks/task-411 - Separate-Buddy-static-source-sheets-from-animation-outputs.md"
git commit -m "chore: record buddy catalog metadata verification"
```

If `TASK-411` is not present in the worktree, use the `TASK-410` design task
path instead and note the fallback in the commit summary.

## Completion Criteria

This slice is complete when:

- `static_talking_reaction_sheet` is present only as source/expected asset metadata, not as a recipe animation output.
- catalog service validation and API schema validation reject static/source
  groups in `animation_outputs`.
- starter catalog/API/job tests pass.
- docs explain the distinction clearly.
- issue #1787 and the implementation Backlog task record the verification.

## Out Of Scope For This Plan

- Generating the nine final Buddy art packs.
- Adding UI for neutral-anchor upload or production packets.
- Adding a new renderer.
- Bumping wire-level `manifest_version` for `sprite_frames`.
- Changing `BuddyShellHost` runtime state resolution.
