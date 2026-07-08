# Cooking Recipe Card Tool-Result UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only `cooking.recipe_card.render` MCP tool whose structured tool result renders as a shared recipe card UI in WebUI and extension chat.

**Architecture:** Implement one backend MCP module that validates recipe card input and returns a namespaced `tldw_ui` payload. Add a small shared frontend parser and `RecipeCard` component, then integrate it into the existing `ToolCallBlock` fallback path without touching OpenUI or adding generic UI rendering.

**Tech Stack:** FastAPI-side MCP Unified modules in Python, pytest/pytest-asyncio, React 18, TypeScript, Vitest, Testing Library, Tailwind classes, lucide-react.

---

## Spec

Approved design spec:

- `Docs/superpowers/specs/2026-07-05-cooking-recipe-card-tool-result-ui-design.md`

Backlog:

- `TASK-12150`

## Scope Check

This is one reviewable feature with two coupled slices:

- backend tool result producer
- frontend typed tool-result renderer

Do not split into a generic `ui.render` system, OpenUI changes, timers, grocery export, recipe persistence, or notification work.

## File Structure

Backend:

- Create `tldw_Server_API/app/core/MCP_unified/modules/implementations/cooking_module.py`
  - Owns `CookingModule`, `cooking.recipe_card.render`, validation helpers, and output normalization.
- Modify `tldw_Server_API/Config_Files/mcp_modules.yaml`
  - Adds enabled read-only `cooking` module entry.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py`
  - Unit tests for tool schema, happy path, bounds, invalid input, unknown tool.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`
  - Adds checked-in YAML coverage proving the safe `cooking` module is present and enabled while high-risk modules stay disabled.

Frontend:

- Create `apps/packages/ui/src/types/recipe-card.ts`
  - Shared TypeScript types for the normalized recipe card payload.
- Create `apps/packages/ui/src/utils/recipe-card-ui.ts`
  - Safe parser for `ToolCallResult.content`; returns `RecipeCardPayload | null`.
- Create `apps/packages/ui/src/utils/__tests__/recipe-card-ui.test.ts`
  - Parser tests for valid payload, malformed JSON, unsupported version, error results, oversized arrays.
- Create `apps/packages/ui/src/components/Common/RecipeCard/RecipeCard.tsx`
  - Shared compact recipe card and local serving stepper.
- Create `apps/packages/ui/src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx`
  - Component behavior tests.
- Modify `apps/packages/ui/src/components/Sidepanel/Chat/ToolCallBlock.tsx`
  - Adds label/icon for `cooking.recipe_card.render` and renders `RecipeCard` for valid non-error recipe tool results.
- Create `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx`
  - Integration tests for recipe rendering and fallback behavior.
- Create `apps/packages/ui/src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts`
  - Source/contract guard that WebUI and extension chat pass persisted `toolResults` into message rendering.

Verification:

- Backend targeted pytest.
- Frontend targeted Vitest.
- Bandit on touched backend module.
- Optional browser screenshot after implementation if the dev server is already available or can be started.

---

### Task 1: Backend Cooking Module Contract Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py`
- Later implementation: `tldw_Server_API/app/core/MCP_unified/modules/implementations/cooking_module.py`

- [ ] **Step 1: Write failing tests for tool metadata and happy path**

Create `test_cooking_module.py` with:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.cooking_module import (
    CookingModule,
)

pytestmark = pytest.mark.asyncio


def _module() -> CookingModule:
    return CookingModule(ModuleConfig(name="Cooking"))


def _recipe_args() -> dict:
    return {
        "title": "Cajun Alfredo Sauce",
        "servings": {"value": 2, "label": "2 servings"},
        "ingredients": [
            {
                "display": "3 tbsp butter",
                "name": "butter",
                "quantity": 3,
                "unit": "tbsp",
                "scalable": True,
            },
            {"display": "salt to taste", "scalable": False},
        ],
        "steps": [{"display": "Melt butter.", "timer_seconds": None}],
        "summary": "Rich sauce for pasta.",
        "notes": ["Add pasta water slowly."],
    }


async def test_get_tools_exposes_recipe_card_contract() -> None:
    tools = await _module().get_tools()

    assert [tool["name"] for tool in tools] == ["cooking.recipe_card.render"]  # nosec B101
    tool = tools[0]
    assert tool["inputSchema"]["required"] == ["title", "servings", "ingredients", "steps"]  # nosec B101
    assert tool["metadata"]["readOnlyHint"] is True  # nosec B101
    assert tool["metadata"]["category"] == "cooking"  # nosec B101


async def test_render_recipe_card_returns_tldw_ui_envelope() -> None:
    result = await _module().execute_tool("cooking.recipe_card.render", _recipe_args())

    assert set(result) == {"tldw_ui"}  # nosec B101
    envelope = result["tldw_ui"]
    assert envelope["kind"] == "recipe_card"  # nosec B101
    assert envelope["version"] == 1  # nosec B101
    recipe = envelope["recipe"]
    assert recipe["title"] == "Cajun Alfredo Sauce"  # nosec B101
    assert recipe["ingredients"][0]["display"] == "3 tbsp butter"  # nosec B101
    assert recipe["ingredients"][0]["quantity"] == 3  # nosec B101
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `cooking_module`.

- [ ] **Step 3: Commit only if this task is split into its own branch checkpoint**

Skip a commit here if implementing Task 2 immediately in the same working set.

---

### Task 2: Backend Cooking Module Implementation

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/cooking_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py`

- [ ] **Step 1: Implement minimal module shell and constants**

Use the existing `TemplateModule` pattern. Keep validation local to this module.

```python
from __future__ import annotations

from typing import Any

from ..base import BaseModule, create_tool_definition

TOOL_RECIPE_CARD = "cooking.recipe_card.render"
UI_KIND_RECIPE_CARD = "recipe_card"
UI_VERSION = 1

MAX_TITLE_CHARS = 120
MAX_SUMMARY_CHARS = 300
MAX_NOTE_CHARS = 300
MAX_NOTES = 8
MAX_INGREDIENTS = 60
MAX_INGREDIENT_DISPLAY_CHARS = 180
MAX_STEPS = 40
MAX_STEP_DISPLAY_CHARS = 600
MAX_TIMER_SECONDS = 86400
MAX_SERVINGS = 50


class CookingModule(BaseModule):
    """Read-only cooking UI helpers for MCP tool-call rendering."""

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name=TOOL_RECIPE_CARD,
                description=(
                    "Validate structured recipe data and return a tldw recipe "
                    "card UI payload. Use when an answer includes a cooking "
                    "recipe that should render as an inline recipe card."
                ),
                parameters=_recipe_card_parameters(),
                metadata={
                    "category": "cooking",
                    "readOnlyHint": True,
                    "auth_required": False,
                    "capabilities": ["ui.recipe_card", "cooking.recipe"],
                },
            )
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        if tool_name != TOOL_RECIPE_CARD:
            return {"ok": False, "reason_code": "unknown_tool", "error": f"Unknown tool: {tool_name}"}
        try:
            recipe = _normalize_recipe(self.sanitize_input(arguments))
        except ValueError as exc:
            return {"ok": False, "reason_code": "invalid_arguments", "error": str(exc)}
        return {"tldw_ui": {"kind": UI_KIND_RECIPE_CARD, "version": UI_VERSION, "recipe": recipe}}
```

- [ ] **Step 2: Add JSON schema function**

Keep the schema explicit. Do not add a dependency.

```python
def _recipe_card_parameters() -> dict[str, Any]:
    return {
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": MAX_TITLE_CHARS},
            "servings": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "minimum": 1, "maximum": MAX_SERVINGS},
                    "label": {"type": "string", "maxLength": 80},
                },
                "required": ["value"],
            },
            "ingredients": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_INGREDIENTS,
                "items": {
                    "type": "object",
                    "properties": {
                        "display": {"type": "string", "minLength": 1, "maxLength": MAX_INGREDIENT_DISPLAY_CHARS},
                        "name": {"type": "string", "maxLength": 120},
                        "quantity": {"type": "number"},
                        "unit": {"type": "string", "maxLength": 32},
                        "note": {"type": ["string", "null"], "maxLength": 160},
                        "scalable": {"type": "boolean"},
                    },
                    "required": ["display"],
                },
            },
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_STEPS,
                "items": {
                    "type": "object",
                    "properties": {
                        "display": {"type": "string", "minLength": 1, "maxLength": MAX_STEP_DISPLAY_CHARS},
                        "timer_seconds": {"type": ["integer", "null"], "minimum": 1, "maximum": MAX_TIMER_SECONDS},
                    },
                    "required": ["display"],
                },
            },
            "summary": {"type": ["string", "null"], "maxLength": MAX_SUMMARY_CHARS},
            "notes": {
                "type": "array",
                "maxItems": MAX_NOTES,
                "items": {"type": "string", "maxLength": MAX_NOTE_CHARS},
            },
        },
        "required": ["title", "servings", "ingredients", "steps"],
    }
```

- [ ] **Step 3: Add normalization helpers**

Implement boring validators. Preserve `display` as authoritative.

```python
def _clean_text(value: Any, *, field: str, max_chars: int, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if required and not text:
        raise ValueError(f"{field} is required")
    if len(text) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    return text or None
```

Add `_normalize_servings`, `_normalize_ingredients`, `_normalize_steps`, and `_normalize_recipe` with the limits from the spec.

- [ ] **Step 4: Run backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py -q
```

Expected: PASS.

- [ ] **Step 5: Add invalid-input tests**

Add parametrized tests for:

- empty title
- servings `0`
- more than 60 ingredients
- ingredient missing `display`
- more than 40 steps
- step text longer than 600 characters
- timer over 86400
- unknown tool

Expected assertion shape:

```python
result = await _module().execute_tool("cooking.recipe_card.render", bad_args)
assert result["ok"] is False  # nosec B101
assert result["reason_code"] == "invalid_arguments"  # nosec B101
```

- [ ] **Step 6: Run full backend module test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit backend module slice**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/cooking_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py
git commit -m "feat: add cooking recipe card MCP tool"
```

---

### Task 3: MCP Config Registration

**Files:**
- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py`

- [ ] **Step 1: Add failing config tests**

In `test_basic_functionality.py`, extend the existing YAML defaults test:

```python
def test_default_mcp_modules_yaml_includes_safe_cooking_module():
    import yaml
    from pathlib import Path

    data = yaml.safe_load(Path("tldw_Server_API/Config_Files/mcp_modules.yaml").read_text(encoding="utf-8"))
    modules = {entry["id"]: entry for entry in data["modules"]}

    assert modules["cooking"]["enabled"] is True  # nosec B101
    assert modules["cooking"]["class"].endswith("cooking_module:CookingModule")  # nosec B101
```

In `test_cooking_module.py`, add a normal server registration test:

```python
async def test_server_registers_cooking_module_from_default_yaml(monkeypatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    server = MCPServer()
    registrations = []

    async def _register_module(module_id, cls, config):
        registrations.append((module_id, cls, config))

    monkeypatch.setattr(server.module_registry, "register_module", _register_module)
    monkeypatch.delenv("MCP_MODULES_CONFIG", raising=False)
    monkeypatch.delenv("MCP_MODULES", raising=False)

    await server._register_default_modules()

    cooking = next(item for item in registrations if item[0] == "cooking")
    assert cooking[1].__name__ == "CookingModule"  # nosec B101
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_default_mcp_modules_yaml_includes_safe_cooking_module \
  tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py::test_server_registers_cooking_module_from_default_yaml \
  -q
```

Expected: FAIL because YAML lacks `cooking`.

- [ ] **Step 3: Add YAML module entry**

Add to `tldw_Server_API/Config_Files/mcp_modules.yaml` near other safe read-only/user-facing modules:

```yaml
  - id: cooking
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.cooking_module:CookingModule
    enabled: true
    name: Cooking
    version: "1.0.0"
    department: utility
    max_concurrent: 20
    description: Read-only cooking recipe card UI payload tools
    settings: {}
```

- [ ] **Step 4: Run targeted registration tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_default_mcp_modules_yaml_includes_safe_cooking_module \
  tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py::test_server_registers_cooking_module_from_default_yaml \
  -q
```

Expected: PASS.

- [ ] **Step 5: Run related MCP tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_default_mcp_modules_yaml_disables_local_file_and_process_modules \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_default_mcp_modules_yaml_includes_safe_cooking_module \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit config slice**

```bash
git add tldw_Server_API/Config_Files/mcp_modules.yaml \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py
git commit -m "feat: register cooking MCP module"
```

---

### Task 4: Frontend Recipe Payload Parser

**Files:**
- Create: `apps/packages/ui/src/types/recipe-card.ts`
- Create: `apps/packages/ui/src/utils/recipe-card-ui.ts`
- Create: `apps/packages/ui/src/utils/__tests__/recipe-card-ui.test.ts`

- [ ] **Step 1: Write parser tests first**

Create `recipe-card-ui.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import { parseRecipeCardToolResult } from "../recipe-card-ui"

const validContent = JSON.stringify({
  tldw_ui: {
    kind: "recipe_card",
    version: 1,
    recipe: {
      title: "Cajun Alfredo Sauce",
      servings: { value: 2, label: "2 servings" },
      ingredients: [
        {
          display: "3 tbsp butter",
          name: "butter",
          quantity: 3,
          unit: "tbsp",
          scalable: true
        },
        { display: "salt to taste", scalable: false }
      ],
      steps: [{ display: "Melt butter.", timer_seconds: null }],
      summary: null,
      notes: []
    }
  }
})

describe("parseRecipeCardToolResult", () => {
  it("parses a valid recipe card payload", () => {
    const parsed = parseRecipeCardToolResult({ content: validContent })
    expect(parsed?.recipe.title).toBe("Cajun Alfredo Sauce")
    expect(parsed?.recipe.ingredients[0].quantity).toBe(3)
  })

  it("returns null for errors, malformed JSON, and unsupported versions", () => {
    expect(parseRecipeCardToolResult({ content: validContent, error: true })).toBeNull()
    expect(parseRecipeCardToolResult({ content: "not json" })).toBeNull()
    expect(
      parseRecipeCardToolResult({
        content: JSON.stringify({ tldw_ui: { kind: "recipe_card", version: 2, recipe: {} } })
      })
    ).toBeNull()
  })
})
```

- [ ] **Step 2: Run parser tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/utils/__tests__/recipe-card-ui.test.ts
```

Expected: FAIL because files do not exist.

- [ ] **Step 3: Add shared types**

Create `recipe-card.ts`:

```ts
export type RecipeCardIngredient = {
  display: string
  name?: string
  quantity?: number
  unit?: string
  note?: string | null
  scalable?: boolean
}

export type RecipeCardStep = {
  display: string
  timer_seconds?: number | null
}

export type RecipeCardPayload = {
  kind: "recipe_card"
  version: 1
  recipe: {
    title: string
    servings: {
      value: number
      label?: string
    }
    ingredients: RecipeCardIngredient[]
    steps: RecipeCardStep[]
    summary?: string | null
    notes?: string[]
  }
}
```

- [ ] **Step 4: Implement parser with caps**

Create `recipe-card-ui.ts`. Keep this dependency-free.

```ts
import type { ToolCallResult } from "@/types/tool-calls"
import type { RecipeCardPayload } from "@/types/recipe-card"

const MAX_INGREDIENTS = 60
const MAX_STEPS = 40

export const parseRecipeCardToolResult = (
  result?: Pick<ToolCallResult, "content" | "error">
): RecipeCardPayload | null => {
  if (!result || result.error) return null
  try {
    const parsed = JSON.parse(result.content)
    const ui = parsed?.tldw_ui
    if (!ui || ui.kind !== "recipe_card" || ui.version !== 1) return null
    const recipe = ui.recipe
    if (!recipe || typeof recipe.title !== "string") return null
    if (!Array.isArray(recipe.ingredients) || recipe.ingredients.length === 0) return null
    if (!Array.isArray(recipe.steps) || recipe.steps.length === 0) return null
    if (recipe.ingredients.length > MAX_INGREDIENTS || recipe.steps.length > MAX_STEPS) return null
    return ui as RecipeCardPayload
  } catch {
    return null
  }
}
```

Add stricter string/number checks after the first passing test. Do not parse ingredient strings.

- [ ] **Step 5: Run parser tests**

```bash
bunx vitest run src/utils/__tests__/recipe-card-ui.test.ts
```

Expected: PASS.

- [ ] **Step 6: Add bounds tests**

Add tests for:

- missing `tldw_ui`
- wrong `kind`
- empty ingredients
- too many ingredients
- too many steps
- non-string ingredient display
- invalid servings value

- [ ] **Step 7: Run parser tests again**

```bash
bunx vitest run src/utils/__tests__/recipe-card-ui.test.ts
```

Expected: PASS.

- [ ] **Step 8: Commit parser slice**

```bash
git add apps/packages/ui/src/types/recipe-card.ts \
  apps/packages/ui/src/utils/recipe-card-ui.ts \
  apps/packages/ui/src/utils/__tests__/recipe-card-ui.test.ts
git commit -m "feat: parse recipe card tool results"
```

---

### Task 5: Shared RecipeCard Component

**Files:**
- Create: `apps/packages/ui/src/components/Common/RecipeCard/RecipeCard.tsx`
- Create: `apps/packages/ui/src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx`

- [ ] **Step 1: Write component tests**

Create `RecipeCard.test.tsx`:

```tsx
// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"
import { RecipeCard } from "../RecipeCard"
import type { RecipeCardPayload } from "@/types/recipe-card"

const payload: RecipeCardPayload = {
  kind: "recipe_card",
  version: 1,
  recipe: {
    title: "Cajun Alfredo Sauce",
    servings: { value: 2, label: "2 servings" },
    ingredients: [
      { display: "3 tbsp butter", quantity: 3, unit: "tbsp", name: "butter", scalable: true },
      { display: "salt to taste", scalable: false }
    ],
    steps: [
      { display: "Melt butter.", timer_seconds: null },
      { display: "Simmer for 5 minutes.", timer_seconds: 300 }
    ],
    notes: []
  }
}

describe("RecipeCard", () => {
  it("renders recipe title, counts, ingredients, and steps", () => {
    render(<RecipeCard payload={payload} />)
    expect(screen.getByText("Cajun Alfredo Sauce")).toBeInTheDocument()
    expect(screen.getByText("2 ingredients")).toBeInTheDocument()
    expect(screen.getByText("2 steps")).toBeInTheDocument()
    expect(screen.getByText("3 tbsp butter")).toBeInTheDocument()
    expect(screen.getByText("salt to taste")).toBeInTheDocument()
  })

  it("scales only structured scalable ingredients", async () => {
    const user = userEvent.setup()
    render(<RecipeCard payload={payload} />)
    await user.click(screen.getByRole("button", { name: "Increase servings" }))
    expect(screen.getByText("4 servings")).toBeInTheDocument()
    expect(screen.getByText("6 tbsp butter")).toBeInTheDocument()
    expect(screen.getByText("salt to taste")).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run component tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx
```

Expected: FAIL because component does not exist.

- [ ] **Step 3: Implement `RecipeCard`**

Use lucide icons, native buttons, and text rendering only. Avoid nested cards. Example structure:

```tsx
import React from "react"
import { ChefHat, ListChecks, Minus, Plus } from "lucide-react"
import { classNames } from "@/libs/class-name"
import type { RecipeCardPayload, RecipeCardIngredient } from "@/types/recipe-card"

type RecipeCardProps = {
  payload: RecipeCardPayload
  className?: string
}

export const RecipeCard: React.FC<RecipeCardProps> = ({ payload, className }) => {
  const { recipe } = payload
  const baseServings = recipe.servings.value
  const [servings, setServings] = React.useState(baseServings)
  const factor = servings / baseServings

  return (
    <section className={classNames("rounded-md border border-border bg-surface px-3 py-3", className)}>
      <div className="flex items-start gap-2">
        <ChefHat className="mt-0.5 size-4 text-primary" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-text">{recipe.title}</h3>
          <p className="text-xs text-text-muted">
            {recipe.ingredients.length} ingredients - {recipe.steps.length} steps
          </p>
        </div>
      </div>
      {/* servings controls, ingredients, cooking mode */}
    </section>
  )
}
```

Use simple pluralization inline. Do not add a new i18n namespace in this slice.

- [ ] **Step 4: Add scaling helper inside component file**

```ts
const formatQuantity = (value: number) =>
  Number.isInteger(value) ? String(value) : String(Math.round(value * 100) / 100)

const formatIngredient = (ingredient: RecipeCardIngredient, factor: number) => {
  if (
    ingredient.scalable &&
    typeof ingredient.quantity === "number" &&
    Number.isFinite(ingredient.quantity) &&
    ingredient.unit &&
    ingredient.name
  ) {
    const scaled = ingredient.quantity * factor
    return `${formatQuantity(scaled)} ${ingredient.unit} ${ingredient.name}`
  }
  return ingredient.display
}
```

- [ ] **Step 5: Implement inline cooking mode**

Add a `Cooking mode` button that toggles a compact ordered step view. It does not start timers. Durations render as text.

```tsx
const [showSteps, setShowSteps] = React.useState(false)

<button type="button" onClick={() => setShowSteps((value) => !value)}>
  <ListChecks className="size-4" aria-hidden="true" />
  Cooking mode
</button>
```

- [ ] **Step 6: Run component tests**

```bash
bunx vitest run src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit component slice**

```bash
git add apps/packages/ui/src/components/Common/RecipeCard/RecipeCard.tsx \
  apps/packages/ui/src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx
git commit -m "feat: add recipe card component"
```

---

### Task 6: ToolCallBlock Integration

**Files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/ToolCallBlock.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx`

- [ ] **Step 1: Write integration tests**

Create `ToolCallBlock.recipe-card.test.tsx`:

```tsx
// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ToolCallBlock } from "../ToolCallBlock"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

const toolCall = {
  id: "call_recipe",
  type: "function" as const,
  function: {
    name: "cooking.recipe_card.render",
    arguments: JSON.stringify({ title: "Cajun Alfredo Sauce" })
  }
}

const recipeResult = {
  tool_call_id: "call_recipe",
  content: JSON.stringify({
    tldw_ui: {
      kind: "recipe_card",
      version: 1,
      recipe: {
        title: "Cajun Alfredo Sauce",
        servings: { value: 2, label: "2 servings" },
        ingredients: [{ display: "3 tbsp butter", quantity: 3, unit: "tbsp", name: "butter", scalable: true }],
        steps: [{ display: "Melt butter.", timer_seconds: null }],
        summary: null,
        notes: []
      }
    }
  })
}

describe("ToolCallBlock recipe card rendering", () => {
  it("renders a valid recipe card tool result without expanding raw JSON", () => {
    render(<ToolCallBlock toolCalls={[toolCall]} results={[recipeResult]} />)
    expect(screen.getByText("Recipe Card")).toBeInTheDocument()
    expect(screen.getByText("Cajun Alfredo Sauce")).toBeInTheDocument()
    expect(screen.queryByText(/\"tldw_ui\"/)).not.toBeInTheDocument()
  })

  it("falls back for failed recipe tool results", () => {
    render(<ToolCallBlock toolCalls={[toolCall]} results={[{ ...recipeResult, error: true }]} />)
    expect(screen.queryByText("Cajun Alfredo Sauce")).not.toBeInTheDocument()
    expect(screen.getByText("Error")).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx
```

Expected: FAIL because recipe rendering is not integrated.

- [ ] **Step 3: Add label/icon mapping**

Modify imports and maps in `ToolCallBlock.tsx`:

```ts
import { ChefHat, ... } from "lucide-react"

const TOOL_ICONS = {
  ...,
  "cooking.recipe_card.render": ChefHat
}

const TOOL_LABELS = {
  ...,
  "cooking.recipe_card.render": "Recipe Card"
}
```

- [ ] **Step 4: Integrate parser and component**

Inside `toolCalls.map`, compute:

```ts
const recipePayload = parseRecipeCardToolResult(result)
```

Render special body when present:

```tsx
{recipePayload && (
  <div className="border-t border-border/40 px-2 py-2">
    <RecipeCard payload={recipePayload} />
  </div>
)}
```

Keep the existing expanded generic content for arguments and fallback results. For recipe payloads, expanded content may show arguments only; do not duplicate raw `tldw_ui` JSON unless the recipe parser returned `null`.

- [ ] **Step 5: Run integration tests**

```bash
bunx vitest run src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Run parser and component tests together**

```bash
bunx vitest run \
  src/utils/__tests__/recipe-card-ui.test.ts \
  src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx \
  src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit integration slice**

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/ToolCallBlock.tsx \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx
git commit -m "feat: render recipe card tool results"
```

---

### Task 7: Tool Result Replay Guard

**Files:**
- Create: `apps/packages/ui/src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts`
- Modify only if needed:
  - `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx`
  - `apps/packages/ui/src/components/Sidepanel/Chat/body.tsx`
  - relevant message/store adapters found during implementation

- [ ] **Step 1: Write a guard for the two MVP surfaces**

Create a source guard that documents the replay dependency:

```ts
import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (relativePath: string) =>
  fs.readFileSync(path.resolve(__dirname, relativePath), "utf8")

describe("tool result replay wiring", () => {
  it("passes persisted toolResults through WebUI chat messages", () => {
    const source = readSource("../../../Option/Playground/PlaygroundChat.tsx")
    expect(source).toContain("toolResults={message?.toolResults}")
  })

  it("passes persisted toolResults through extension sidepanel messages", () => {
    const source = readSource("../../../Sidepanel/Chat/body.tsx")
    expect(source).toContain("toolResults={message?.toolResults}")
  })
})
```

- [ ] **Step 2: Run guard**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts
```

Expected: PASS if existing wiring is intact.

- [ ] **Step 3: If the guard fails, add the smallest adapter**

Only touch message plumbing if a target surface drops `toolResults`. Keep the change local. Do not redesign chat persistence in this feature.

- [ ] **Step 4: Run the guard and integration tests**

```bash
bunx vitest run \
  src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts \
  src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit replay guard**

```bash
git add apps/packages/ui/src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts
git commit -m "test: guard recipe card tool result replay"
```

---

### Task 8: Verification And Hardening

**Files:**
- Update if needed: files touched in earlier tasks only.
- Update: `backlog/tasks/task-12150 - Design-cooking-recipe-card-tool-result-UI.md`

- [ ] **Step 1: Run backend targeted tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_default_mcp_modules_yaml_disables_local_file_and_process_modules \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_default_mcp_modules_yaml_includes_safe_cooking_module \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run frontend targeted tests**

From `apps/packages/ui`:

```bash
bunx vitest run \
  src/utils/__tests__/recipe-card-ui.test.ts \
  src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx \
  src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx \
  src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched backend code**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/cooking_module.py \
  -f json -o /tmp/bandit_cooking_recipe_card.json
```

Expected: PASS or only non-actionable findings outside changed code. Fix any new finding in `cooking_module.py`.

- [ ] **Step 4: Run TypeScript typecheck if targeted tests pass**

From `apps/tldw-frontend`:

```bash
bun run typecheck
```

Expected: PASS. If this is too slow or blocked by existing unrelated errors, record the failure and run the targeted Vitest suite as the required frontend verification.

- [ ] **Step 5: Visual smoke check**

If a dev server is already running, open a page with a fixture or temporary test harness that renders `ToolCallBlock` with a recipe payload. Verify:

- the recipe card is visible without opening raw JSON
- the card fits a narrow sidepanel width
- serving controls do not shift layout badly
- cooking mode expands inline

If no dev server is available, rely on the Testing Library coverage and document that browser visual QA was not run.

- [ ] **Step 6: Update Backlog task**

Record:

- files changed
- test commands and results
- Bandit result path
- any skipped visual QA reason

- [ ] **Step 7: Final commit**

If earlier tasks were not committed individually, commit all remaining implementation files:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/cooking_module.py \
  tldw_Server_API/Config_Files/mcp_modules.yaml \
  tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  apps/packages/ui/src/types/recipe-card.ts \
  apps/packages/ui/src/utils/recipe-card-ui.ts \
  apps/packages/ui/src/utils/__tests__/recipe-card-ui.test.ts \
  apps/packages/ui/src/components/Common/RecipeCard/RecipeCard.tsx \
  apps/packages/ui/src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx \
  apps/packages/ui/src/components/Sidepanel/Chat/ToolCallBlock.tsx \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx \
  apps/packages/ui/src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts \
  "backlog/tasks/task-12150 - Design-cooking-recipe-card-tool-result-UI.md"
git commit -m "feat: render cooking recipe card tool results"
```

---

## Final Verification Checklist

- [ ] `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py -q`
- [ ] `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_default_mcp_modules_yaml_includes_safe_cooking_module -q`
- [ ] `bunx vitest run src/utils/__tests__/recipe-card-ui.test.ts src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts`
- [ ] `python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/cooking_module.py -f json -o /tmp/bandit_cooking_recipe_card.json`
- [ ] Typecheck or documented targeted-test fallback
- [ ] Browser visual QA or documented skip
- [ ] Backlog `TASK-12150` updated
