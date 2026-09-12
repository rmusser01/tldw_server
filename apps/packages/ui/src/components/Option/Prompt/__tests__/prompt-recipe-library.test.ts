import { describe, expect, it } from "vitest"
import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
import type { SingleTextRecipeDefinitionV2 } from "../structured-prompt-utils"
import {
  buildRecipePromptFields,
  classifyPromptRecipe,
  cloneSavedRecipeSource,
  getRecipePersistenceState,
  matchesRecipeTarget
} from "../prompt-recipe-library"

const recipeRecord = (overrides: Record<string, unknown> = {}) => ({
  id: "recipe-17",
  name: "Clear task",
  title: "Clear task",
  promptFormat: "structured",
  promptSchemaVersion: 2,
  structuredPromptDefinition: structuredClone(CLEAR_TASK_RECIPE.definition),
  content: "old snapshot",
  system_prompt: "old snapshot",
  user_prompt: "",
  ...overrides
})

describe("prompt recipe library identity", () => {
  it.each([
    [false, undefined, "offline"],
    [true, undefined, "Checking"],
    [
      true,
      {
        availability: "available",
        prompt_improvement_v1: { supported: false, limits: null },
        single_text_recipe_v2: { supported: false }
      },
      "does not support"
    ],
    [
      true,
      {
        availability: "unavailable",
        prompt_improvement_v1: { supported: false, limits: null },
        single_text_recipe_v2: { supported: false }
      },
      "could not be confirmed"
    ]
  ] as const)(
    "disables persistence for offline, unknown, or unsupported capability",
    (isOnline, capabilities, reason) => {
      expect(getRecipePersistenceState(isOnline, capabilities)).toMatchObject({
        available: false,
        reason: expect.stringContaining(reason)
      })
    }
  )

  it("enables persistence only after positive v2 advertisement", () => {
    expect(
      getRecipePersistenceState(true, {
        availability: "available",
        prompt_improvement_v1: { supported: false, limits: null },
        single_text_recipe_v2: { supported: true }
      })
    ).toEqual({ available: true, reason: "" })
  })

  it("recognizes only a strictly validated v2 recipe and exposes its target", () => {
    expect(classifyPromptRecipe(recipeRecord())).toMatchObject({
      kind: "recipe",
      target: "system"
    })
  })

  it.each([
    { promptSchemaVersion: 99 },
    { promptSchemaVersion: 1 },
    {
      structuredPromptDefinition: {
        ...CLEAR_TASK_RECIPE.definition,
        definition_kind: "future_recipe"
      }
    },
    {
      structuredPromptDefinition: {
        ...CLEAR_TASK_RECIPE.definition,
        blocks: [
          {
            ...CLEAR_TASK_RECIPE.definition.blocks![0],
            role: "user"
          }
        ]
      }
    }
  ])(
    "quarantines invalid, old, future, or mismatched v2 identity",
    (override) => {
      expect(classifyPromptRecipe(recipeRecord(override))).toEqual({
        kind: "quarantined_recipe"
      })
    }
  )

  it("leaves legacy, v1 structured, system, and quick prompts on their existing paths", () => {
    expect(classifyPromptRecipe({ id: "quick", content: "hello" })).toEqual({
      kind: "ordinary"
    })
    expect(
      classifyPromptRecipe({
        id: "v1",
        promptFormat: "structured",
        promptSchemaVersion: 1,
        structuredPromptDefinition: {
          schema_version: 1,
          format: "structured",
          variables: [],
          blocks: [],
          assembly_config: {
            legacy_system_roles: ["system"],
            legacy_user_roles: ["user"],
            block_separator: "\n\n"
          }
        }
      })
    ).toEqual({ kind: "ordinary" })
  })

  it("filters recipes by their authored target without including ordinary prompts", () => {
    const systemRecipe = recipeRecord()
    const userRecipe = recipeRecord({
      id: "user-recipe",
      structuredPromptDefinition: {
        ...CLEAR_TASK_RECIPE.definition,
        assembly_config: {
          ...CLEAR_TASK_RECIPE.definition.assembly_config,
          target_role: "user"
        },
        blocks: CLEAR_TASK_RECIPE.definition.blocks!.map((block) => ({
          ...block,
          role: "user"
        }))
      }
    })

    expect(matchesRecipeTarget(systemRecipe, "recipe")).toBe(true)
    expect(matchesRecipeTarget(systemRecipe, "recipe_system")).toBe(true)
    expect(matchesRecipeTarget(systemRecipe, "recipe_user")).toBe(false)
    expect(matchesRecipeTarget(userRecipe, "recipe_user")).toBe(true)
    expect(
      matchesRecipeTarget({ id: "quick", content: "hello" }, "recipe")
    ).toBe(false)
  })

  it("deep-clones a saved source so opening and editing cannot mutate the record", () => {
    const record = recipeRecord()
    const source = cloneSavedRecipeSource(record)

    expect(source).toMatchObject({
      source_kind: "saved",
      id: "recipe-17",
      name: "Clear task"
    })
    ;(source.definition as SingleTextRecipeDefinitionV2).blocks![0].content =
      "edited copy"
    expect(
      (record.structuredPromptDefinition as SingleTextRecipeDefinitionV2)
        .blocks![0].content
    ).not.toBe("edited copy")
  })
})

describe("recipe persistence snapshots", () => {
  it.each(["system", "user"] as const)(
    "builds strict %s-target storage fields from authored template text only",
    (target) => {
      const definition = structuredClone(CLEAR_TASK_RECIPE.definition)
      definition.assembly_config.target_role = target
      definition.blocks = definition.blocks!.map((block) => ({
        ...block,
        role: target,
        content:
          block.id === "objective" ? "Complete {{ task }} now" : block.content
      }))
      definition.variables![0].default_value =
        "MUST_NOT_BE_PERSISTED_IN_SNAPSHOT"

      const fields = buildRecipePromptFields(definition)
      const expected = fields.content

      expect(fields).toMatchObject({
        promptFormat: "structured",
        promptSchemaVersion: 2,
        is_system: target === "system",
        content: expected,
        system_prompt: target === "system" ? expected : "",
        user_prompt: target === "user" ? expected : ""
      })
      expect(expected).toContain("{{ task }}")
      expect(expected).not.toContain("MUST_NOT_BE_PERSISTED_IN_SNAPSHOT")
      expect(JSON.stringify(fields)).not.toMatch(
        /runtimeValues|runtime_values|variable_values|resolved_values/
      )
    }
  )

  it("returns a canonical deep clone rather than the editor-owned definition", () => {
    const definition = structuredClone(CLEAR_TASK_RECIPE.definition)
    const fields = buildRecipePromptFields(definition)

    fields.structuredPromptDefinition.blocks![0].content =
      "mutated persisted copy"
    expect(definition.blocks![0].content).not.toBe("mutated persisted copy")
  })
})
