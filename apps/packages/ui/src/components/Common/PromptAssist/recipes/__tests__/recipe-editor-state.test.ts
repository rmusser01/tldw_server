import { describe, expect, it } from "vitest";

import { CLEAR_TASK_RECIPE } from "../built-in-recipes";
import {
  canApplyRecipe,
  canSaveRecipeAsNew,
  canUpdateRecipe,
  createRecipeWorkingCopy,
  parseRecipeDefinition,
  proposeRecipeSectionKey,
  recipeEditorReducer,
  serializeRecipeDefinitionForSave,
} from "../recipe-editor-state";
import type {
  RecipeEditorState,
  SingleTextRecipeDefinition,
  SingleTextRecipeVariable,
} from "../types";

const reduce = (
  state: RecipeEditorState,
  ...actions: Parameters<typeof recipeEditorReducer>[1][]
): RecipeEditorState => actions.reduce(recipeEditorReducer, state);

const audienceVariable: SingleTextRecipeVariable = {
  name: "audience",
  label: "Audience",
  description: null,
  required: true,
  default_value: null,
  input_type: "text",
  options: null,
  max_length: null,
};

describe("recipe editor state", () => {
  it("maps UI targets explicitly to persisted roles", () => {
    const system = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const user = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "user_message");

    expect(system.definition.assembly_config.target_role).toBe("system");
    expect(system.definition.blocks?.map((block) => block.role)).toEqual([
      "system",
      "system",
      "system",
      "system",
    ]);
    expect(user.definition.assembly_config.target_role).toBe("user");
    expect(user.definition.blocks?.map((block) => block.role)).toEqual([
      "user",
      "user",
      "user",
      "user",
    ]);
  });

  it("rejects unvalidated stored JSON before creating editor state", () => {
    const invalidStoredRecipe = {
      ...CLEAR_TASK_RECIPE.definition,
      runtime_values: { task: "private" },
    };

    expect(() =>
      createRecipeWorkingCopy(
        {
          source_kind: "saved",
          id: "saved-1",
          name: "Saved recipe",
          definition: invalidStoredRecipe,
        },
        "system",
      ),
    ).toThrow("invalid_recipe_runtime_values");
  });

  it("adds a stable target-role block and proposes its section key once", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "user_message");
    const added = recipeEditorReducer(initial, {
      type: "block_added",
      block: {
        id: "success_criteria",
        name: "Success criteria!",
        content: "Define done.",
      },
    });

    expect(added.definition.blocks?.at(-1)).toEqual({
      id: "success_criteria",
      name: "Success criteria!",
      section_key: "success_criteria",
      role: "user",
      kind: null,
      content: "Define done.",
      enabled: true,
      order: 50,
      is_template: false,
    });

    const renamed = recipeEditorReducer(added, {
      type: "block_updated",
      blockId: "success_criteria",
      changes: { name: "Definition of done" },
    });
    expect(renamed.definition.blocks?.at(-1)).toMatchObject({
      name: "Definition of done",
      section_key: "success_criteria",
    });
  });

  it("proposes conservative, unique XML section keys", () => {
    expect(proposeRecipeSectionKey("  Source rules & evidence  ")).toBe(
      "source_rules_evidence",
    );
    expect(proposeRecipeSectionKey("123")).toBe("_123");
    expect(proposeRecipeSectionKey("資料")).toBe("section");
    expect(
      proposeRecipeSectionKey("Output", new Set(["output", "output_2"])),
    ).toBe("output_3");
  });

  it("removes, renames, reorders, and toggles blocks without changing IDs", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const edited = reduce(
      initial,
      {
        type: "block_updated",
        blockId: "context_inputs",
        changes: { name: "Inputs" },
      },
      { type: "block_toggled", blockId: "constraints" },
      { type: "block_reordered", blockId: "output", toIndex: 0 },
      { type: "block_removed", blockId: "context_inputs" },
    );

    expect(
      edited.definition.blocks?.map(({ id, order }) => [id, order]),
    ).toEqual([
      ["output", 10],
      ["objective", 20],
      ["constraints", 40],
    ]);
    expect(
      edited.definition.blocks?.find((block) => block.id === "constraints")
        ?.enabled,
    ).toBe(false);
    expect(
      edited.definition.blocks?.find((block) => block.id === "objective")?.id,
    ).toBe("objective");
  });

  it("switches render format without inventing or rewriting section keys", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const withoutKey = recipeEditorReducer(initial, {
      type: "block_updated",
      blockId: "objective",
      changes: { section_key: null },
    });
    const markdown = recipeEditorReducer(withoutKey, {
      type: "format_changed",
      renderFormat: "markdown",
    });
    const xml = recipeEditorReducer(markdown, {
      type: "format_changed",
      renderFormat: "xml",
    });

    expect(markdown.definition.assembly_config.render_format).toBe("markdown");
    expect(xml.definition.assembly_config.render_format).toBe("xml");
    expect(xml.definition.blocks?.[0].section_key).toBeNull();
  });

  it("keeps declarations in the definition and runtime values in sibling state", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const withVariable = reduce(
      initial,
      { type: "variable_added", variable: audienceVariable },
      {
        type: "runtime_value_changed",
        variableName: "audience",
        value: "Maintainers",
      },
    );

    expect(withVariable.definition.variables?.at(-1)).toEqual(audienceVariable);
    expect(withVariable.runtimeValues).toEqual({ audience: "Maintainers" });
    expect("runtimeValues" in withVariable.definition).toBe(false);
    expect("runtime_values" in withVariable.definition).toBe(false);
    expect(withVariable.definition.variables?.at(-1)?.default_value).toBeNull();
  });

  it("removes obsolete runtime values when a declaration is removed", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const edited = reduce(
      initial,
      { type: "variable_added", variable: audienceVariable },
      {
        type: "runtime_value_changed",
        variableName: "audience",
        value: "Maintainers",
      },
      { type: "variable_removed", variableName: "audience" },
    );

    expect(
      edited.definition.variables?.some(({ name }) => name === "audience"),
    ).toBe(false);
    expect(edited.runtimeValues).toEqual({});
  });

  it("gates Apply on required values without gating a valid recipe save", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");

    expect(canApplyRecipe(initial)).toBe(false);
    expect(canSaveRecipeAsNew(initial)).toBe(true);

    const ready = recipeEditorReducer(initial, {
      type: "runtime_value_changed",
      variableName: "task",
      value: "Review this patch",
    });
    expect(canApplyRecipe(ready)).toBe(true);
    expect(canSaveRecipeAsNew(ready)).toBe(true);
  });

  it("tracks source-definition dirtiness but ignores ephemeral runtime values", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const runtimeOnly = recipeEditorReducer(initial, {
      type: "runtime_value_changed",
      variableName: "task",
      value: "Review this patch",
    });
    const renamed = recipeEditorReducer(runtimeOnly, {
      type: "block_updated",
      blockId: "objective",
      changes: { name: "Goal" },
    });
    const restored = recipeEditorReducer(renamed, {
      type: "block_updated",
      blockId: "objective",
      changes: { name: "Objective" },
    });

    expect(initial.isDirty).toBe(false);
    expect(runtimeOnly.isDirty).toBe(false);
    expect(renamed.isDirty).toBe(true);
    expect(restored.isDirty).toBe(false);
  });

  it("always offers Save as new for valid sources and Update only for saved sources", () => {
    const builtIn = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const saved = createRecipeWorkingCopy(
      {
        source_kind: "saved",
        id: "saved-1",
        name: "Saved recipe",
        definition: CLEAR_TASK_RECIPE.definition,
      },
      "system",
    );

    expect(canSaveRecipeAsNew(builtIn)).toBe(true);
    expect(canUpdateRecipe(builtIn)).toBe(false);
    expect(canSaveRecipeAsNew(saved)).toBe(true);
    expect(canUpdateRecipe(saved)).toBe(true);
  });

  it("rejects invalid definitions for Save as new and Update", () => {
    const saved = createRecipeWorkingCopy(
      {
        source_kind: "saved",
        id: "saved-1",
        name: "Saved recipe",
        definition: CLEAR_TASK_RECIPE.definition,
      },
      "system",
    );
    const invalid: RecipeEditorState = {
      ...saved,
      definition: {
        ...saved.definition,
        blocks: saved.definition.blocks?.map((block, index) =>
          index === 0 ? { ...block, role: "user" } : block,
        ),
      },
    };

    expect(canSaveRecipeAsNew(invalid)).toBe(false);
    expect(canUpdateRecipe(invalid)).toBe(false);
  });

  it("serializes only a validated definition and never strips hidden runtime data", () => {
    const working = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const serialized = serializeRecipeDefinitionForSave(working.definition);
    serialized.blocks![0].content = "serialized copy";

    expect(serialized).not.toBe(working.definition);
    expect(working.definition.blocks?.[0].content).toBe(
      "Complete this task:\n\n{{task}}",
    );

    const contaminated = {
      ...working.definition,
      runtimeValues: { task: "private" },
    } as unknown as SingleTextRecipeDefinition;
    expect(() => serializeRecipeDefinitionForSave(contaminated)).toThrow(
      "invalid_prompt_definition",
    );
  });

  it("returns a narrowed v2 definition from unknown stored JSON", () => {
    const unknownDefinition: unknown = structuredClone(
      CLEAR_TASK_RECIPE.definition,
    );
    const parsed = parseRecipeDefinition(unknownDefinition);

    expect(parsed.schema_version).toBe(2);
    expect(parsed.definition_kind).toBe("single_text_recipe");
  });
});
