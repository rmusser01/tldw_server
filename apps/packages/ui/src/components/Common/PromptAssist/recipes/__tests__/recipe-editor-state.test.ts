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

  it("checks every final truncated section-key candidate for uniqueness", () => {
    const fullLengthKey = "a".repeat(128);
    const secondKey = `${"a".repeat(126)}_2`;
    const expectedThirdKey = `${"a".repeat(126)}_3`;

    expect(
      proposeRecipeSectionKey(
        fullLengthKey,
        new Set([fullLengthKey, secondKey]),
      ),
    ).toBe(expectedThirdKey);

    const sourceDefinition = {
      schema_version: 2,
      format: "structured",
      definition_kind: "single_text_recipe",
      assembly_config: {
        assembly_mode: "single_text",
        target_role: "system",
        render_format: "xml",
      },
      variables: [],
      blocks: [
        {
          id: "first",
          name: "First",
          role: "system",
          content: "First",
          order: 10,
          section_key: fullLengthKey,
        },
        {
          id: "second",
          name: "Second",
          role: "system",
          content: "Second",
          order: 20,
          section_key: secondKey,
        },
      ],
    };
    const initial = createRecipeWorkingCopy(
      {
        source_kind: "saved",
        id: "saved-long-keys",
        name: "Long keys",
        definition: sourceDefinition,
      },
      "system",
    );
    const added = recipeEditorReducer(initial, {
      type: "block_added",
      block: { id: "third", name: fullLengthKey, content: "Third" },
    });

    expect(added.definition.blocks?.at(-1)?.section_key).toBe(expectedThirdKey);
    expect(canSaveRecipeAsNew(added)).toBe(true);
  });

  it("renormalizes safely before adding after the maximum supported order", () => {
    const sourceDefinition = {
      schema_version: 2,
      format: "structured",
      definition_kind: "single_text_recipe",
      assembly_config: {
        assembly_mode: "single_text",
        target_role: "system",
        render_format: "markdown",
      },
      variables: [],
      blocks: [
        {
          id: "late",
          name: "Late",
          role: "system",
          content: "Late",
          order: Number.MAX_SAFE_INTEGER,
        },
        {
          id: "early",
          name: "Early",
          role: "system",
          content: "Early",
          order: -5,
        },
      ],
    };
    const initial = createRecipeWorkingCopy(
      {
        source_kind: "saved",
        id: "saved-boundary-order",
        name: "Boundary order",
        definition: sourceDefinition,
      },
      "system",
    );
    const added = recipeEditorReducer(initial, {
      type: "block_added",
      block: { id: "new", name: "New", content: "New" },
    });

    expect(
      added.definition.blocks?.map(({ id, order }) => [id, order]),
    ).toEqual([
      ["early", 10],
      ["late", 20],
      ["new", 30],
    ]);
    expect(sourceDefinition.blocks.map(({ id, order }) => [id, order])).toEqual(
      [
        ["late", Number.MAX_SAFE_INTEGER],
        ["early", -5],
      ],
    );
    expect(canSaveRecipeAsNew(added)).toBe(true);
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

  it("atomically updates every persisted variable field without changing its order", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const withAudience = recipeEditorReducer(initial, {
      type: "variable_added",
      variable: audienceVariable,
    });
    const updated = recipeEditorReducer(withAudience, {
      type: "variable_updated",
      variableName: "audience",
      changes: {
        name: "reader",
        label: "Reader",
        description: "The intended reader.",
        required: false,
        default_value: "Maintainers",
        input_type: "select",
        options: ["Maintainers", "Contributors"],
        max_length: 40,
      },
    });

    expect(updated.definition.variables?.map(({ name }) => name)).toEqual([
      "task",
      "reader",
    ]);
    expect(updated.definition.variables?.[1]).toEqual({
      name: "reader",
      label: "Reader",
      description: "The intended reader.",
      required: false,
      default_value: "Maintainers",
      input_type: "select",
      options: ["Maintainers", "Contributors"],
      max_length: 40,
    });
    expect(audienceVariable).toEqual({
      name: "audience",
      label: "Audience",
      description: null,
      required: true,
      default_value: null,
      input_type: "text",
      options: null,
      max_length: null,
    });
  });

  it("migrates a runtime value on rename but leaves template references authored", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const prepared = reduce(
      initial,
      { type: "variable_added", variable: audienceVariable },
      {
        type: "block_added",
        block: {
          id: "audience_block",
          name: "Audience",
          content: "Write for {{audience}}.",
          isTemplate: true,
        },
      },
      {
        type: "runtime_value_changed",
        variableName: "audience",
        value: "Maintainers",
      },
    );
    const renamed = recipeEditorReducer(prepared, {
      type: "variable_updated",
      variableName: "audience",
      changes: { name: "reader" },
    });

    expect(renamed.definition.variables?.at(-1)?.name).toBe("reader");
    expect(renamed.runtimeValues).toEqual({ reader: "Maintainers" });
    expect(renamed.definition.blocks?.at(-1)?.content).toBe(
      "Write for {{audience}}.",
    );
    expect(canSaveRecipeAsNew(renamed)).toBe(false);
  });

  it("rejects invalid, duplicate, or runtime-colliding variable renames atomically", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const withAudience = reduce(
      initial,
      { type: "variable_added", variable: audienceVariable },
      {
        type: "runtime_value_changed",
        variableName: "audience",
        value: "Maintainers",
      },
    );

    for (const name of ["", "not valid", "task", "x".repeat(129)]) {
      const result = recipeEditorReducer(withAudience, {
        type: "variable_updated",
        variableName: "audience",
        changes: { name },
      });
      expect(result).toBe(withAudience);
    }

    const runtimeCollision: RecipeEditorState = {
      ...withAudience,
      runtimeValues: {
        ...withAudience.runtimeValues,
        reader: "Existing value",
      },
    };
    const collided = recipeEditorReducer(runtimeCollision, {
      type: "variable_updated",
      variableName: "audience",
      changes: { name: "reader", label: "Reader" },
    });
    expect(collided).toBe(runtimeCollision);
    expect(collided.definition.variables?.at(-1)).toEqual(audienceVariable);
    expect(collided.runtimeValues).toEqual({
      audience: "Maintainers",
      reader: "Existing value",
    });
  });

  it("keeps authored defaults separate from runtime input during variable edits", () => {
    const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "system");
    const runtimeOnly = recipeEditorReducer(initial, {
      type: "runtime_value_changed",
      variableName: "task",
      value: "Runtime task",
    });
    const withDefault = recipeEditorReducer(runtimeOnly, {
      type: "variable_updated",
      variableName: "task",
      changes: { default_value: "Starter task" },
    });

    expect(withDefault.definition.variables?.[0].default_value).toBe(
      "Starter task",
    );
    expect(withDefault.runtimeValues).toEqual({ task: "Runtime task" });
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

  it("canonicalizes sparse saved definitions before semantic dirty tracking", () => {
    const sparseDefinition = {
      schema_version: 2,
      format: "structured",
      definition_kind: "single_text_recipe",
      assembly_config: {
        assembly_mode: "single_text",
        target_role: "system",
        render_format: "markdown",
      },
      variables: [{ name: "topic" }],
      blocks: [
        {
          id: "objective",
          name: "Objective",
          role: "system",
          content: "Explain the topic.",
          order: 10,
        },
      ],
    };
    const initial = createRecipeWorkingCopy(
      {
        source_kind: "saved",
        id: "saved-sparse",
        name: "Sparse recipe",
        definition: sparseDefinition,
      },
      "system",
    );

    expect(initial.definition.assembly_config.block_separator).toBe("\n\n");
    expect(initial.definition.variables).toEqual([
      {
        name: "topic",
        label: null,
        description: null,
        required: false,
        default_value: null,
        input_type: "text",
        options: null,
        max_length: null,
      },
    ]);
    expect(initial.definition.blocks).toEqual([
      {
        id: "objective",
        name: "Objective",
        role: "system",
        kind: null,
        content: "Explain the topic.",
        enabled: true,
        order: 10,
        is_template: false,
        section_key: null,
      },
    ]);
    expect(sparseDefinition).not.toHaveProperty(
      "assembly_config.block_separator",
    );
    expect(sparseDefinition.variables[0]).toEqual({ name: "topic" });
    expect(sparseDefinition.blocks[0]).not.toHaveProperty("enabled");
    expect(sparseDefinition.blocks[0]).not.toHaveProperty("is_template");

    const toggledTwice = reduce(
      initial,
      { type: "block_toggled", blockId: "objective" },
      { type: "block_toggled", blockId: "objective" },
    );
    const templateRestored = reduce(
      toggledTwice,
      {
        type: "block_updated",
        blockId: "objective",
        changes: { is_template: true },
      },
      {
        type: "block_updated",
        blockId: "objective",
        changes: { is_template: false },
      },
    );
    const variableRestored = reduce(
      templateRestored,
      {
        type: "variable_updated",
        variableName: "topic",
        changes: {
          label: "Topic",
          description: "Topic to explain.",
          required: true,
          default_value: "Testing",
          input_type: "textarea",
          options: ["Testing"],
          max_length: 100,
        },
      },
      {
        type: "variable_updated",
        variableName: "topic",
        changes: {
          label: null,
          description: null,
          required: false,
          default_value: null,
          input_type: "text",
          options: null,
          max_length: null,
        },
      },
    );

    expect(toggledTwice.isDirty).toBe(false);
    expect(templateRestored.isDirty).toBe(false);
    expect(variableRestored.isDirty).toBe(false);
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
