import {
  renderSingleTextRecipe,
  SINGLE_TEXT_RECIPE_LIMITS,
  stableSerializePromptSnapshot,
} from "@/components/Option/Prompt/structured-prompt-utils";
import { parseStructuredPromptDefinitionForTransport } from "@/services/structured-prompt-transport";

import type {
  RecipeEditorAction,
  RecipeEditorState,
  RecipeRole,
  RecipeSource,
  RecipeTarget,
  SingleTextRecipeDefinition,
  SingleTextRecipeVariable,
} from "./types";

export const recipeTargetRole = (target: RecipeTarget): RecipeRole =>
  target === "system" ? "system" : "user";

const canonicalizeRecipeDefinition = (
  definition: SingleTextRecipeDefinition,
): SingleTextRecipeDefinition => ({
  schema_version: 2,
  format: "structured",
  definition_kind: "single_text_recipe",
  assembly_config: {
    assembly_mode: "single_text",
    target_role: definition.assembly_config.target_role,
    render_format: definition.assembly_config.render_format,
    block_separator: definition.assembly_config.block_separator ?? "\n\n",
  },
  variables: (definition.variables ?? []).map((variable) => ({
    name: variable.name,
    label: variable.label ?? null,
    description: variable.description ?? null,
    required: variable.required ?? false,
    default_value: structuredClone(
      variable.default_value === undefined ? null : variable.default_value,
    ),
    input_type: variable.input_type ?? "text",
    options:
      variable.options == null ? null : structuredClone(variable.options),
    max_length: variable.max_length ?? null,
  })),
  blocks: (definition.blocks ?? []).map((recipeBlock) => ({
    id: recipeBlock.id,
    name: recipeBlock.name,
    role: recipeBlock.role,
    kind: recipeBlock.kind ?? null,
    content: recipeBlock.content,
    enabled: recipeBlock.enabled ?? true,
    order: recipeBlock.order,
    is_template: recipeBlock.is_template ?? false,
    section_key: recipeBlock.section_key ?? null,
  })),
});

export const parseRecipeDefinition = (
  value: unknown,
): SingleTextRecipeDefinition => {
  const parsed = parseStructuredPromptDefinitionForTransport(
    value,
    "structured",
    2,
  );
  if (parsed?.schema_version !== 2)
    throw new Error("invalid_prompt_definition");
  return canonicalizeRecipeDefinition(parsed);
};

export const proposeRecipeSectionKey = (
  label: string,
  usedKeys: ReadonlySet<string> = new Set(),
): string => {
  const normalized = label
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^[-._]+|[-._]+$/g, "");
  const validStart = /^[a-z_]/.test(normalized) ? normalized : `_${normalized}`;
  const base = (validStart === "_" ? "section" : validStart).slice(
    0,
    SINGLE_TEXT_RECIPE_LIMITS.max_key_length,
  );
  if (!usedKeys.has(base)) return base;

  for (let suffix = 2; ; suffix += 1) {
    const suffixText = `_${suffix}`;
    const candidate = `${base.slice(
      0,
      SINGLE_TEXT_RECIPE_LIMITS.max_key_length - suffixText.length,
    )}${suffixText}`;
    if (!usedKeys.has(candidate)) return candidate;
  }
};

const orderRecipeBlocks = <Block extends { order: number }>(
  blocks: readonly Block[],
): Block[] =>
  blocks
    .map((recipeBlock, index) => ({ recipeBlock, index }))
    .sort(
      (left, right) =>
        left.recipeBlock.order - right.recipeBlock.order ||
        left.index - right.index,
    )
    .map(({ recipeBlock }) => recipeBlock);

export const createRecipeWorkingCopy = (
  source: RecipeSource,
  target: RecipeTarget,
): RecipeEditorState => {
  const parsed = parseRecipeDefinition(source.definition);
  const role = recipeTargetRole(target);
  const definition: SingleTextRecipeDefinition = {
    ...structuredClone(parsed),
    assembly_config: {
      ...structuredClone(parsed.assembly_config),
      target_role: role,
    },
    variables: structuredClone(parsed.variables ?? []),
    blocks: (parsed.blocks ?? []).map((recipeBlock) => ({
      ...structuredClone(recipeBlock),
      role,
    })),
  };
  const sourceSnapshot = stableSerializePromptSnapshot(definition);

  return {
    source: {
      source_kind: source.source_kind,
      id: source.id,
      name: source.name,
      ...(source.source_kind === "saved" && source.syncStatus
        ? { syncStatus: source.syncStatus }
        : {}),
    },
    target,
    definition,
    runtimeValues: {},
    sourceSnapshot,
    isDirty: false,
  };
};

const withDefinition = (
  state: RecipeEditorState,
  definition: SingleTextRecipeDefinition,
): RecipeEditorState => ({
  ...state,
  definition,
  isDirty: stableSerializePromptSnapshot(definition) !== state.sourceSnapshot,
});

export const recipeEditorReducer = (
  state: RecipeEditorState,
  action: RecipeEditorAction,
): RecipeEditorState => {
  switch (action.type) {
    case "block_added": {
      const blocks = state.definition.blocks ?? [];
      if (
        blocks.length >= SINGLE_TEXT_RECIPE_LIMITS.max_blocks ||
        blocks.some((recipeBlock) => recipeBlock.id === action.block.id)
      ) {
        return state;
      }
      const usedKeys = new Set(
        blocks.flatMap((recipeBlock) =>
          recipeBlock.section_key ? [recipeBlock.section_key] : [],
        ),
      );
      const highestOrder = blocks.reduce(
        (highest, recipeBlock) => Math.max(highest, recipeBlock.order),
        0,
      );
      const requiresSafeRenormalization =
        highestOrder > SINGLE_TEXT_RECIPE_LIMITS.max_abs_order - 10;
      const existingBlocks = requiresSafeRenormalization
        ? orderRecipeBlocks(blocks).map((recipeBlock, index) => ({
            ...recipeBlock,
            order: (index + 1) * 10,
          }))
        : blocks;
      const nextOrder = requiresSafeRenormalization
        ? (existingBlocks.length + 1) * 10
        : highestOrder + 10;
      return withDefinition(state, {
        ...state.definition,
        blocks: [
          ...existingBlocks,
          {
            id: action.block.id,
            name: action.block.name,
            section_key:
              action.block.sectionKey === undefined
                ? proposeRecipeSectionKey(action.block.name, usedKeys)
                : action.block.sectionKey,
            role: state.definition.assembly_config.target_role,
            kind: action.block.kind ?? null,
            content: action.block.content ?? "",
            enabled: true,
            order: nextOrder,
            is_template: action.block.isTemplate === true,
          },
        ],
      });
    }
    case "block_removed": {
      const blocks = state.definition.blocks ?? [];
      if (!blocks.some((recipeBlock) => recipeBlock.id === action.blockId)) {
        return state;
      }
      const remaining = blocks.filter(
        (recipeBlock) => recipeBlock.id !== action.blockId,
      );
      return withDefinition(state, {
        ...state.definition,
        blocks: remaining,
      });
    }
    case "block_updated": {
      let found = false;
      const blocks = (state.definition.blocks ?? []).map((recipeBlock) => {
        if (recipeBlock.id !== action.blockId) return recipeBlock;
        found = true;
        return { ...recipeBlock, ...action.changes };
      });
      return found
        ? withDefinition(state, { ...state.definition, blocks })
        : state;
    }
    case "block_reordered": {
      const ordered = orderRecipeBlocks(state.definition.blocks ?? []);
      const fromIndex = ordered.findIndex(
        (recipeBlock) => recipeBlock.id === action.blockId,
      );
      if (fromIndex < 0) return state;
      const [moved] = ordered.splice(fromIndex, 1);
      const toIndex = Math.max(0, Math.min(action.toIndex, ordered.length));
      ordered.splice(toIndex, 0, moved);
      return withDefinition(state, {
        ...state.definition,
        blocks: ordered.map((recipeBlock, index) => ({
          ...recipeBlock,
          order: (index + 1) * 10,
        })),
      });
    }
    case "block_toggled": {
      let found = false;
      const blocks = (state.definition.blocks ?? []).map((recipeBlock) => {
        if (recipeBlock.id !== action.blockId) return recipeBlock;
        found = true;
        return { ...recipeBlock, enabled: recipeBlock.enabled === false };
      });
      return found
        ? withDefinition(state, { ...state.definition, blocks })
        : state;
    }
    case "format_changed":
      return withDefinition(state, {
        ...state.definition,
        assembly_config: {
          ...state.definition.assembly_config,
          render_format: action.renderFormat,
        },
      });
    case "variable_added": {
      const variables = state.definition.variables ?? [];
      if (
        variables.some((variable) => variable.name === action.variable.name)
      ) {
        return state;
      }
      return withDefinition(state, {
        ...state.definition,
        variables: [...variables, structuredClone(action.variable)],
      });
    }
    case "variable_updated": {
      const variables = state.definition.variables ?? [];
      const variableIndex = variables.findIndex(
        (variable) => variable.name === action.variableName,
      );
      if (variableIndex < 0) return state;

      const nextVariable = {
        ...variables[variableIndex],
        ...structuredClone(action.changes),
      };
      const renamed = nextVariable.name !== action.variableName;
      if (
        renamed &&
        Object.prototype.hasOwnProperty.call(
          state.runtimeValues,
          nextVariable.name,
        )
      ) {
        return state;
      }

      const candidateVariables = variables.map((variable, index) =>
        index === variableIndex ? nextVariable : variable,
      );
      let validatedVariables: SingleTextRecipeVariable[];
      try {
        validatedVariables =
          parseRecipeDefinition({
            ...state.definition,
            variables: candidateVariables,
            blocks: [],
          }).variables ?? [];
      } catch {
        return state;
      }

      const runtimeValues = { ...state.runtimeValues };
      if (renamed) {
        if (
          Object.prototype.hasOwnProperty.call(
            runtimeValues,
            action.variableName,
          )
        ) {
          runtimeValues[nextVariable.name] = runtimeValues[action.variableName];
        }
        delete runtimeValues[action.variableName];
      }
      return {
        ...withDefinition(state, {
          ...state.definition,
          variables: validatedVariables,
        }),
        runtimeValues,
      };
    }
    case "variable_removed": {
      const variables = state.definition.variables ?? [];
      if (
        !variables.some((variable) => variable.name === action.variableName)
      ) {
        return state;
      }
      const runtimeValues = { ...state.runtimeValues };
      delete runtimeValues[action.variableName];
      return {
        ...withDefinition(state, {
          ...state.definition,
          variables: variables.filter(
            (variable) => variable.name !== action.variableName,
          ),
        }),
        runtimeValues,
      };
    }
    case "runtime_value_changed":
      if (
        !(state.definition.variables ?? []).some(
          (variable) => variable.name === action.variableName,
        )
      ) {
        return state;
      }
      return {
        ...state,
        runtimeValues: {
          ...state.runtimeValues,
          [action.variableName]: action.value,
        },
      };
    case "runtime_values_cleared":
      return Object.keys(state.runtimeValues).length
        ? { ...state, runtimeValues: {} }
        : state;
    default:
      return state;
  }
};

const isValidRecipeDefinition = (
  definition: SingleTextRecipeDefinition,
): boolean => {
  try {
    parseRecipeDefinition(definition);
    return true;
  } catch {
    return false;
  }
};

export const canApplyRecipe = (state: RecipeEditorState): boolean => {
  try {
    const definition = parseRecipeDefinition(state.definition);
    renderSingleTextRecipe(definition, state.runtimeValues);
    return true;
  } catch {
    return false;
  }
};

export const canSaveRecipeAsNew = (state: RecipeEditorState): boolean =>
  isValidRecipeDefinition(state.definition);

export const canUpdateRecipe = (state: RecipeEditorState): boolean =>
  state.source.source_kind === "saved" &&
  state.source.syncStatus !== "conflict" &&
  isValidRecipeDefinition(state.definition);

export const serializeRecipeDefinitionForSave = (
  definition: SingleTextRecipeDefinition,
): SingleTextRecipeDefinition =>
  structuredClone(parseRecipeDefinition(definition));
