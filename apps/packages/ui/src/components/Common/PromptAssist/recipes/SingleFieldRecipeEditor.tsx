import { Button } from "@/components/Common/Button";
import { BlockEditorPanel } from "@/components/Option/Prompt/Structured/BlockEditorPanel";
import { BlockListPanel } from "@/components/Option/Prompt/Structured/BlockListPanel";
import { VariableEditorPanel } from "@/components/Option/Prompt/Structured/VariableEditorPanel";
import {
  renderSingleTextRecipe,
  SINGLE_TEXT_RECIPE_LIMITS,
  SingleTextRecipeRenderError,
} from "@/components/Option/Prompt/structured-prompt-utils";
import React, {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { BUILT_IN_RECIPES, CLEAR_TASK_RECIPE } from "./built-in-recipes";
import {
  createRecipeWorkingCopy,
  parseRecipeDefinition,
  recipeEditorReducer,
  serializeRecipeDefinitionForSave,
} from "./recipe-editor-state";
import type {
  RecipeBlockChanges,
  RecipeEditorAction,
  RecipeRenderFormat,
  RecipeSource,
  RecipeTarget,
  RecipeVariableChanges,
  SavedRecipeSource,
  SingleTextRecipeBlock,
  SingleTextRecipeDefinition,
  SingleTextRecipeVariable,
} from "./types";

export type SingleFieldRecipeEditorProps = {
  target: RecipeTarget;
  initialSource?: RecipeSource;
  savedRecipes?: readonly SavedRecipeSource[];
  persistenceAvailable?: boolean;
  persistenceUnavailableReason?: string;
  onApply: (compiledText: string) => void;
  onSaveAsNew?: (
    definition: SingleTextRecipeDefinition,
  ) => void | Promise<void>;
  onUpdate?: (
    savedSourceId: string,
    definition: SingleTextRecipeDefinition,
  ) => void | Promise<void>;
};

const XML_SECTION_KEY = /^[A-Za-z_][A-Za-z0-9_.-]*$/;

const orderBlocks = (blocks: readonly SingleTextRecipeBlock[]) =>
  blocks
    .map((block, index) => ({ block, index }))
    .sort(
      (left, right) =>
        left.block.order - right.block.order || left.index - right.index,
    )
    .map(({ block }) => block);

const sourceValue = (source: Pick<RecipeSource, "source_kind" | "id">) =>
  `${source.source_kind}:${source.id}`;

const sectionKeyError = (
  block: SingleTextRecipeBlock | null,
  blocks: readonly SingleTextRecipeBlock[],
  renderFormat: RecipeRenderFormat,
): string | null => {
  if (!block || block.enabled === false || renderFormat !== "xml") return null;
  if (
    typeof block.section_key !== "string" ||
    !XML_SECTION_KEY.test(block.section_key)
  ) {
    return "Use a valid section key starting with a letter or underscore.";
  }
  if (
    blocks.some(
      (candidate) =>
        candidate.id !== block.id &&
        candidate.enabled !== false &&
        candidate.section_key === block.section_key,
    )
  ) {
    return "Enabled XML blocks need unique section keys.";
  }
  return null;
};

const renderErrorMessage = (
  error: unknown,
  variables: readonly SingleTextRecipeVariable[],
): string => {
  if (!(error instanceof SingleTextRecipeRenderError)) {
    return "Fix the recipe definition before applying or saving.";
  }
  const variableLabel =
    variables.find((variable) => variable.name === error.variable_name)
      ?.label ?? error.variable_name;
  switch (error.code) {
    case "missing_required_variable":
      return `Enter a current value for ${variableLabel ?? "the required variable"}.`;
    case "closing_tag_collision":
      return "A compiled value contains the closing tag for its XML section.";
    case "unknown_variable_reference":
      return `Declare ${error.variable_name ?? "the referenced variable"} or update the block content.`;
    case "rendered_output_too_large":
      return "The compiled prompt is too large. Shorten its blocks or current values.";
    default:
      return "The recipe cannot be rendered with the current values.";
  }
};

export function SingleFieldRecipeEditor({
  target,
  initialSource = CLEAR_TASK_RECIPE,
  savedRecipes = [],
  persistenceAvailable,
  persistenceUnavailableReason = "Recipe saving requires a supported online server.",
  onApply,
  onSaveAsNew,
  onUpdate,
}: SingleFieldRecipeEditorProps) {
  const ownerKey = `${target}\u0000${sourceValue(initialSource)}`;
  const [state, setState] = useState(() =>
    createRecipeWorkingCopy(initialSource, target),
  );
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(
    state.definition.blocks?.[0]?.id ?? null,
  );
  const [variableNameDrafts, setVariableNameDrafts] = useState<
    Record<string, string>
  >({});
  const [variableNameErrors, setVariableNameErrors] = useState<
    Record<string, string>
  >({});
  const pendingFocus = useRef<"editor" | "add" | null>(null);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const addButtonRef = useRef<HTMLButtonElement>(null);
  const previousOwnerKey = useRef(ownerKey);
  const persistenceRequest = useRef(0);
  const persistencePending = useRef(false);
  const [persistenceAction, setPersistenceAction] = useState<
    "save" | "update" | null
  >(null);
  const [persistenceError, setPersistenceError] = useState<string | null>(null);

  const sources = useMemo(() => {
    const all: RecipeSource[] = [...BUILT_IN_RECIPES];
    for (const source of [initialSource, ...savedRecipes]) {
      if (
        !all.some((candidate) => sourceValue(candidate) === sourceValue(source))
      ) {
        all.push(source);
      }
    }
    return all;
  }, [initialSource, savedRecipes]);

  const orderedBlocks = useMemo(
    () => orderBlocks(state.definition.blocks ?? []),
    [state.definition.blocks],
  );
  const selectedBlock =
    orderedBlocks.find((block) => block.id === selectedBlockId) ?? null;
  const currentSectionKeyError = sectionKeyError(
    selectedBlock,
    orderedBlocks,
    state.definition.assembly_config.render_format,
  );

  const preview = useMemo(() => {
    try {
      const definition = parseRecipeDefinition(state.definition);
      return {
        definitionValid: true,
        renderedText: renderSingleTextRecipe(definition, state.runtimeValues)
          .rendered_text,
        error: null as string | null,
      };
    } catch (error) {
      const isRenderError = error instanceof SingleTextRecipeRenderError;
      return {
        definitionValid: isRenderError,
        renderedText: null,
        error: renderErrorMessage(error, state.definition.variables ?? []),
      };
    }
  }, [state.definition, state.runtimeValues]);

  useLayoutEffect(() => {
    if (previousOwnerKey.current === ownerKey) return;
    previousOwnerKey.current = ownerKey;
    persistenceRequest.current += 1;
    persistencePending.current = false;
    setPersistenceAction(null);
    setPersistenceError(null);
    const next = createRecipeWorkingCopy(initialSource, target);
    setState(next);
    setVariableNameDrafts({});
    setVariableNameErrors({});
    setSelectedBlockId(next.definition.blocks?.[0]?.id ?? null);
    pendingFocus.current = next.definition.blocks?.length ? "editor" : "add";
  }, [initialSource, ownerKey, target]);

  useEffect(
    () => () => {
      persistenceRequest.current += 1;
      persistencePending.current = false;
    },
    [],
  );

  useEffect(() => {
    if (!pendingFocus.current) return;
    if (pendingFocus.current === "editor") nameInputRef.current?.focus();
    else addButtonRef.current?.focus();
    pendingFocus.current = null;
  }, [selectedBlockId, state.definition.blocks]);

  const dispatch = (action: RecipeEditorAction) => {
    setState((current) => recipeEditorReducer(current, action));
  };

  const validateVariableNameDraft = (variableName: string, value: string) => {
    if (
      !/^[A-Za-z0-9_]+$/.test(value) ||
      value.length > SINGLE_TEXT_RECIPE_LIMITS.max_key_length
    ) {
      return "Use a valid variable name with letters, numbers, or underscores.";
    }
    if (
      (state.definition.variables ?? []).some(
        (variable) => variable.name !== variableName && variable.name === value,
      )
    ) {
      return "Variable names must be unique.";
    }
    if (
      value !== variableName &&
      Object.prototype.hasOwnProperty.call(state.runtimeValues, value)
    ) {
      return "Clear the existing current value before using this variable name.";
    }
    return null;
  };

  const selectSource = (value: string) => {
    const source = sources.find(
      (candidate) => sourceValue(candidate) === value,
    );
    if (!source) return;
    const next = createRecipeWorkingCopy(source, target);
    persistenceRequest.current += 1;
    persistencePending.current = false;
    setPersistenceAction(null);
    setPersistenceError(null);
    setState(next);
    setVariableNameDrafts({});
    setVariableNameErrors({});
    setSelectedBlockId(next.definition.blocks?.[0]?.id ?? null);
    pendingFocus.current = next.definition.blocks?.length ? "editor" : "add";
  };

  const addBlock = () => {
    if (
      (state.definition.blocks ?? []).length >=
      SINGLE_TEXT_RECIPE_LIMITS.max_blocks
    ) {
      return;
    }
    const usedIds = new Set(
      (state.definition.blocks ?? []).map(({ id }) => id),
    );
    let suffix = usedIds.size + 1;
    while (usedIds.has(`block_${suffix}`)) suffix += 1;
    const id = `block_${suffix}`;
    dispatch({
      type: "block_added",
      block: { id, name: "New block", content: "" },
    });
    setSelectedBlockId(id);
    pendingFocus.current = "editor";
  };

  const removeBlock = (blockId: string) => {
    dispatch({ type: "block_removed", blockId });
    if (selectedBlockId !== blockId) return;
    const remaining = orderedBlocks.filter((block) => block.id !== blockId);
    setSelectedBlockId(remaining[0]?.id ?? null);
    pendingFocus.current = remaining.length ? "editor" : "add";
  };

  const moveBlock = (blockId: string, direction: "up" | "down") => {
    const index = orderedBlocks.findIndex((block) => block.id === blockId);
    if (index < 0) return;
    dispatch({
      type: "block_reordered",
      blockId,
      toIndex: direction === "up" ? index - 1 : index + 1,
    });
  };

  const updateSelectedBlock = (
    changes: Partial<{
      name: string;
      role: "system" | "developer" | "user" | "assistant";
      content: string;
      enabled: boolean;
      is_template: boolean;
    }>,
  ) => {
    if (!selectedBlock) return;
    if (
      typeof changes.enabled === "boolean" &&
      changes.enabled !== (selectedBlock.enabled !== false)
    ) {
      dispatch({ type: "block_toggled", blockId: selectedBlock.id });
      return;
    }
    const recipeChanges: RecipeBlockChanges = {};
    if (changes.name !== undefined) recipeChanges.name = changes.name;
    if (changes.content !== undefined) recipeChanges.content = changes.content;
    if (changes.is_template !== undefined) {
      recipeChanges.is_template = changes.is_template;
    }
    dispatch({
      type: "block_updated",
      blockId: selectedBlock.id,
      changes: recipeChanges,
    });
  };

  const addVariable = () => {
    if (
      (state.definition.variables ?? []).length >=
      SINGLE_TEXT_RECIPE_LIMITS.max_variables
    ) {
      return;
    }
    const usedNames = new Set(
      (state.definition.variables ?? []).map(({ name }) => name),
    );
    let suffix = usedNames.size + 1;
    while (usedNames.has(`variable_${suffix}`)) suffix += 1;
    dispatch({
      type: "variable_added",
      variable: {
        name: `variable_${suffix}`,
        label: `Variable ${suffix}`,
        description: null,
        required: false,
        default_value: null,
        input_type: "text",
        options: null,
        max_length: null,
      },
    });
  };

  const changeVariableNameDraft = (variableName: string, value: string) => {
    setVariableNameDrafts((current) => ({ ...current, [variableName]: value }));
    const error = validateVariableNameDraft(variableName, value);
    setVariableNameErrors((current) => {
      const next = { ...current };
      if (error) next[variableName] = error;
      else delete next[variableName];
      return next;
    });
  };

  const removeVariable = (variableName: string) => {
    dispatch({ type: "variable_removed", variableName });
    setVariableNameDrafts((current) => {
      const next = { ...current };
      delete next[variableName];
      return next;
    });
    setVariableNameErrors((current) => {
      const next = { ...current };
      delete next[variableName];
      return next;
    });
  };

  const commitVariableName = (variableName: string) => {
    const value = variableNameDrafts[variableName];
    if (value === undefined || value === variableName) return;
    const error = validateVariableNameDraft(variableName, value);
    if (error) {
      setVariableNameErrors((current) => ({
        ...current,
        [variableName]: error,
      }));
      return;
    }
    dispatch({
      type: "variable_updated",
      variableName,
      changes: { name: value },
    });
    setVariableNameDrafts((current) => {
      const next = { ...current };
      delete next[variableName];
      return next;
    });
    setVariableNameErrors((current) => {
      const next = { ...current };
      delete next[variableName];
      return next;
    });
  };

  const saveDefinition = () =>
    serializeRecipeDefinitionForSave(state.definition);

  const runPersistence = async (
    action: "save" | "update",
    persist: () => void | Promise<void>,
  ) => {
    if (persistencePending.current) return;
    persistencePending.current = true;
    const request = ++persistenceRequest.current;
    setPersistenceAction(action);
    setPersistenceError(null);
    try {
      await persist();
    } catch {
      if (request === persistenceRequest.current) {
        setPersistenceError(
          action === "save"
            ? "Could not save the recipe. Try again."
            : "Could not update the recipe. Try again.",
        );
      }
    } finally {
      if (request === persistenceRequest.current) {
        persistencePending.current = false;
        setPersistenceAction(null);
      }
    }
  };

  const targetLabel =
    target === "system" ? "System prompt" : "User message draft";
  const applyLabel =
    target === "system" ? "Apply to system prompt" : "Apply to user message";
  const variableNamesReady =
    Object.keys(variableNameErrors).length === 0 &&
    !Object.entries(variableNameDrafts).some(([name, value]) => name !== value);
  const saveEnabled =
    persistenceAvailable === true &&
    preview.definitionValid &&
    variableNamesReady &&
    Boolean(onSaveAsNew);
  const refreshedSavedSource =
    state.source.source_kind === "saved"
      ? savedRecipes.find(
          (source) => sourceValue(source) === sourceValue(state.source),
        )
      : undefined;
  const liveSelectedSyncStatus = refreshedSavedSource
    ? refreshedSavedSource.syncStatus
    : initialSource.source_kind === "saved" &&
        sourceValue(initialSource) === sourceValue(state.source)
      ? initialSource.syncStatus
      : state.source.syncStatus;
  const sourceHasConflict =
    state.source.source_kind === "saved" &&
    liveSelectedSyncStatus === "conflict";
  const updateEnabled =
    persistenceAvailable === true &&
    preview.definitionValid &&
    variableNamesReady &&
    state.source.source_kind === "saved" &&
    !sourceHasConflict &&
    Boolean(onUpdate);

  return (
    <section
      role="region"
      aria-label="Structured recipe builder"
      data-testid="single-field-recipe-editor"
      className="min-w-0 max-w-full space-y-4"
    >
      <div className="flex min-w-0 flex-col gap-3 rounded-xl border border-border bg-surface1 p-4 sm:flex-row sm:items-end sm:justify-between">
        <label className="min-w-0 flex-1">
          <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
            Recipe source
          </span>
          <select
            aria-label="Recipe source"
            value={sourceValue(state.source)}
            onChange={(event) => selectSource(event.target.value)}
            className="min-h-11 w-full min-w-0 rounded-md border border-border bg-background px-3 py-2 text-sm text-text"
          >
            <optgroup label="Starters">
              {sources
                .filter((source) => source.source_kind === "built_in")
                .map((source) => (
                  <option key={sourceValue(source)} value={sourceValue(source)}>
                    {source.name}
                  </option>
                ))}
            </optgroup>
            {sources.some((source) => source.source_kind === "saved") ? (
              <optgroup label="Saved recipes">
                {sources
                  .filter((source) => source.source_kind === "saved")
                  .map((source) => (
                    <option
                      key={sourceValue(source)}
                      value={sourceValue(source)}
                    >
                      {source.name}
                    </option>
                  ))}
              </optgroup>
            ) : null}
          </select>
        </label>

        <div className="min-w-0 sm:max-w-xs">
          <div className="text-xs font-medium uppercase tracking-wide text-text-muted">
            Target
          </div>
          <div className="mt-1 text-sm font-semibold text-text">
            {targetLabel}
          </div>
          <p className="text-xs text-text-muted">
            Every block compiles into this field only.
          </p>
        </div>
      </div>

      <label className="block max-w-sm">
        <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-text-muted">
          Output format
        </span>
        <select
          aria-label="Output format"
          value={state.definition.assembly_config.render_format}
          onChange={(event) =>
            dispatch({
              type: "format_changed",
              renderFormat: event.target.value as RecipeRenderFormat,
            })
          }
          className="min-h-11 w-full rounded-md border border-border bg-background px-3 py-2 text-sm text-text"
        >
          <option value="xml">XML-style sections</option>
          <option value="markdown">Markdown sections</option>
          <option value="freeform">Free-form text</option>
        </select>
      </label>

      <div
        data-testid="single-field-recipe-workspace"
        className="grid min-w-0 gap-4 xl:grid-cols-[18rem_minmax(0,1fr)]"
      >
        <BlockListPanel
          blocks={orderedBlocks.map((block) => ({
            ...block,
            enabled: block.enabled !== false,
            is_template: block.is_template === true,
          }))}
          selectedBlockId={selectedBlockId}
          onSelect={setSelectedBlockId}
          onAddBlock={addBlock}
          onMoveBlock={moveBlock}
          onReorderBlock={(blockId, toIndex) =>
            dispatch({ type: "block_reordered", blockId, toIndex })
          }
          onRemoveBlock={removeBlock}
          showRole={false}
          description="Drag blocks to order them, or use the named move controls."
          addButtonRef={addButtonRef}
        />
        <BlockEditorPanel
          block={
            selectedBlock
              ? {
                  ...selectedBlock,
                  enabled: selectedBlock.enabled !== false,
                  is_template: selectedBlock.is_template === true,
                }
              : null
          }
          onChange={updateSelectedBlock}
          allowedRoles={[state.definition.assembly_config.target_role]}
          showRole={false}
          sectionKey={selectedBlock?.section_key}
          onSectionKeyChange={(value) => {
            if (!selectedBlock) return;
            dispatch({
              type: "block_updated",
              blockId: selectedBlock.id,
              changes: { section_key: value },
            });
          }}
          sectionKeyError={currentSectionKeyError}
          nameInputRef={nameInputRef}
        />
      </div>

      <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <VariableEditorPanel
          variables={
            (state.definition.variables ?? []) as SingleTextRecipeVariable[]
          }
          previewValues={{}}
          runtimeValues={state.runtimeValues}
          onVariablesChange={() => undefined}
          onPreviewValuesChange={() => undefined}
          onAddVariable={addVariable}
          onRemoveVariable={removeVariable}
          onVariableChange={(variableName, changes) =>
            dispatch({
              type: "variable_updated",
              variableName,
              changes: changes as RecipeVariableChanges,
            })
          }
          onRuntimeValueChange={(variableName, value) =>
            dispatch({ type: "runtime_value_changed", variableName, value })
          }
          variableNameDrafts={variableNameDrafts}
          variableNameErrors={variableNameErrors}
          onVariableNameDraftChange={changeVariableNameDraft}
          onVariableNameCommit={commitVariableName}
          showDeclarationFields
        />

        <section className="min-w-0 rounded-xl border border-border bg-surface1 p-4">
          <div className="mb-3">
            <h3 className="text-sm font-semibold text-text">
              Compiled preview
            </h3>
            <p className="text-xs text-text-muted">
              Exact plain text that will replace the current{" "}
              {targetLabel.toLowerCase()}.
            </p>
          </div>
          <textarea
            readOnly
            aria-label="Compiled prompt preview"
            value={preview.renderedText ?? ""}
            rows={12}
            className="w-full min-w-0 resize-y rounded-md border border-border bg-background p-3 font-mono text-sm leading-6 text-text"
          />
          {preview.error && !currentSectionKeyError ? (
            <p role="alert" className="mt-2 text-sm text-danger">
              {preview.error}
            </p>
          ) : null}
        </section>
      </div>

      {persistenceAvailable !== true ? (
        <p role="status" className="text-sm text-warn">
          {persistenceUnavailableReason}
        </p>
      ) : null}

      {persistenceError ? (
        <p role="alert" className="text-sm text-danger">
          {persistenceError}
        </p>
      ) : null}

      <div className="flex flex-wrap justify-end gap-2">
        <Button
          variant="outline"
          size="lg"
          disabled={!saveEnabled || persistenceAction !== null}
          onClick={() => {
            if (!onSaveAsNew) return;
            void runPersistence("save", () => onSaveAsNew(saveDefinition()));
          }}
        >
          Save as new recipe
        </Button>
        {state.source.source_kind === "saved" && !sourceHasConflict ? (
          <Button
            variant="outline"
            size="lg"
            disabled={!updateEnabled || persistenceAction !== null}
            onClick={() => {
              if (!onUpdate || state.source.source_kind !== "saved") return;
              const savedSourceId = state.source.id;
              void runPersistence("update", () =>
                onUpdate(savedSourceId, saveDefinition()),
              );
            }}
          >
            Update recipe
          </Button>
        ) : null}
        <Button
          variant="primary"
          size="lg"
          disabled={preview.renderedText === null || !variableNamesReady}
          onClick={() => {
            if (preview.renderedText !== null) onApply(preview.renderedText);
          }}
        >
          {applyLabel}
        </Button>
      </div>
    </section>
  );
}
