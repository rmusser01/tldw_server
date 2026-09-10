import type { SingleTextRecipeDefinitionV2 } from "@/components/Option/Prompt/structured-prompt-utils";

export type RecipeTarget = "system" | "user_message";
export type RecipeRole = "system" | "user";
export type RecipeRenderFormat = "xml" | "markdown" | "freeform";

export type SingleTextRecipeDefinition = SingleTextRecipeDefinitionV2;
export type SingleTextRecipeBlock = NonNullable<
  SingleTextRecipeDefinition["blocks"]
>[number];
export type SingleTextRecipeVariable = NonNullable<
  SingleTextRecipeDefinition["variables"]
>[number];

export type BuiltInRecipeId =
  | "clear_task"
  | "research_and_analysis"
  | "agent_workflow"
  | "blank";

export type BuiltInRecipe = {
  source_kind: "built_in";
  id: BuiltInRecipeId;
  name: string;
  definition: SingleTextRecipeDefinition;
};

export type SavedRecipeSource = {
  source_kind: "saved";
  id: string;
  name: string;
  definition: unknown;
  syncStatus?: "local" | "synced" | "pending" | "conflict" | "error";
};

export type RecipeSource = BuiltInRecipe | SavedRecipeSource;

export type RecipeSourceMetadata = Pick<
  RecipeSource,
  "source_kind" | "id" | "name"
> & {
  syncStatus?: SavedRecipeSource["syncStatus"];
};

export type RecipeRuntimeValues = Readonly<Record<string, string>>;

export type RecipeEditorState = {
  source: RecipeSourceMetadata;
  target: RecipeTarget;
  definition: SingleTextRecipeDefinition;
  runtimeValues: RecipeRuntimeValues;
  sourceSnapshot: string;
  isDirty: boolean;
};

export type NewRecipeBlock = {
  id: string;
  name: string;
  sectionKey?: string | null;
  kind?: string | null;
  content?: string;
  isTemplate?: boolean;
};

export type RecipeBlockChanges = Partial<
  Pick<
    SingleTextRecipeBlock,
    "name" | "section_key" | "kind" | "content" | "is_template"
  >
>;

export type RecipeVariableChanges = Partial<
  Pick<
    SingleTextRecipeVariable,
    | "name"
    | "label"
    | "description"
    | "required"
    | "default_value"
    | "input_type"
    | "options"
    | "max_length"
  >
>;

export type RecipeEditorAction =
  | { type: "block_added"; block: NewRecipeBlock }
  | { type: "block_removed"; blockId: string }
  | { type: "block_updated"; blockId: string; changes: RecipeBlockChanges }
  | { type: "block_reordered"; blockId: string; toIndex: number }
  | { type: "block_toggled"; blockId: string }
  | { type: "format_changed"; renderFormat: RecipeRenderFormat }
  | { type: "variable_added"; variable: SingleTextRecipeVariable }
  | {
      type: "variable_updated";
      variableName: string;
      changes: RecipeVariableChanges;
    }
  | { type: "variable_removed"; variableName: string }
  | {
      type: "runtime_value_changed";
      variableName: string;
      value: string;
    }
  | { type: "runtime_values_cleared" };
