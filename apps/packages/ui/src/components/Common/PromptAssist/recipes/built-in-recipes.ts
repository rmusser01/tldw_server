import type {
  BuiltInRecipe,
  SingleTextRecipeBlock,
  SingleTextRecipeVariable,
} from "./types";

const taskVariable: SingleTextRecipeVariable = {
  name: "task",
  label: "Task",
  description: "The task to complete.",
  required: true,
  default_value: null,
  input_type: "textarea",
  options: null,
  max_length: null,
};

const researchQuestionVariable: SingleTextRecipeVariable = {
  name: "research_question",
  label: "Research question",
  description: "The question to investigate.",
  required: true,
  default_value: null,
  input_type: "textarea",
  options: null,
  max_length: null,
};

const objectiveVariable: SingleTextRecipeVariable = {
  name: "objective",
  label: "Objective",
  description: "The outcome the agent should achieve.",
  required: true,
  default_value: null,
  input_type: "textarea",
  options: null,
  max_length: null,
};

const block = (
  id: string,
  name: string,
  content: string,
  order: number,
  isTemplate = false,
): SingleTextRecipeBlock => ({
  id,
  name,
  section_key: id,
  role: "system",
  kind: id,
  content,
  enabled: true,
  order,
  is_template: isTemplate,
});

const freezeBuiltIn = (recipe: BuiltInRecipe): BuiltInRecipe => {
  for (const variable of recipe.definition.variables ?? []) {
    if (variable.options) Object.freeze(variable.options);
    Object.freeze(variable);
  }
  for (const recipeBlock of recipe.definition.blocks ?? []) {
    Object.freeze(recipeBlock);
  }
  Object.freeze(recipe.definition.variables);
  Object.freeze(recipe.definition.blocks);
  Object.freeze(recipe.definition.assembly_config);
  Object.freeze(recipe.definition);
  return Object.freeze(recipe);
};

export const CLEAR_TASK_RECIPE = freezeBuiltIn({
  source_kind: "built_in",
  id: "clear_task",
  name: "Clear task",
  definition: {
    schema_version: 2,
    format: "structured",
    definition_kind: "single_text_recipe",
    assembly_config: {
      assembly_mode: "single_text",
      target_role: "system",
      render_format: "xml",
      block_separator: "\n\n",
    },
    variables: [taskVariable],
    blocks: [
      block(
        "objective",
        "Objective",
        "Complete this task:\n\n{{task}}",
        10,
        true,
      ),
      block(
        "context_inputs",
        "Context / inputs",
        "Use the context and inputs provided by the user. If essential information is missing, state what is needed before proceeding.",
        20,
      ),
      block(
        "constraints",
        "Constraints",
        "Follow every explicit constraint. Preserve supplied names, facts, code, and required formatting; do not invent requirements.",
        30,
      ),
      block(
        "output",
        "Output",
        "Return the requested result directly. Make it clear, complete, and concise.",
        40,
      ),
    ],
  },
});

export const RESEARCH_AND_ANALYSIS_RECIPE = freezeBuiltIn({
  source_kind: "built_in",
  id: "research_and_analysis",
  name: "Research and analysis",
  definition: {
    schema_version: 2,
    format: "structured",
    definition_kind: "single_text_recipe",
    assembly_config: {
      assembly_mode: "single_text",
      target_role: "system",
      render_format: "xml",
      block_separator: "\n\n",
    },
    variables: [researchQuestionVariable],
    blocks: [
      block(
        "question",
        "Question",
        "Investigate and answer this question:\n\n{{research_question}}",
        10,
        true,
      ),
      block(
        "evidence_source_rules",
        "Evidence / source rules",
        "Prefer primary and authoritative sources. Cite evidence near the claim it supports, and distinguish sourced facts from inference.",
        20,
      ),
      block(
        "analysis_method",
        "Analysis method",
        "Compare the evidence, test plausible alternatives, and explain material tradeoffs before reaching a conclusion.",
        30,
      ),
      block(
        "uncertainty",
        "Uncertainty",
        "State important assumptions, evidence gaps, conflicts, and confidence limits.",
        40,
      ),
      block(
        "deliverable",
        "Deliverable",
        "Lead with a concise answer, then present the supporting evidence, analysis, and unresolved issues.",
        50,
      ),
    ],
  },
});

export const AGENT_WORKFLOW_RECIPE = freezeBuiltIn({
  source_kind: "built_in",
  id: "agent_workflow",
  name: "Agent workflow",
  definition: {
    schema_version: 2,
    format: "structured",
    definition_kind: "single_text_recipe",
    assembly_config: {
      assembly_mode: "single_text",
      target_role: "system",
      render_format: "xml",
      block_separator: "\n\n",
    },
    variables: [objectiveVariable],
    blocks: [
      block(
        "objective",
        "Objective",
        "Achieve this objective:\n\n{{objective}}",
        10,
        true,
      ),
      block(
        "allowed_actions_tools",
        "Allowed actions / tools",
        "Use only the actions and tools that are available and authorized. Do not claim an action succeeded without evidence.",
        20,
      ),
      block(
        "plan_execute_loop",
        "Plan / execute loop",
        "Make a concise plan, execute it step by step, verify each material result, and adapt when evidence changes.",
        30,
      ),
      block(
        "stop_confirmation_conditions",
        "Stop / confirmation conditions",
        "Stop and request confirmation before destructive, irreversible, costly, or externally visible actions, or when required authority is missing.",
        40,
      ),
      block(
        "final_report",
        "Final report",
        "Report the outcome first, then summarize material changes, verification, and any remaining blockers.",
        50,
      ),
    ],
  },
});

export const BLANK_RECIPE = freezeBuiltIn({
  source_kind: "built_in",
  id: "blank",
  name: "Blank",
  definition: {
    schema_version: 2,
    format: "structured",
    definition_kind: "single_text_recipe",
    assembly_config: {
      assembly_mode: "single_text",
      target_role: "system",
      render_format: "xml",
      block_separator: "\n\n",
    },
    variables: [],
    blocks: [],
  },
});

export const BUILT_IN_RECIPES: readonly BuiltInRecipe[] = Object.freeze([
  CLEAR_TASK_RECIPE,
  RESEARCH_AND_ANALYSIS_RECIPE,
  AGENT_WORKFLOW_RECIPE,
  BLANK_RECIPE,
]);
