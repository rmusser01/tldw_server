import { describe, expect, it } from "vitest";

import {
  AGENT_WORKFLOW_RECIPE,
  BLANK_RECIPE,
  BUILT_IN_RECIPES,
  CLEAR_TASK_RECIPE,
  RESEARCH_AND_ANALYSIS_RECIPE,
} from "../built-in-recipes";
import { createRecipeWorkingCopy } from "../recipe-editor-state";

const expectedBlocks = {
  clear_task: [
    [
      "objective",
      "Objective",
      "objective",
      "Complete this task:\n\n{{task}}",
      10,
    ],
    [
      "context_inputs",
      "Context / inputs",
      "context_inputs",
      "Use the context and inputs provided by the user. If essential information is missing, state what is needed before proceeding.",
      20,
    ],
    [
      "constraints",
      "Constraints",
      "constraints",
      "Follow every explicit constraint. Preserve supplied names, facts, code, and required formatting; do not invent requirements.",
      30,
    ],
    [
      "output",
      "Output",
      "output",
      "Return the requested result directly. Make it clear, complete, and concise.",
      40,
    ],
  ],
  research_and_analysis: [
    [
      "question",
      "Question",
      "question",
      "Investigate and answer this question:\n\n{{research_question}}",
      10,
    ],
    [
      "evidence_source_rules",
      "Evidence / source rules",
      "evidence_source_rules",
      "Prefer primary and authoritative sources. Cite evidence near the claim it supports, and distinguish sourced facts from inference.",
      20,
    ],
    [
      "analysis_method",
      "Analysis method",
      "analysis_method",
      "Compare the evidence, test plausible alternatives, and explain material tradeoffs before reaching a conclusion.",
      30,
    ],
    [
      "uncertainty",
      "Uncertainty",
      "uncertainty",
      "State important assumptions, evidence gaps, conflicts, and confidence limits.",
      40,
    ],
    [
      "deliverable",
      "Deliverable",
      "deliverable",
      "Lead with a concise answer, then present the supporting evidence, analysis, and unresolved issues.",
      50,
    ],
  ],
  agent_workflow: [
    [
      "objective",
      "Objective",
      "objective",
      "Achieve this objective:\n\n{{objective}}",
      10,
    ],
    [
      "allowed_actions_tools",
      "Allowed actions / tools",
      "allowed_actions_tools",
      "Use only the actions and tools that are available and authorized. Do not claim an action succeeded without evidence.",
      20,
    ],
    [
      "plan_execute_loop",
      "Plan / execute loop",
      "plan_execute_loop",
      "Make a concise plan, execute it step by step, verify each material result, and adapt when evidence changes.",
      30,
    ],
    [
      "stop_confirmation_conditions",
      "Stop / confirmation conditions",
      "stop_confirmation_conditions",
      "Stop and request confirmation before destructive, irreversible, costly, or externally visible actions, or when required authority is missing.",
      40,
    ],
    [
      "final_report",
      "Final report",
      "final_report",
      "Report the outcome first, then summarize material changes, verification, and any remaining blockers.",
      50,
    ],
  ],
  blank: [],
} as const;

describe("built-in single-text recipes", () => {
  it("publishes the four approved starters in their stable order", () => {
    expect(BUILT_IN_RECIPES.map(({ id, name }) => [id, name])).toEqual([
      ["clear_task", "Clear task"],
      ["research_and_analysis", "Research and analysis"],
      ["agent_workflow", "Agent workflow"],
      ["blank", "Blank"],
    ]);
  });

  it.each([
    CLEAR_TASK_RECIPE,
    RESEARCH_AND_ANALYSIS_RECIPE,
    AGENT_WORKFLOW_RECIPE,
    BLANK_RECIPE,
  ])("keeps exact ordered starter content for $name", (starter) => {
    expect(
      starter.definition.blocks?.map((block) => [
        block.id,
        block.name,
        block.section_key,
        block.content,
        block.order,
      ]),
    ).toEqual(expectedBlocks[starter.id]);
  });

  it("declares only the editable task inputs and leaves Blank empty", () => {
    expect(CLEAR_TASK_RECIPE.definition.variables).toEqual([
      {
        name: "task",
        label: "Task",
        description: "The task to complete.",
        required: true,
        default_value: null,
        input_type: "textarea",
        options: null,
        max_length: null,
      },
    ]);
    expect(RESEARCH_AND_ANALYSIS_RECIPE.definition.variables?.[0]).toEqual({
      name: "research_question",
      label: "Research question",
      description: "The question to investigate.",
      required: true,
      default_value: null,
      input_type: "textarea",
      options: null,
      max_length: null,
    });
    expect(AGENT_WORKFLOW_RECIPE.definition.variables?.[0]).toEqual({
      name: "objective",
      label: "Objective",
      description: "The outcome the agent should achieve.",
      required: true,
      default_value: null,
      input_type: "textarea",
      options: null,
      max_length: null,
    });
    expect(BLANK_RECIPE.definition.variables).toEqual([]);
    expect(BLANK_RECIPE.definition.blocks).toEqual([]);
  });

  it.each(["system", "user_message"] as const)(
    "creates valid %s working copies without changing starter content",
    (target) => {
      for (const starter of BUILT_IN_RECIPES) {
        const working = createRecipeWorkingCopy(starter, target);
        const role = target === "system" ? "system" : "user";

        expect(working.definition.assembly_config.target_role).toBe(role);
        expect(
          working.definition.blocks?.every((block) => block.role === role),
        ).toBe(true);
        expect(
          working.definition.blocks?.map((block) => block.content),
        ).toEqual(starter.definition.blocks?.map((block) => block.content));
        expect(working.isDirty).toBe(false);
      }
    },
  );

  it("deep-freezes source starters and returns isolated editable copies", () => {
    expect(Object.isFrozen(BUILT_IN_RECIPES)).toBe(true);
    expect(Object.isFrozen(CLEAR_TASK_RECIPE)).toBe(true);
    expect(Object.isFrozen(CLEAR_TASK_RECIPE.definition)).toBe(true);
    expect(Object.isFrozen(CLEAR_TASK_RECIPE.definition.blocks)).toBe(true);
    expect(Object.isFrozen(CLEAR_TASK_RECIPE.definition.blocks?.[0])).toBe(
      true,
    );

    const first = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "user_message");
    const second = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, "user_message");
    first.definition.blocks![0].content = "changed";

    expect(CLEAR_TASK_RECIPE.definition.blocks?.[0].content).toBe(
      "Complete this task:\n\n{{task}}",
    );
    expect(second.definition.blocks?.[0].content).toBe(
      "Complete this task:\n\n{{task}}",
    );
  });
});
