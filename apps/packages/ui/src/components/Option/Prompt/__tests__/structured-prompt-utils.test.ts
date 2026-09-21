import { describe, expect, it, vi } from "vitest"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import * as utils from "../structured-prompt-utils"
import {
  convertLegacyPromptToStructuredDefinition,
  renderStructuredPromptLegacySnapshot,
  stableSerializePromptSnapshot
} from "../structured-prompt-utils"

type Repeat = { text: string; count: number; prefix?: string; suffix?: string }
type RenderCase = {
  name: string
  definition: Parameters<typeof utils.renderSingleTextRecipe>[0]
  runtimeValues?: Record<string, unknown>
  runtime_repeats?: Record<string, Repeat>
  expected_text?: string
  expected_legacy?: { system_prompt: string; user_prompt: string }
  expected_repeat?: Repeat
  expected_error_code?: string
}
const fixturePath = resolve(process.cwd(), "../../../Docs/fixtures/single-text-recipes")
const readCases = (name: string): RenderCase[] =>
  JSON.parse(readFileSync(resolve(fixturePath, name), "utf8"))
const renderCases = readCases("render-cases.json")
const errorCases = readCases("error-cases.json")
const runtimeValues = (testCase: RenderCase) => ({
  ...testCase.runtimeValues,
  ...Object.fromEntries(
    Object.entries(testCase.runtime_repeats ?? {}).map(([name, repeat]) => [
      name, repeat.text.repeat(repeat.count)
    ])
  )
})

describe("single-text recipe shared fixtures", () => {
  it.each([true, "2", 2.5, 3, NaN, Infinity])("rejects invalid runtime schema versions %s", (version) => {
    const definition = { ...renderCases[0].definition, schema_version: version } as unknown as utils.SingleTextRecipeDefinitionV2
    expect(() => utils.renderSingleTextRecipe(definition)).toThrowError(
      expect.objectContaining({ code: "unsupported_schema_version" })
    )
  })

  it.each([NaN, Infinity, -Infinity])("rejects nonfinite runtime order %s", (order) => {
    const definition: utils.SingleTextRecipeDefinitionV2 = {
      ...renderCases[0].definition,
      blocks: [{ id: "a", name: "A", role: "system", order, content: "" }]
    }
    expect(() => utils.renderSingleTextRecipe(definition)).toThrowError(
      expect.objectContaining({ code: "int_type" })
    )
  })

  it.each([false, true])("checks the budget before materializing repeated values (default=%s)", (useDefault) => {
    const content = "{{x}}".repeat(50)
    const value = "😀".repeat(10000)
    const definition: utils.SingleTextRecipeDefinitionV2 = {
      ...renderCases[0].definition,
      blocks: [{ id: "a", name: "A", role: "system", order: 0, content, is_template: true }],
      variables: [{ name: "x", default_value: useDefault ? value : null }]
    }
    const originalReplace = String.prototype.replace
    const guard = vi.spyOn(String.prototype, "replace").mockImplementation(function (this: string, pattern, replacement) {
      if (String(this) === content) throw new Error("Oversized substitution was materialized")
      return Reflect.apply(originalReplace, this, [pattern, replacement])
    })
    try {
      expect(() => utils.renderSingleTextRecipe(definition, useDefault ? {} : { x: value }))
        .toThrowError(expect.objectContaining({ code: "rendered_output_too_large" }))
    } finally {
      guard.mockRestore()
    }
  })

  it("routes the existing legacy snapshot helper through v2 rendering", () => {
    const testCase = renderCases.find((item) => item.name === "xml-style-preserves-content-and-unicode")!
    expect(utils.renderStructuredPromptLegacySnapshot(testCase.definition)).toEqual({
      systemPrompt: testCase.expected_text, userPrompt: "",
      content: testCase.expected_text
    })
  })

  it.each(renderCases)("$name", (testCase) => {
    const values = runtimeValues(testCase)
    const original = JSON.stringify([testCase.definition, values])
    const repeat = testCase.expected_repeat
    const text = repeat
      ? repeat.prefix + repeat.text.repeat(repeat.count) + repeat.suffix
      : testCase.expected_text
    const legacy = testCase.expected_legacy ?? {
      system_prompt: testCase.definition.assembly_config.target_role === "system" ? text : "",
      user_prompt: testCase.definition.assembly_config.target_role === "user" ? text : ""
    }
    expect(utils.renderSingleTextRecipe).toBeTypeOf("function")
    expect(utils.renderSingleTextRecipe(testCase.definition, values)).toEqual({
      definition_kind: "single_text_recipe", rendered_text: text, legacy
    })
    expect(JSON.stringify([testCase.definition, values])).toBe(original)
  })

  it.each(errorCases)("$name", (testCase) => {
    const values = runtimeValues(testCase)
    const original = JSON.stringify([testCase.definition, values])
    expect(utils.renderSingleTextRecipe).toBeTypeOf("function")
    expect(() => utils.renderSingleTextRecipe(testCase.definition, values)).toThrowError(
      expect.objectContaining({ code: testCase.expected_error_code })
    )
    expect(JSON.stringify([testCase.definition, values])).toBe(original)
  })

  it("does not read undeclared inherited runtime properties", () => {
    const testCase = renderCases.find((item) => item.name === "prototype-shaped-variable-is-ordinary-declaration")!
    expect(utils.renderSingleTextRecipe).toBeTypeOf("function")
    expect(utils.renderSingleTextRecipe(testCase.definition, {})).toEqual({
      definition_kind: "single_text_recipe", rendered_text: "/",
      legacy: { system_prompt: "/", user_prompt: "" }
    })
  })
})

describe("structured-prompt-utils", () => {
  it("converts legacy prompts into the backend-canonical structured shape", () => {
    expect(
      convertLegacyPromptToStructuredDefinition(
        "Be precise about {{topic}}.",
        "Summarize {{topic}} against {{baseline}}."
      )
    ).toEqual({
      schema_version: 1,
      format: "structured",
      assembly_config: {
        legacy_system_roles: ["system", "developer"],
        legacy_user_roles: ["user"],
        block_separator: "\n\n"
      },
      variables: [
        {
          name: "topic",
          label: "Topic",
          required: true,
          input_type: "textarea"
        },
        {
          name: "baseline",
          label: "Baseline",
          required: true,
          input_type: "textarea"
        }
      ],
      blocks: [
        {
          id: "legacy_system",
          name: "System Instructions",
          role: "system",
          kind: "instructions",
          content: "Be precise about {{topic}}.",
          enabled: true,
          order: 10,
          is_template: true
        },
        {
          id: "legacy_user",
          name: "User Prompt",
          role: "user",
          kind: "task",
          content: "Summarize {{topic}} against {{baseline}}.",
          enabled: true,
          order: 20,
          is_template: true
        }
      ]
    })
  })

  it("renders legacy snapshots using assembly_config role mapping and separators", () => {
    expect(
      renderStructuredPromptLegacySnapshot({
        schema_version: 1,
        format: "structured",
        assembly_config: {
          legacy_system_roles: ["developer"],
          legacy_user_roles: ["assistant"],
          block_separator: "\n--\n"
        },
        variables: [],
        blocks: [
          {
            id: "dev_one",
            name: "Developer One",
            role: "developer",
            content: "Rule one",
            enabled: true,
            order: 10,
            is_template: false
          },
          {
            id: "dev_two",
            name: "Developer Two",
            role: "developer",
            content: "Rule two",
            enabled: true,
            order: 20,
            is_template: false
          },
          {
            id: "assistant_example",
            name: "Assistant Example",
            role: "assistant",
            content: "Worked example",
            enabled: true,
            order: 30,
            is_template: false
          }
        ]
      })
    ).toEqual({
      systemPrompt: "Rule one\n--\nRule two",
      userPrompt: "Worked example",
      content: "Worked example"
    })
  })

  it("serializes equivalent prompt snapshots stably regardless of object key order", () => {
    const first = {
      promptFormat: "structured",
      structuredPromptDefinition: {
        format: "structured",
        schema_version: 1,
        blocks: [
          {
            role: "user",
            id: "task",
            content: "Summarize {{topic}}",
            order: 10,
            enabled: true,
            is_template: true,
            name: "Task"
          }
        ],
        assembly_config: {
          block_separator: "\n\n",
          legacy_user_roles: ["user"],
          legacy_system_roles: ["system", "developer"]
        }
      }
    }
    const second = {
      structuredPromptDefinition: {
        schema_version: 1,
        format: "structured",
        assembly_config: {
          legacy_system_roles: ["system", "developer"],
          legacy_user_roles: ["user"],
          block_separator: "\n\n"
        },
        blocks: [
          {
            name: "Task",
            is_template: true,
            enabled: true,
            order: 10,
            content: "Summarize {{topic}}",
            id: "task",
            role: "user"
          }
        ]
      },
      promptFormat: "structured"
    }

    expect(stableSerializePromptSnapshot(first)).toBe(
      stableSerializePromptSnapshot(second)
    )
  })
})
