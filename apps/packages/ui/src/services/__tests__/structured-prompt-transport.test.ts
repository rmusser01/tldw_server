import { describe, expect, it } from "vitest"
import { existsSync, readFileSync } from "node:fs"
import { resolve } from "node:path"

import { parseStructuredPromptDefinitionForTransport } from "@/services/structured-prompt-transport"

const recipe = (targetRole: "system" | "user" = "user") => ({
  schema_version: 2,
  format: "structured",
  definition_kind: "single_text_recipe",
  variables: [
    {
      name: "topic",
      label: "Topic",
      description: null,
      required: true,
      default_value: null,
      input_type: "text",
      options: null,
      max_length: null
    }
  ],
  blocks: [
    {
      id: "objective",
      name: "Objective",
      section_key: "objective",
      role: targetRole,
      kind: "objective",
      content: "Explain {{topic}}.",
      enabled: true,
      order: 10,
      is_template: true
    }
  ],
  assembly_config: {
    assembly_mode: "single_text",
    target_role: targetRole,
    render_format: "markdown",
    block_separator: "\n\n"
  }
})

const v1Definition = () => ({
  schema_version: 1,
  format: "structured",
  variables: [{ name: "topic", required: true, input_type: "text" }],
  blocks: [
    {
      id: "task",
      name: "Task",
      role: "user",
      content: "Explain {{topic}}.",
      enabled: true,
      order: 10,
      is_template: true
    }
  ]
})

const clone = <T>(value: T): T => structuredClone(value)

type V1TransportCase = {
  name: string
  input: Record<string, unknown>
  expected: Record<string, unknown>
}
type V1IntegerSyncCase = {
  name: string
  field: "order" | "max_length" | "schema_version"
  input_value: number | string | null
  python_value: number | null
  sync_eligible: boolean
}
const fixtureRelativePath =
  "Docs/fixtures/single-text-recipes/v1-transport-cases.json"
const fixturePath = ["", "..", "../..", "../../.."]
  .map((prefix) => resolve(process.cwd(), prefix, fixtureRelativePath))
  .find(existsSync)
if (!fixturePath) throw new Error("v1 transport fixture not found")
const v1TransportCases = JSON.parse(
  readFileSync(fixturePath, "utf8")
) as V1TransportCase[]
const v1IntegerSyncCases = JSON.parse(
  readFileSync(resolve(fixturePath, "..", "v1-integer-sync-cases.json"), "utf8")
) as V1IntegerSyncCase[]

const invalidRecipeCases: Array<[string, (value: any) => void]> = [
  [
    "extra root field",
    (value) => {
      value.extra = true
    }
  ],
  [
    "missing format",
    (value) => {
      delete value.format
    }
  ],
  [
    "wrong definition kind",
    (value) => {
      value.definition_kind = "messages"
    }
  ],
  [
    "variables are null",
    (value) => {
      value.variables = null
    }
  ],
  [
    "variables not an array",
    (value) => {
      value.variables = {}
    }
  ],
  [
    "too many variables",
    (value) => {
      value.variables = Array.from({ length: 101 }, (_, index) => ({
        name: `v${index}`
      }))
    }
  ],
  [
    "variable extra field",
    (value) => {
      value.variables[0].extra = true
    }
  ],
  [
    "variable name is missing",
    (value) => {
      delete value.variables[0].name
    }
  ],
  [
    "variable name is not an identifier",
    (value) => {
      value.variables[0].name = "bad-name"
    }
  ],
  [
    "variable name is too long",
    (value) => {
      value.variables[0].name = "a".repeat(129)
    }
  ],
  [
    "variable label is not a string",
    (value) => {
      value.variables[0].label = 1
    }
  ],
  [
    "variable label is too long",
    (value) => {
      value.variables[0].label = "a".repeat(201)
    }
  ],
  [
    "variable description is not a string",
    (value) => {
      value.variables[0].description = 1
    }
  ],
  [
    "variable required is not boolean",
    (value) => {
      value.variables[0].required = "true"
    }
  ],
  [
    "variable input type is not a string",
    (value) => {
      value.variables[0].input_type = 1
    }
  ],
  [
    "variable options are not an array",
    (value) => {
      value.variables[0].options = {}
    }
  ],
  [
    "variable options contain non-string",
    (value) => {
      value.variables[0].options = [1]
    }
  ],
  [
    "variable max length is invalid",
    (value) => {
      value.variables[0].max_length = 0
    }
  ],
  [
    "variable max length is fractional",
    (value) => {
      value.variables[0].max_length = 1.5
    }
  ],
  [
    "duplicate variable",
    (value) => {
      value.variables.push(clone(value.variables[0]))
    }
  ],
  [
    "blocks are null",
    (value) => {
      value.blocks = null
    }
  ],
  [
    "too many blocks",
    (value) => {
      value.blocks = Array.from({ length: 101 }, (_, index) => ({
        ...clone(value.blocks[0]),
        id: `b${index}`,
        order: index
      }))
    }
  ],
  [
    "block extra field",
    (value) => {
      value.blocks[0].extra = true
    }
  ],
  [
    "block id is missing",
    (value) => {
      delete value.blocks[0].id
    }
  ],
  [
    "blank block id",
    (value) => {
      value.blocks[0].id = "   "
    }
  ],
  [
    "block id is too long",
    (value) => {
      value.blocks[0].id = "a".repeat(129)
    }
  ],
  [
    "block name is empty",
    (value) => {
      value.blocks[0].name = ""
    }
  ],
  [
    "block name is not a string",
    (value) => {
      value.blocks[0].name = 1
    }
  ],
  [
    "block name is too long",
    (value) => {
      value.blocks[0].name = "a".repeat(201)
    }
  ],
  [
    "block role is invalid",
    (value) => {
      value.blocks[0].role = "assistant"
    }
  ],
  [
    "block kind is not a string",
    (value) => {
      value.blocks[0].kind = 1
    }
  ],
  [
    "block content is not string",
    (value) => {
      value.blocks[0].content = 1
    }
  ],
  [
    "block content is too long",
    (value) => {
      value.blocks[0].content = "a".repeat(20_001)
    }
  ],
  [
    "block enabled is not boolean",
    (value) => {
      value.blocks[0].enabled = 1
    }
  ],
  [
    "block template flag is not boolean",
    (value) => {
      value.blocks[0].is_template = 1
    }
  ],
  [
    "block order is missing",
    (value) => {
      delete value.blocks[0].order
    }
  ],
  [
    "block order is fractional",
    (value) => {
      value.blocks[0].order = 1.5
    }
  ],
  [
    "block order is unsafe",
    (value) => {
      value.blocks[0].order = Number.MAX_SAFE_INTEGER + 1
    }
  ],
  [
    "block role differs from target",
    (value) => {
      value.blocks[0].role = "system"
    }
  ],
  [
    "duplicate block id",
    (value) => {
      value.blocks.push({ ...clone(value.blocks[0]), order: 20 })
    }
  ],
  [
    "duplicate normalized block id",
    (value) => {
      value.blocks.push({
        ...clone(value.blocks[0]),
        id: " objective ",
        order: 20
      })
    }
  ],
  [
    "unknown template reference",
    (value) => {
      value.blocks[0].content = "{{missing}}"
    }
  ],
  [
    "assembly config is missing",
    (value) => {
      delete value.assembly_config
    }
  ],
  [
    "assembly config is null",
    (value) => {
      value.assembly_config = null
    }
  ],
  [
    "assembly config has extra field",
    (value) => {
      value.assembly_config.extra = true
    }
  ],
  [
    "assembly mode is invalid",
    (value) => {
      value.assembly_config.assembly_mode = "messages"
    }
  ],
  [
    "target role is invalid",
    (value) => {
      value.assembly_config.target_role = "assistant"
    }
  ],
  [
    "render format is invalid",
    (value) => {
      value.assembly_config.render_format = "html"
    }
  ],
  [
    "separator is not string",
    (value) => {
      value.assembly_config.block_separator = 1
    }
  ],
  [
    "separator is too long",
    (value) => {
      value.assembly_config.block_separator = "a".repeat(21)
    }
  ],
  [
    "section key is not a string",
    (value) => {
      value.blocks[0].section_key = 1
    }
  ],
  [
    "section key is too long",
    (value) => {
      value.blocks[0].section_key = "a".repeat(129)
    }
  ],
  [
    "enabled XML section key is missing",
    (value) => {
      value.assembly_config.render_format = "xml"
      value.blocks[0].section_key = null
    }
  ],
  [
    "enabled XML section key is invalid",
    (value) => {
      value.assembly_config.render_format = "xml"
      value.blocks[0].section_key = "1bad"
    }
  ],
  [
    "duplicate XML section key",
    (value) => {
      value.assembly_config.render_format = "xml"
      value.blocks.push({ ...clone(value.blocks[0]), id: "second", order: 20 })
    }
  ],
  [
    "XML closing tag collision",
    (value) => {
      value.assembly_config.render_format = "xml"
      value.blocks[0].content = "before </objective> after"
    }
  ],
  [
    "authored rendered output is too large",
    (value) => {
      value.blocks = Array.from({ length: 6 }, (_, index) => ({
        ...clone(value.blocks[0]),
        id: `b${index}`,
        name: "a".repeat(200),
        content: "a".repeat(20_000),
        is_template: false,
        order: index
      }))
    }
  ],
  [
    "nested runtime map",
    (value) => {
      value.variables[0].default_value = {
        resolved_values: { topic: "secret" }
      }
    }
  ]
]

const invalidV1Cases: Array<[string, (value: any) => void]> = [
  [
    "extra root field",
    (value) => {
      value.extra = true
    }
  ],
  [
    "wrong format",
    (value) => {
      value.format = "legacy"
    }
  ],
  [
    "variables are not an array",
    (value) => {
      value.variables = {}
    }
  ],
  [
    "variable extra field",
    (value) => {
      value.variables[0].extra = true
    }
  ],
  [
    "variable name is missing",
    (value) => {
      delete value.variables[0].name
    }
  ],
  [
    "variable required is not boolean",
    (value) => {
      value.variables[0].required = null
    }
  ],
  [
    "variable options contain non-string",
    (value) => {
      value.variables[0].options = [1]
    }
  ],
  [
    "variable max length is below one",
    (value) => {
      value.variables[0].max_length = 0
    }
  ],
  [
    "duplicate trimmed variable name",
    (value) => {
      value.variables.push({ name: " topic " })
    }
  ],
  [
    "blocks are not an array",
    (value) => {
      value.blocks = {}
    }
  ],
  [
    "block extra field",
    (value) => {
      value.blocks[0].extra = true
    }
  ],
  [
    "block id is empty",
    (value) => {
      value.blocks[0].id = ""
    }
  ],
  [
    "block name is empty",
    (value) => {
      value.blocks[0].name = ""
    }
  ],
  [
    "block role is invalid",
    (value) => {
      value.blocks[0].role = "tool"
    }
  ],
  [
    "block content is not a string",
    (value) => {
      value.blocks[0].content = 1
    }
  ],
  [
    "block order is not an integer",
    (value) => {
      value.blocks[0].order = 1.5
    }
  ],
  [
    "duplicate trimmed block id",
    (value) => {
      value.blocks.push({ ...clone(value.blocks[0]), id: " task ", order: 20 })
    }
  ],
  [
    "unknown template reference",
    (value) => {
      value.blocks[0].content = "{{missing}}"
    }
  ],
  [
    "assembly config is not an object",
    (value) => {
      value.assembly_config = []
    }
  ],
  [
    "assembly config has an extra field",
    (value) => {
      value.assembly_config = { extra: true }
    }
  ],
  [
    "legacy system roles contain non-string",
    (value) => {
      value.assembly_config = { legacy_system_roles: [1] }
    }
  ],
  [
    "legacy system roles are null",
    (value) => {
      value.assembly_config = { legacy_system_roles: null }
    }
  ],
  [
    "legacy user roles are not an array",
    (value) => {
      value.assembly_config = { legacy_user_roles: {} }
    }
  ],
  [
    "block separator is not a string",
    (value) => {
      value.assembly_config = { block_separator: 1 }
    }
  ],
  [
    "nested runtime map",
    (value) => {
      value.variables[0].default_value = { variable_values: {} }
    }
  ]
]

describe("structured prompt transport validation", () => {
  it.each(v1TransportCases)(
    "canonicalizes shared Python-compatible v1 case: $name",
    ({ input, expected }) => {
      const before = JSON.stringify(input)

      expect(
        parseStructuredPromptDefinitionForTransport(input, "structured", 1)
      ).toEqual(expected)
      expect(JSON.stringify(input)).toBe(before)
    }
  )

  it.each(v1IntegerSyncCases)(
    "enforces shared v1 integer sync eligibility: $name",
    ({
      field,
      input_value: inputValue,
      python_value: pythonValue,
      sync_eligible: syncEligible
    }) => {
      const input = v1Definition()
      if (field === "schema_version")
        input.schema_version = inputValue as number
      else if (field === "order") input.blocks[0].order = inputValue as number
      else
        (input.variables[0] as Record<string, unknown>).max_length = inputValue
      const before = JSON.stringify(input)

      if (syncEligible) {
        const parsed = parseStructuredPromptDefinitionForTransport(
          input,
          "structured",
          1
        )
        const parsedValue =
          field === "schema_version"
            ? parsed!.schema_version
            : field === "order"
              ? parsed!.blocks[0].order
              : parsed!.variables[0].max_length
        expect(parsedValue).toBe(pythonValue)
      } else {
        expect(() =>
          parseStructuredPromptDefinitionForTransport(input, "structured", 1)
        ).toThrow("invalid_prompt_definition")
      }
      expect(JSON.stringify(input)).toBe(before)
    }
  )

  it.each(
    (["order", "max_length", "schema_version"] as const).flatMap((field) =>
      ["", "\u0085", "\ufeff"].map((whitespace) => ({ field, whitespace }))
    )
  )(
    "retains strict v2 rejection of integer strings in $field ($whitespace)",
    ({ field, whitespace }) => {
      const input = recipe()
      const value = `${whitespace}${field === "schema_version" ? "0_2" : "1_0"}.00${whitespace}`
      if (field === "schema_version") {
        ;(input as Record<string, unknown>).schema_version = value
      } else if (field === "order") {
        ;(input.blocks[0] as Record<string, unknown>).order = value
      } else {
        ;(input.variables[0] as Record<string, unknown>).max_length = value
      }
      const before = JSON.stringify(input)

      expect(() =>
        parseStructuredPromptDefinitionForTransport(input, "structured", 2)
      ).toThrow("invalid_prompt_definition")
      expect(JSON.stringify(input)).toBe(before)
    }
  )

  it.each(["system", "user"] as const)(
    "accepts a complete %s-target recipe without mutating it",
    (targetRole) => {
      const input = recipe(targetRole)
      const before = JSON.stringify(input)

      const parsed = parseStructuredPromptDefinitionForTransport(
        input,
        "structured",
        2
      )

      expect(parsed).toEqual(input)
      expect(JSON.stringify(input)).toBe(before)
    }
  )

  it.each(invalidRecipeCases)("rejects v2 %s", (_name, mutate) => {
    const input = recipe()
    mutate(input)

    expect(() =>
      parseStructuredPromptDefinitionForTransport(input, "structured", 2)
    ).toThrow()
  })

  it.each(invalidV1Cases)("rejects v1 %s", (_name, mutate) => {
    const input = v1Definition()
    mutate(input)

    expect(() =>
      parseStructuredPromptDefinitionForTransport(input, "structured", 1)
    ).toThrow()
  })

  it.each([
    ["v1 definition with v2 outer version", v1Definition(), 2],
    ["v2 definition with v1 outer version", recipe(), 1],
    ["v2 definition with legacy outer format", recipe(), null, "legacy"],
    [
      "future definition and outer version",
      { ...recipe(), schema_version: 99 },
      99
    ],
    ["future outer version", recipe(), 99]
  ])("rejects %s", (_name, definition, outerVersion, format = "structured") => {
    expect(() =>
      parseStructuredPromptDefinitionForTransport(
        definition,
        format,
        outerVersion
      )
    ).toThrow()
  })

  it("accepts v1 with exact outer identity", () => {
    expect(
      parseStructuredPromptDefinitionForTransport(
        v1Definition(),
        "structured",
        1
      )
    ).toEqual({
      schema_version: 1,
      format: "structured",
      variables: [
        {
          name: "topic",
          label: null,
          description: null,
          required: true,
          default_value: null,
          input_type: "text",
          options: null,
          max_length: null
        }
      ],
      blocks: [
        {
          id: "task",
          name: "Task",
          role: "user",
          kind: null,
          content: "Explain {{topic}}.",
          enabled: true,
          order: 10,
          is_template: true
        }
      ],
      assembly_config: {
        legacy_system_roles: ["system", "developer"],
        legacy_user_roles: ["user"],
        block_separator: "\n\n"
      }
    })
  })

  it("treats structural key words inside authored strings as opaque", () => {
    const input = recipe()
    input.blocks[0].content =
      "runtime_values variable_values resolved_values {{topic}}"
    expect(
      parseStructuredPromptDefinitionForTransport(input, "structured", 2)
    ).toBeTruthy()
  })

  it("accepts an XML-disabled block without a section key", () => {
    const input = recipe()
    input.assembly_config.render_format = "xml"
    input.blocks[0].enabled = false
    input.blocks[0].section_key = null

    expect(
      parseStructuredPromptDefinitionForTransport(input, "structured", 2)
    ).toBeTruthy()
  })
})
