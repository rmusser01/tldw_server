import { describe, expect, it } from "vitest"
import { validateWithAjv } from "../lib/ajv"

describe("workflow input schema validation", () => {
  const schema = { type: "object" as const, properties: { text: { type: "string" as const, minLength: 1 } }, required: ["text"], additionalProperties: false }
  it("names the actual missing input and preserves valid input without coercion", async () => {
    expect(await validateWithAjv({}, schema, { strict: true })).toEqual([expect.stringContaining("required property 'text'")])
    const inputs = { text: "hello" }
    expect(await validateWithAjv(inputs, schema, { strict: true })).toEqual([])
    expect(inputs).toEqual({ text: "hello" })
    expect(await validateWithAjv({ text: 7 }, schema, { strict: true })).toEqual([expect.stringContaining("string")])
  })
  it.each([{ ...schema, misspelledRequired: ["text"] }, { ...schema, $ref: "https://invalid.example/schema" }, { ...schema, $async: true }])("refuses unsupported schema constraints instead of silently dispatching", async (unsupported) => {
    await expect(validateWithAjv({}, unsupported, { strict: true })).rejects.toThrow()
  })
})
