import { describe, expect, it } from "vitest"
import { encode } from "gpt-tokenizer"
import { Formatter } from "../Formatter"

describe("Formatter", () => {
  it("returns directory tree and token count", async () => {
    const output = await Formatter.formatAsync(
      [{ name: "a.ts", path: "a.ts", type: "file" }],
      [{ path: "a.ts", text: "const a=1" }]
    )
    expect(output.directoryTree).toContain("a.ts")
    expect(output.tokenCount).toBeGreaterThan(0)
  })
})

it("counts the complete output including file boundaries and whitespace", async () => {
  const output = await Formatter.formatAsync(
    [{ path: "README.md" }],
    [{ path: "README.md", text: " café 星\n" }]
  )
  expect(output.tokenCount).toBe(
    encode(`${output.directoryTree}\n\n${output.fileContents}`).length
  )
})
