import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("PlaygroundChat dynamic UI action bridge guard", () => {
  it("wires the default OpenUI action bridge to rendered chat messages", () => {
    const source = fs.readFileSync(
      path.resolve(__dirname, "..", "PlaygroundChat.tsx"),
      "utf8"
    )

    expect(source).toContain("useDynamicUIActionBridge")
    expect(source).toContain("confirmSensitiveValues")
    expect(source).toContain("resolvedDynamicUIAction")
    expect(source).toContain("onDynamicUIAction={resolvedDynamicUIAction}")
  })
})
