import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (relativePath: string) =>
  fs.readFileSync(path.resolve(__dirname, relativePath), "utf8")

describe("dynamic UI surface guard", () => {
  it("does not active-render OpenUI when a message caller omits surface", () => {
    const source = readSource("../MessageContent.tsx")
    expect(source).toContain('dynamicUISurface ?? "artifact"')
    expect(source).not.toContain('dynamicUISurface ?? "web-chat"')
  })

  it("opts the main /chat transcript into active dynamic UI rendering explicitly", () => {
    const source = readSource("../../../Option/Playground/PlaygroundChat.tsx")
    expect(source).toContain('dynamicUISurface="web-chat"')
  })
})
