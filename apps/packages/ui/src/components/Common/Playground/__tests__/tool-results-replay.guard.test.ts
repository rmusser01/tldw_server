import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const uiRoot = path.resolve(__dirname, "../../../..")

const readSource = (relativePath: string) =>
  fs.readFileSync(path.resolve(uiRoot, relativePath), "utf8")

describe("tool result replay wiring", () => {
  it("passes persisted toolResults through WebUI chat messages", () => {
    const source = readSource("components/Option/Playground/PlaygroundChat.tsx")

    expect(source).toContain("toolResults={message?.toolResults}")
  })

  it("passes message toolResults into ToolCallBlock", () => {
    const source = readSource("components/Common/Playground/Message.tsx")

    expect(source).toContain("<ToolCallBlock")
    expect(source).toContain("results={props.toolResults}")
  })

  it("passes persisted toolResults through extension sidepanel messages", () => {
    const source = readSource("components/Sidepanel/Chat/body.tsx")

    expect(source).toContain("toolResults={message?.toolResults}")
  })

  it("keeps toolResults available in compare clusters", () => {
    const source = readSource(
      "components/Option/Playground/PlaygroundCompareCluster.tsx"
    )

    expect(source).toMatch(
      /toolResults=\{(?:message|userMessage)\?\.toolResults\}/
    )
  })
})
