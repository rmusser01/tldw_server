import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const sourcePath = path.resolve(
  process.cwd(),
  "src/components/Option/WorkspacePlayground/ShareDialog.tsx"
)

describe("ShareDialog modal prop guard", () => {
  it("uses destroyOnHidden instead of the deprecated destroyOnClose prop", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("destroyOnHidden")
    expect(source).not.toContain("destroyOnClose")
  })

  it("uses centralized access-level labels instead of hardcoded share option copy", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("ACCESS_LEVEL_LABELS.view_chat")
    expect(source).toContain("ACCESS_LEVEL_LABELS.view_chat_add")
    expect(source).toContain("ACCESS_LEVEL_LABELS.full_edit")
    expect(source).not.toContain('>View & Chat<')
    expect(source).not.toContain('>+ Add Sources<')
    expect(source).not.toContain('>Full Edit<')
  })
})
