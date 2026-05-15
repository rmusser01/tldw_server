import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("PlaygroundForm composer options guard", () => {
  it("keeps the chevron toggle, persisted collapse state, and inline send row", () => {
    const sourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const sendControlSourcePath = path.resolve(
      __dirname,
      "../PlaygroundSendControl.tsx"
    )
    const source = fs.readFileSync(sourcePath, "utf8")
    const sendControlSource = fs.readFileSync(sendControlSourcePath, "utf8")

    expect(source).toContain("playgroundComposerOptionsExpanded")
    expect(source).toContain('data-testid="composer-options-toggle"')
    expect(source).toContain('data-testid="composer-inline-send-control"')
    expect(sendControlSource).toContain("playground:actions.attachImage")
    expect(sendControlSource).toContain("min-h-[44px] min-w-[44px]")
    expect(source).toContain('id="composer-options-panel"')
    expect(source).toContain("optionsExpanded={composerOptionsExpanded}")
    expect(source).toContain('sendControlPlacement="external"')
  })

  it("coordinates composer popovers through a single close-except helper", () => {
    const sourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain("closeComposerPopoversExcept")
    expect(source).toContain('closeComposerPopoversExcept("context")')
    expect(source).toContain('closeComposerPopoversExcept("model")')
    expect(source).toContain('closeComposerPopoversExcept("mcp")')
    expect(source).toContain('closeComposerPopoversExcept("tools")')
    expect(source).toContain('closeComposerPopoversExcept("attachment")')
    expect(source).toContain('closeComposerPopoversExcept("send")')
    expect(source).toContain("handleMcpPopoverChange")
    expect(source).toContain("handleToolsPopoverChange")
    expect(source).toContain("handleAttachmentMenuChange")
    expect(source).toContain("handleSendMenuChange")
  })

  it("keeps legacy composer controls before transient notices", () => {
    const sourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")
    const legacyReturnStart = source.indexOf("return (", source.indexOf("if (nextgenComposerEnabled)"))
    const toolbarIndex = source.indexOf("{composerToolbarNode}", legacyReturnStart)
    const noticesIndex = source.indexOf("{composerNoticesNode}", legacyReturnStart)

    expect(toolbarIndex).toBeGreaterThan(legacyReturnStart)
    expect(noticesIndex).toBeGreaterThan(toolbarIndex)
  })
})
