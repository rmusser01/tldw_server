import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("PlaygroundForm composer options guard", () => {
  it("keeps the chevron toggle, persisted collapse state, and inline send row", () => {
    const sourcePath = path.resolve(__dirname, "../PlaygroundForm.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain("playgroundComposerOptionsExpanded")
    expect(source).toContain('"playgroundComposerOptionsExpanded",\n    true')
    expect(source).toContain('data-testid="composer-options-toggle"')
    expect(source).toContain('data-testid="composer-inline-send-control"')
    expect(source).toContain('id="composer-options-panel"')
    expect(source).toContain("optionsExpanded={composerOptionsExpanded}")
    expect(source).toContain('sendControlPlacement="external"')
  })

  it("keeps the image attachment button discoverable by accessible name", () => {
    const sourcePath = path.resolve(__dirname, "../PlaygroundSendControl.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain('ariaLabel="Attach image"')
    expect(source).toContain('data-testid="attachment-button"')
  })
})
