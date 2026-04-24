import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("ComposerToolbar layout guard", () => {
  it("keeps mode-specific layout markers and advanced-controls affordances", () => {
    const sourcePath = path.resolve(__dirname, "../ComposerToolbar.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).toContain('data-testid="composer-options-panel"')
    expect(source).toContain('data-playground-toolbar-layout="casual"')
    expect(source).toContain('data-playground-toolbar-layout="pro-split"')
    expect(source).toContain('data-testid="composer-pro-context-panel"')
    expect(source).toContain('data-testid="composer-pro-generation-panel"')
    expect(source).toContain('data-testid="composer-casual-advanced-chip"')
    expect(source).toContain(
      'data-testid="composer-casual-advanced-controls-row"'
    )
    expect(source).toContain('data-testid="composer-formatting-guide-toggle"')
    expect(source).toContain("composer-session-status-chip")
    expect(source).toContain('data-testid="composer-casual-persistence-chip"')
    expect(source).toContain('data-testid="composer-casual-token-chip"')
    expect(source).toContain('data-playground-toolbar-row="primary"')
    expect(source).toContain('data-playground-toolbar-row="actions"')
    expect(source).toContain('data-testid="composer-advanced-toggle"')
    expect(source).toContain('sendControlPlacement = "toolbar"')
    expect(source).toContain("optionsExpanded = true")
    expect(source).not.toContain(
      'data-testid="composer-casual-runtime-context-chip"'
    )
    expect(source).toContain("playgroundComposerAdvancedControlsOpen")
    expect(source).toContain("playgroundComposerCasualAdvancedControlsOpen")
    expect(source).toContain(
      "PLAYGROUND_APPEND_FORMATTING_GUIDE_PROMPT_STORAGE_KEY"
    )
  })
})
