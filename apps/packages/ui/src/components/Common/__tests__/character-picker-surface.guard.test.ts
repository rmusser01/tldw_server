import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

describe("character picker popup surfaces", () => {
  it("keeps shared assistant picker popups on an opaque themed surface", () => {
    const sourcePath = path.resolve(__dirname, "../AssistantSelect.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).not.toContain("overlayClassName")
    expect(source).toContain('root: "assistant-select-dropdown')
    expect(source).toContain('data-testid="assistant-select-panel"')
    expect(source).toContain("bg-surface")
    expect(source).not.toContain("bg-background shadow-lg")
  })

  it("keeps character picker popups on an opaque themed surface", () => {
    const sourcePath = path.resolve(__dirname, "../CharacterSelect.tsx")
    const source = fs.readFileSync(sourcePath, "utf8")

    expect(source).not.toContain("overlayClassName")
    expect(source).toContain('root: "character-select-dropdown')
    expect(source).toContain('data-testid="character-select-popup"')
    expect(source).toContain("bg-surface")
  })

  it("uses current AntD popup APIs for character chat controls", () => {
    const greetingSource = fs.readFileSync(
      path.resolve(__dirname, "../ChatGreetingPicker.tsx"),
      "utf8"
    )
    const attachmentSource = fs.readFileSync(
      path.resolve(__dirname, "../../Option/Playground/PlaygroundSendControl.tsx"),
      "utf8"
    )
    const toolsSource = fs.readFileSync(
      path.resolve(__dirname, "../../Option/Playground/PlaygroundToolsPopover.tsx"),
      "utf8"
    )

    expect(greetingSource).not.toContain("dropdownRender")
    expect(greetingSource).toContain("popupRender")
    expect(attachmentSource).not.toContain("overlayClassName")
    expect(attachmentSource).toContain('root: "playground-attachment-menu"')
    expect(toolsSource).not.toContain("overlayClassName")
    expect(toolsSource).toContain('root: "playground-more-tools"')
  })
})
