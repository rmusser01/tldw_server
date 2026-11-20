import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

// XP-12 (#3101): the side panel and the full page share one placeholder key,
// "Write a message…". The old side-panel text promised "/ for commands".
const formSource = readFileSync(path.resolve(__dirname, "../form.tsx"), "utf8")

describe("side-panel composer vocabulary contract", () => {
  // Review fix: the side panel names "/ commands" too, with the full page's hint.
  it("names the slash commands with the full page's shortcut hint", () => {
    expect(formSource).toContain("composerShortcutHint(t, false)")
  })

  // Review fix: the side panel has no image provider picker to point at.
  it("does not point at an image provider picker the side panel lacks", () => {
    expect(formSource).not.toContain("Pick an Image provider in More tools")
  })

  it("uses the full page's placeholder key", () => {
    expect(formSource).toContain('"playground:composer.calmPlaceholder"')
    expect(formSource).not.toContain('"form.textarea.placeholder"')
  })
})
