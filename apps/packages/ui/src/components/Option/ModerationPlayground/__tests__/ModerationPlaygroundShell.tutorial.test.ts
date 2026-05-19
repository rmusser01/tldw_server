import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readShellSource = () =>
  fs.readFileSync(
    path.resolve(__dirname, "..", "ModerationPlaygroundShell.tsx"),
    "utf8"
  )

describe("ModerationPlaygroundShell tutorial integration", () => {
  it("calls startTutorial with moderation-basics", () => {
    const source = readShellSource()
    expect(source).toContain('startTutorial("moderation-basics")')
  })

  it("imports useTutorialStore", () => {
    const source = readShellSource()
    expect(source).toContain("useTutorialStore")
  })

  it("checks tutorial completion before auto-starting", () => {
    const source = readShellSource()
    expect(source).toContain('isTutorialCompleted("moderation-basics")')
  })

  it("has data-testid on hero section", () => {
    const source = readShellSource()
    expect(source).toContain('data-testid="moderation-hero"')
  })

  it("has data-testid on tab buttons", () => {
    const source = readShellSource()
    expect(source).toContain("data-testid={`moderation-tab-${tab.key}`}")
  })
})

describe("moderation tutorial in registry", () => {
  it("registry imports moderation tutorials", () => {
    const registrySource = fs.readFileSync(
      path.resolve(__dirname, "../../../../tutorials/registry.ts"),
      "utf8"
    )
    expect(registrySource).toContain("moderationTutorials")
    expect(registrySource).toContain("...moderationTutorials")
  })
})
