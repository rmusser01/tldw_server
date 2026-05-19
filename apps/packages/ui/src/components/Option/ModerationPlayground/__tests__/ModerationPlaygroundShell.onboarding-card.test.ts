import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readShellSource = () =>
  fs.readFileSync(
    path.resolve(__dirname, "..", "ModerationPlaygroundShell.tsx"),
    "utf8"
  )

describe("ModerationPlaygroundShell onboarding card", () => {
  it("links to the Family Guardrails Wizard", () => {
    const source = readShellSource()
    expect(source).toContain("/settings/family-guardrails")
  })

  it("has a data-testid for the family guardrails link", () => {
    const source = readShellSource()
    expect(source).toContain('data-testid="moderation-family-guardrails-link"')
  })

  it("shows recommended tab order guidance", () => {
    const source = readShellSource()
    expect(source).toContain("Start here")
  })

  it("uses i18n for onboarding text", () => {
    const source = readShellSource()
    expect(source).toContain("moderationPlayground.onboarding.title")
    expect(source).toContain("moderationPlayground.onboarding.guardrailsCta")
  })
})
