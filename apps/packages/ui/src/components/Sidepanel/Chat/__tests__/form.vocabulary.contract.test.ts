import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

// XP-12 (#3101): the side panel and the full page share one placeholder key,
// "Write a message…". The old side-panel text promised "/ for commands".
const formSource = readFileSync(path.resolve(__dirname, "../form.tsx"), "utf8")

describe("side-panel composer vocabulary contract", () => {
  it("uses the full page's placeholder key", () => {
    expect(formSource).toContain('"playground:composer.calmPlaceholder"')
    expect(formSource).not.toContain('"form.textarea.placeholder"')
  })
})
