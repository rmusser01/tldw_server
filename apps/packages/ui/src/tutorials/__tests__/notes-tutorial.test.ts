import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { getTutorialById } from "../registry"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../")
const notesLocale = (
  JSON.parse(
    readFileSync(path.resolve(srcRoot, "assets/locale/en/tutorials.json"), "utf8")
  ) as { notes: { basics: Record<string, string> } }
).notes.basics

const stepFor = (testId: string) => {
  const tutorial = getTutorialById("notes-basics")
  expect(tutorial).toBeDefined()
  const step = tutorial!.steps.find((candidate) =>
    candidate.target.includes(`"${testId}"`)
  )
  expect(step).toBeDefined()
  return step!
}

describe("Notes tour copy (NO-02)", () => {
  it("introduces creating a note without assuming the library is empty", () => {
    // The tour can be replayed from "?" with notes already in the library.
    const step = stepFor("notes-new-note-button")
    expect(step.titleFallback).toBe("Create a note")
    expect(notesLocale.createTitle).toBe("Create a note")
  })

  it("shows the keyboard shortcuts for saving a note", () => {
    const step = stepFor("notes-save-button")
    expect(step.contentFallback).toContain("Ctrl+S or ⌘S")
    expect(notesLocale.quickStartSave).toBe(step.contentFallback)
  })
})
