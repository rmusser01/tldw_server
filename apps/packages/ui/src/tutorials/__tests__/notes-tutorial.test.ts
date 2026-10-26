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
  it("ends on creating a note without assuming the library is empty", () => {
    // The tour can be replayed from "?" with notes already in the library.
    const step = stepFor("notes-new-note-button")
    expect(step.titleFallback).toBe("Create a note")
    expect(notesLocale.createTitle).toBe("Create a note")
  })

  it("does not ask the user to open the Filters section, which starts open", () => {
    const step = stepFor("notes-sort-select")
    expect(step.contentFallback).not.toMatch(/\bOpen Filters\b/)
    expect(step.contentFallback).toBe(
      "Sort the list by date or title. Use Filters below it to narrow the list by tag."
    )
    expect(notesLocale.filtersContent).toBe(step.contentFallback)
  })
})
