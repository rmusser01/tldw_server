import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

const readSource = (filename: string) =>
  fs.readFileSync(path.resolve(__dirname, "..", filename), "utf8")

describe("writing tiptap parity guards", () => {
  it("routes tiptap edits through applyPromptValue with prompt_rich", () => {
    const source = readSource("index.tsx")

    expect(source).toContain("applyPromptValue(plain, { promptRich: json })")
    expect(source).not.toContain("setEditorText(plain)")
  })

  it("keeps tiptap rendered in split view", () => {
    const source = readSource("index.tsx")

    expect(source).toMatch(/editorView === "split"[\s\S]*editorMode === "tiptap"[\s\S]*LazyWritingTipTapEditor/)
  })

  it("uses the shared editor adapter for selection-based helpers", () => {
    const source = readSource("index.tsx")

    expect(source).toContain("activeEditorAdapterRef")
    expect(source).toContain("createTextareaEditorAdapter(editorRef)")
  })
})
