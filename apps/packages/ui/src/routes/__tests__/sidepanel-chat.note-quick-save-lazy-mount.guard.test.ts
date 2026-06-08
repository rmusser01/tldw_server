import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import path from "node:path"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const sourcePath = path.resolve(testDir, "../sidepanel-chat.tsx")

describe("sidepanel-chat note quick save mounting", () => {
  it("only mounts the quick-save modal while it is open", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("{noteModalOpen ? (")
    expect(source).not.toContain(
      'import NoteQuickSaveModal from "~/components/Sidepanel/Notes/NoteQuickSaveModal"'
    )
    expect(source).toContain('import("~/components/Sidepanel/Notes/NoteQuickSaveModal")')
    expect(source).toContain("<LazyNoteQuickSaveModal")
  })

  it("persists capture context through supported note fields", () => {
    const source = readFileSync(sourcePath, "utf8")

    expect(source).toContain("buildCapturedNoteContent")
    expect(source).toContain("CAPTURED_NOTE_KEYWORD")
    expect(source).toContain("keywords: [CAPTURED_NOTE_KEYWORD]")
    expect(source).not.toContain("source_url: noteSourceUrl")
    expect(source).not.toContain('origin: "context-menu"')
  })
})
