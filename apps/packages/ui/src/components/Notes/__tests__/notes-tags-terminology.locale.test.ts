import { describe, expect, it } from "vitest"
import appOptionLocale from "../../../assets/locale/en/option.json"
import appSidepanelLocale from "../../../assets/locale/en/sidepanel.json"
import publicOptionLocale from "../../../public/_locales/en/option.json"
import publicSidepanelLocale from "../../../public/_locales/en/sidepanel.json"

const notesTagCopy = {
  keywordsPlaceholder: "Filter by tag",
  keywordsBrowse: "Browse tags",
  keywordPickerTitle: "Browse tags",
  keywordPickerSearch: "Search tags",
  keywordPickerCount: "{{count}} tags",
  keywordPickerEmpty: "No tags found",
  keywordsEditorPlaceholder: "Tags",
  tagsHelp: "Tags help you organize and filter notes. Add tags in the editor, then filter here."
} as const

const publicNotesTagCopy = {
  notesSearch_keywordsPlaceholder: "Filter by tag",
  notesSearch_keywordsBrowse: "Browse tags",
  notesSearch_keywordPickerTitle: "Browse tags",
  notesSearch_keywordPickerSearch: "Search tags",
  notesSearch_keywordPickerCount: "{{count}} tags",
  notesSearch_keywordPickerEmpty: "No tags found",
  notesSearch_keywordsEditorPlaceholder: "Tags",
  notesSearch_tagsHelp: "Tags help you organize and filter notes. Add tags in the editor, then filter here."
} as const

const keywordCopyPattern = /\bkeywords?\b/i

describe("notes tag terminology locale contract", () => {
  it("uses Tags in app-facing Notes organization copy", () => {
    for (const [key, expected] of Object.entries(notesTagCopy)) {
      const value = (appOptionLocale as any).notesSearch[key]

      expect(value).toBe(expected)
      expect(value).not.toMatch(keywordCopyPattern)
    }
  })

  it("keeps public extension Notes locale copy aligned with Tags terminology", () => {
    for (const [key, expected] of Object.entries(publicNotesTagCopy)) {
      const value = (publicOptionLocale as any)[key]?.message

      expect(value).toBe(expected)
      expect(value).not.toMatch(keywordCopyPattern)
    }
  })

  it("uses Tags in directly connected Web Clipper capture copy", () => {
    expect((appSidepanelLocale as any).clipper.tagsLabel).toBe("Tags")
    expect((appSidepanelLocale as any).clipper.tagsPlaceholder).not.toMatch(keywordCopyPattern)
    expect((publicSidepanelLocale as any).clipper_tagsLabel.message).toBe("Tags")
    expect((publicSidepanelLocale as any).clipper_tagsPlaceholder.message).not.toMatch(
      keywordCopyPattern
    )
  })
})
