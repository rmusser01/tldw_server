import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

type ExtensionLocaleJson = Record<string, { message?: unknown }>

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../../")

const optionLocale = JSON.parse(
  readFileSync(
    path.resolve(srcRoot, "public/_locales/en/option.json"),
    "utf8"
  )
) as ExtensionLocaleJson

const sidepanelLocale = JSON.parse(
  readFileSync(
    path.resolve(srcRoot, "public/_locales/en/sidepanel.json"),
    "utf8"
  )
) as ExtensionLocaleJson

const message = (locale: ExtensionLocaleJson, key: string) =>
  String(locale[key]?.message ?? "")

describe("notes tags terminology locale contract", () => {
  it("uses Tags language for notes organization controls while preserving keywords keys", () => {
    expect(message(optionLocale, "notesSearch_keywordsPlaceholder")).toBe("Filter by tag")
    expect(message(optionLocale, "notesSearch_keywordsEditorPlaceholder")).toBe("Tags")
    expect(message(optionLocale, "notesSearch_tagsHelp")).toBe(
      "Tags help you find this note using the tag filter on the left."
    )
    expect(message(optionLocale, "notesSearch_keywordsBrowse")).toBe("Browse tags")
    expect(message(optionLocale, "notesSearch_keywordPickerTitle")).toBe("Browse tags")
    expect(message(optionLocale, "notesSearch_keywordPickerSearch")).toBe("Search tags")
    expect(message(optionLocale, "notesSearch_keywordPickerCount")).toBe("{{count}} tags")
    expect(message(optionLocale, "notesSearch_keywordPickerEmpty")).toBe("No tags found")
  })

  it("keeps directly connected clipper capture controls on Tags language", () => {
    expect(message(sidepanelLocale, "clipper_tagsLabel")).toBe("Tags")
    expect(message(sidepanelLocale, "clipper_tagsPlaceholder")).not.toMatch(/keyword/i)
  })
})
