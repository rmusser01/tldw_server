import { readFileSync, readdirSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = dirname(fileURLToPath(import.meta.url))
const localeRoot = resolve(testDir, "..", "..", "assets", "locale")
const publicLocaleRoot = resolve(testDir, "..", "..", "public", "_locales")
const i18nSource = readFileSync(resolve(testDir, "..", "index.ts"), "utf8")

const readNotesSearch = (locale: string): Record<string, unknown> => {
  const payload = JSON.parse(
    readFileSync(resolve(localeRoot, locale, "option.json"), "utf8")
  ) as { notesSearch?: Record<string, unknown> }
  return payload.notesSearch ?? {}
}

const semanticKeys = (notesSearch: Record<string, unknown>): string[] =>
  Object.keys(notesSearch).filter(
    (key) => key === "graphSimilarContent" || key.startsWith("semantic")
  )

describe("Notes semantic locale fallback policy", () => {
  it("keeps complete English copy and delegates every asset locale to fallbackLng=en", () => {
    const english = readNotesSearch("en")
    expect(i18nSource).toMatch(/fallbackLng:\s*["']en["']/)
    expect(semanticKeys(english)).toEqual(
      expect.arrayContaining([
        "graphSimilarContent",
        "semanticIndex",
        "semanticRenewConsent",
        "semanticRenewStarted",
        "semanticConfirm",
        "semanticDetail"
      ])
    )

    const locales = readdirSync(localeRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && entry.name !== "en")
      .map((entry) => entry.name)
    expect(locales.length).toBeGreaterThan(0)
    for (const locale of locales) {
      expect(semanticKeys(readNotesSearch(locale)), locale).toEqual([])
    }

    for (const entry of readdirSync(publicLocaleRoot, {
      withFileTypes: true
    })) {
      if (!entry.isDirectory() || entry.name === "en") continue
      const payload = JSON.parse(
        readFileSync(
          resolve(publicLocaleRoot, entry.name, "option.json"),
          "utf8"
        )
      ) as Record<string, unknown>
      const publicSemanticKeys = Object.keys(payload).filter(
        (key) =>
          key === "notesSearch_graphSimilarContent" ||
          key.startsWith("notesSearch_semantic")
      )
      expect(publicSemanticKeys, entry.name).toEqual([])
    }
  })
})
