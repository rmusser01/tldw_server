import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import i18next from "i18next"
import { describe, expect, it } from "vitest"
import { formatWatchlistOccurrenceDate } from "../OverviewTab/LatestBriefing"

type NestedLocale = Record<string, unknown>
type ExtensionLocale = Record<string, { message?: unknown }>

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../../../")
const canonicalLocales = [
  "ar", "da", "de", "es", "fa", "fr", "it", "ja-JP", "ko", "ml",
  "no", "pt-BR", "ru", "sv", "uk", "zh", "zh-TW"
] as const
const publicAliases = { ja: "ja-JP", zh_CN: "zh", zh_TW: "zh-TW" } as const

const readNested = (locale: string): NestedLocale => JSON.parse(readFileSync(
  path.resolve(srcRoot, `assets/locale/${locale}/watchlists.json`),
  "utf8"
)) as NestedLocale
const readPublic = (locale: string): ExtensionLocale => JSON.parse(readFileSync(
  path.resolve(srcRoot, `public/_locales/${locale}/watchlists.json`),
  "utf8"
)) as ExtensionLocale

const flatten = (value: unknown, prefix: string[] = []): Record<string, string> => {
  if (typeof value === "string") return { [prefix.join("_")]: value }
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  return Object.entries(value as Record<string, unknown>).reduce((all, [key, nested]) => ({
    ...all,
    ...flatten(nested, [...prefix, key])
  }), {})
}

const latestCopy = (locale: NestedLocale): Record<string, string> => flatten(
  (locale.overview as Record<string, unknown> | undefined)?.latest,
  ["overview", "latest"]
)
const placeholders = (value: string) => [...value.matchAll(/{{\s*([^}]+?)\s*}}/g)]
  .map((match) => match[1])
  .sort()

describe("Watchlists Latest briefing locale contract", () => {
  const english = latestCopy(readNested("en"))
  const englishKeys = Object.keys(english).sort()

  it.each(canonicalLocales)("%s ships the complete translated Latest contract", (locale) => {
    const translated = latestCopy(readNested(locale))
    expect(englishKeys.length).toBeGreaterThan(50)
    expect(Object.keys(translated).sort()).toEqual(englishKeys)
    expect(englishKeys.filter((key) => translated[key] !== english[key]).length)
      .toBeGreaterThan(englishKeys.length * 0.8)
    for (const key of englishKeys) {
      expect(translated[key]?.trim(), `${locale}:${key}`).toBeTruthy()
      expect(placeholders(translated[key]), `${locale}:${key}`).toEqual(placeholders(english[key]))
    }
  })

  it.each(canonicalLocales)("%s mirrors Latest copy into the extension locale", (locale) => {
    const nested = latestCopy(readNested(locale))
    const extension = readPublic(locale)
    for (const [key, value] of Object.entries(nested)) {
      expect(extension[key]?.message, `${locale}:${key}`).toBe(value)
    }
  })

  it.each(Object.entries(publicAliases))("%s mirrors its canonical Latest locale", (alias, canonical) => {
    const aliasMessages = readPublic(alias)
    const canonicalMessages = readPublic(canonical)
    for (const key of Object.keys(latestCopy(readNested(canonical)))) {
      expect(aliasMessages[key]?.message, `${alias}:${key}`).toBe(canonicalMessages[key]?.message)
    }
  })

  it("formats an exact occurrence in the active locale and authoritative timezone", () => {
    expect(formatWatchlistOccurrenceDate(
      "2026-07-12T18:00:00-07:00",
      "America/Los_Angeles",
      "es"
    )).toMatch(/domingo.*12.*julio.*18:00/i)
  })

  it("uses active-locale plural categories for provenance counts", async () => {
    const instance = i18next.createInstance()
    await instance.init({
      lng: "ru",
      fallbackLng: false,
      resources: { ru: { watchlists: readNested("ru") } },
      ns: ["watchlists"],
      defaultNS: "watchlists",
      interpolation: { escapeValue: false }
    })
    expect(instance.t("overview.latest.provenance.sources", { count: 1 })).toContain("1")
    expect(instance.t("overview.latest.provenance.sources", { count: 2 }))
      .not.toBe(instance.t("overview.latest.provenance.sources", { count: 5 }))
  })
})
