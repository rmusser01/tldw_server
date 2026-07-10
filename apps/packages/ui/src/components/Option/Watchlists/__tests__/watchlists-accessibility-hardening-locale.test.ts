import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

type NestedLocale = Record<string, unknown>
type PublicLocale = Record<string, { message?: unknown }>

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../../../")
const canonicalLocales = [
  "ar", "da", "de", "es", "fa", "fr", "it", "ja-JP", "ko", "ml",
  "no", "pt-BR", "ru", "sv", "uk", "zh", "zh-TW"
] as const
const publicLocales = ["en", ...canonicalLocales] as const
const publicAliases = { ja: "ja-JP", zh_CN: "zh", zh_TW: "zh-TW" } as const

const readNested = (locale: string): NestedLocale => JSON.parse(
  readFileSync(path.resolve(srcRoot, `assets/locale/${locale}/watchlists.json`), "utf8")
) as NestedLocale

const readPublic = (locale: string): PublicLocale => JSON.parse(
  readFileSync(path.resolve(srcRoot, `public/_locales/${locale}/watchlists.json`), "utf8")
) as PublicLocale

const flatten = (value: unknown, prefix: string[] = []): Record<string, string> => {
  if (typeof value === "string") return { [prefix.join("_")]: value }
  if (!value || typeof value !== "object" || Array.isArray(value)) return {}
  return Object.entries(value as Record<string, unknown>).reduce(
    (result, [key, nested]) => ({ ...result, ...flatten(nested, [...prefix, key]) }),
    {} as Record<string, string>
  )
}

const taskCopy = (locale: NestedLocale) => flatten(
  locale.accessibilityHardening,
  ["accessibilityHardening"]
)

const placeholders = (value: string) => [...value.matchAll(/{{\s*([^}]+?)\s*}}/g)]
  .map((match) => match[1])
  .sort()

describe("Watchlists accessibility hardening locale contract", () => {
  const english = taskCopy(readNested("en"))
  const requiredKeys = Object.keys(english).sort()

  it("defines the focused Task 9 contract", () => {
    expect(requiredKeys).toHaveLength(21)
  })

  it.each(canonicalLocales)("%s translates every Task 9 key with placeholder parity", (locale) => {
    const translated = taskCopy(readNested(locale))
    expect(Object.keys(translated).sort()).toEqual(requiredKeys)
    expect(requiredKeys.filter((key) => translated[key] !== english[key]).length)
      .toBeGreaterThanOrEqual(20)
    for (const key of requiredKeys) {
      expect(translated[key]?.trim(), `${locale}:${key}`).toBeTruthy()
      expect(placeholders(translated[key]), `${locale}:${key}`).toEqual(placeholders(english[key]))
    }
  })

  it.each(publicLocales)("%s public locale mirrors its canonical Task 9 copy", (locale) => {
    const nested = taskCopy(readNested(locale))
    const publicCopy = readPublic(locale)
    for (const [key, value] of Object.entries(nested)) {
      expect(publicCopy[key]?.message, `${locale}:${key}`).toBe(value)
    }
  })

  it.each(Object.entries(publicAliases))("%s public alias mirrors %s Task 9 copy", (alias, canonical) => {
    const nested = taskCopy(readNested(canonical))
    const publicCopy = readPublic(alias)
    for (const [key, value] of Object.entries(nested)) {
      expect(publicCopy[key]?.message, `${alias}:${key}`).toBe(value)
    }
  })
})
