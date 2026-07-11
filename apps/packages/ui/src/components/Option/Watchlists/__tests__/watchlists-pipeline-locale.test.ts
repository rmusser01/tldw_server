import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import i18next from "i18next"

type NestedLocale = Record<string, unknown>
type ExtensionLocale = Record<string, { message?: unknown }>

const testDir = path.dirname(fileURLToPath(import.meta.url))
const srcRoot = path.resolve(testDir, "../../../../")

const canonicalLocales = [
  "ar",
  "da",
  "de",
  "es",
  "fa",
  "fr",
  "it",
  "ja-JP",
  "ko",
  "ml",
  "no",
  "pt-BR",
  "ru",
  "sv",
  "uk",
  "zh",
  "zh-TW"
] as const

const publicAliases = {
  ja: "ja-JP",
  zh_CN: "zh",
  zh_TW: "zh-TW"
} as const

const readNested = (locale: string): NestedLocale =>
  JSON.parse(
    readFileSync(
      path.resolve(srcRoot, `assets/locale/${locale}/watchlists.json`),
      "utf8"
    )
  ) as NestedLocale

const readPublic = (locale: string): ExtensionLocale =>
  JSON.parse(
    readFileSync(
      path.resolve(srcRoot, `public/_locales/${locale}/watchlists.json`),
      "utf8"
    )
  ) as ExtensionLocale

const flatten = (
  value: unknown,
  prefix: string[] = []
): Record<string, string> => {
  if (typeof value === "string") {
    return { [prefix.join("_")]: value }
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {}
  }

  return Object.entries(value as Record<string, unknown>).reduce(
    (result, [key, nested]) => ({
      ...result,
      ...flatten(nested, [...prefix, key])
    }),
    {} as Record<string, string>
  )
}

const setupCopy = (locale: NestedLocale) => ({
  ...flatten(locale.setupWizard, ["setupWizard"]),
  ...flatten(
    (locale.overview as Record<string, unknown> | undefined)?.pipelineSetup,
    ["overview", "pipelineSetup"]
  )
})

const placeholders = (value: string) =>
  [...value.matchAll(/{{\s*([^}]+?)\s*}}/g)]
    .map((match) => match[1])
    .sort()

describe("Watchlists pipeline locale contract", () => {
  const english = setupCopy(readNested("en"))
  const englishKeys = Object.keys(english).sort()

  it.each(canonicalLocales)(
    "%s ships the complete translated setup contract",
    (locale) => {
      const translated = setupCopy(readNested(locale))

      expect(englishKeys.filter((key) => !(key in translated))).toEqual([])
      expect(
        Object.keys(translated)
          .filter((key) => !englishKeys.includes(key))
          .every((key) => /^overview_pipelineSetup_receipt_(sources|cast|duration)_(zero|two|few|many)$/.test(key))
      ).toBe(true)
      expect(englishKeys.length).toBeGreaterThan(100)
      expect(
        englishKeys.filter((key) => translated[key] !== english[key]).length
      ).toBeGreaterThan(englishKeys.length * 0.8)

      for (const key of englishKeys) {
        expect(translated[key]?.trim(), `${locale}:${key}`).toBeTruthy()
        expect(placeholders(translated[key]), `${locale}:${key}`).toEqual(
          placeholders(english[key])
        )
      }
    }
  )

  it.each(["en", ...canonicalLocales])(
    "%s keeps nested and extension setup copy identical",
    (locale) => {
      const nested = setupCopy(readNested(locale))
      const extension = readPublic(locale)

      for (const [key, value] of Object.entries(nested)) {
        expect(extension[key]?.message, `${locale}:${key}`).toBe(value)
      }
    }
  )

  it.each(Object.entries(publicAliases))(
    "%s mirrors its canonical public locale",
    (alias, canonical) => {
      const aliasMessages = readPublic(alias)
      const canonicalMessages = readPublic(canonical)
      const setupKeys = Object.keys(setupCopy(readNested(canonical)))

      for (const key of setupKeys) {
        expect(aliasMessages[key]?.message, `${alias}:${key}`).toBe(
          canonicalMessages[key]?.message
        )
      }
    }
  )

  it.each(["ar", "fa"])("%s keeps key guidance native and RTL-ready", (locale) => {
    const translated = setupCopy(readNested(locale))
    const guidance = [
      translated.overview_pipelineSetup_sources_help,
      translated.overview_pipelineSetup_test_providerDisclosure,
      translated.overview_pipelineSetup_delivery_help
    ].join(" ")

    expect(guidance).toMatch(/[\u0600-\u06ff]/)
    expect(guidance.length).toBeGreaterThan(120)
  })

  it.each(["en", ...canonicalLocales])(
    "%s ships receipt plural keys for every supported category",
    (locale) => {
      const nested = setupCopy(readNested(locale))
      const categories = new Set([
        "one",
        "other",
        ...new Intl.PluralRules(locale).resolvedOptions().pluralCategories
      ])

      for (const noun of ["sources", "cast", "duration"]) {
        for (const category of categories) {
          const key = `overview_pipelineSetup_receipt_${noun}_${category}`
          expect(nested[key], `${locale}:${key}`).toBeTruthy()
          expect(placeholders(nested[key]), `${locale}:${key}`).toEqual(["count"])
        }
      }
    }
  )

  it("renders singular and plural English plus Russian multi-category grammar", async () => {
    const instance = i18next.createInstance()
    await instance.init({
      lng: "en",
      fallbackLng: false,
      ns: ["watchlists"],
      defaultNS: "watchlists",
      resources: {
        en: { watchlists: readNested("en") },
        ru: { watchlists: readNested("ru") }
      },
      interpolation: { escapeValue: false }
    })

    expect(instance.t("overview.pipelineSetup.receipt.sources", { count: 1 })).toBe("1 source")
    expect(instance.t("overview.pipelineSetup.receipt.sources", { count: 2 })).toBe("2 sources")
    await instance.changeLanguage("ru")
    expect(instance.t("overview.pipelineSetup.receipt.sources", { count: 1 })).toBe("1 источник")
    expect(instance.t("overview.pipelineSetup.receipt.sources", { count: 2 })).toBe("2 источника")
    expect(instance.t("overview.pipelineSetup.receipt.sources", { count: 5 })).toBe("5 источников")
  })

  it("ships localized terminal outcomes and exact-schedule feedback outside English", () => {
    const english = setupCopy(readNested("en"))

    for (const locale of ["es", "ja-JP", "ru"] as const) {
      const translated = setupCopy(readNested(locale))
      for (const key of [
        "overview_pipelineSetup_test_ready",
        "overview_pipelineSetup_test_running",
        "overview_pipelineSetup_test_failedWithCode",
        "overview_pipelineSetup_test_cancelledWithCode",
        "overview_pipelineSetup_schedulePreview_loading",
        "overview_pipelineSetup_schedulePreview_error"
      ]) {
        expect(translated[key], `${locale}:${key}`).toBeTruthy()
        expect(translated[key], `${locale}:${key}`).not.toBe(english[key])
        expect(placeholders(translated[key]), `${locale}:${key}`).toEqual(
          placeholders(english[key])
        )
      }
    }
  })
})
