/**
 * Web-specific i18n initialization for Next.js
 *
 * This file solves SSR/static generation issues by:
 * 1. Statically importing English translations (no import.meta.glob)
 * 2. Initializing i18n synchronously before React renders
 * 3. Providing SSR-safe defaults
 */
import i18n from "i18next"
import ICU from "@tldw/ui/i18n/icu-format"
import { initReactI18next } from "react-i18next"
import { browser } from "wxt/browser"

// Static imports for English locale - these get bundled at build time
import commonEn from "@tldw/ui/assets/locale/en/common.json"
import sidepanelEn from "@tldw/ui/assets/locale/en/sidepanel.json"
import settingsEn from "@tldw/ui/assets/locale/en/settings.json"
import playgroundEn from "@tldw/ui/assets/locale/en/playground.json"
import knowledgeEn from "@tldw/ui/assets/locale/en/knowledge.json"
import optionEn from "@tldw/ui/assets/locale/en/option.json"
import reviewEn from "@tldw/ui/assets/locale/en/review.json"
import dataTablesEn from "@tldw/ui/assets/locale/en/dataTables.json"
import collectionsEn from "@tldw/ui/assets/locale/en/collections.json"
import evaluationsEn from "@tldw/ui/assets/locale/en/evaluations.json"
import audiobookEn from "@tldw/ui/assets/locale/en/audiobook.json"
import watchlistsEn from "@tldw/ui/assets/locale/en/watchlists.json"
import workflowsEn from "@tldw/ui/assets/locale/en/workflows.json"

const isMacPlatform =
  typeof navigator !== "undefined" &&
  /Mac|iPod|iPhone|iPad/.test(navigator.platform)
const commandPaletteShortcut = isMacPlatform ? "Cmd+K" : "Ctrl+K"

const NAMESPACES = [
  "option",
  "playground",
  "common",
  "sidepanel",
  "settings",
  "knowledge",
  "review",
  "dataTables",
  "collections",
  "evaluations",
  "audiobook",
  "watchlists",
  "workflows",
] as const

type Namespace = (typeof NAMESPACES)[number]

const LANGUAGE_ALIASES: Record<string, string> = {
  en: "en",
  es: "es",
  fr: "fr",
  it: "it",
  uk: "uk",
  "uk-UA": "uk",
  ru: "ru",
  "ru-RU": "ru",
  "pt-BR": "pt-BR",
  ml: "ml",
  "zh-CN": "zh",
  zh: "zh",
  "zh-TW": "zh-TW",
  ja: "ja-JP",
  "ja-JP": "ja-JP",
  fa: "fa",
  "fa-IR": "fa",
  da: "da",
  no: "no",
  sv: "sv",
  ko: "ko",
  ar: "ar",
  de: "de",
}

const RTL_LANGUAGES = ["ar", "fa", "he"]

const normalizeLanguage = (lng: string): string => {
  if (!lng) return "en"
  const trimmed = lng.replace("_", "-").trim()
  if (LANGUAGE_ALIASES[trimmed]) return LANGUAGE_ALIASES[trimmed]
  const base = trimmed.split("-")[0]
  if (LANGUAGE_ALIASES[base]) return LANGUAGE_ALIASES[base]
  return "en"
}

// Pre-bundled English resources
const englishResources: Record<Namespace, object> = {
  common: commonEn,
  sidepanel: sidepanelEn,
  settings: settingsEn,
  playground: playgroundEn,
  knowledge: knowledgeEn,
  option: optionEn,
  review: reviewEn,
  dataTables: dataTablesEn,
  collections: collectionsEn,
  evaluations: evaluationsEn,
  audiobook: audiobookEn,
  watchlists: watchlistsEn,
  workflows: workflowsEn,
}

// Build initial resources with English translations
const resources: Record<string, Record<string, object>> = {
  en: englishResources,
}

// Initialize i18n synchronously
i18n
  .use(ICU)
  .use(initReactI18next)
  .init({
    resources,
    fallbackLng: "en",
    lng: "en",
    ns: [...NAMESPACES],
    defaultNS: "common",
    interpolation: {
      escapeValue: false,
      defaultVariables: {
        shortcut: commandPaletteShortcut,
      },
    },
    // Disable async loading - we have pre-bundled resources
    initImmediate: true,
    react: {
      useSuspense: false,
    },
  })

// SSR-safe dir() function
i18n.dir = (lng?: string): "ltr" | "rtl" => {
  const language = lng || i18n.language || "en"
  const normalized = normalizeLanguage(language)
  const base = normalized.split("-")[0]

  if (RTL_LANGUAGES.includes(base)) {
    return "rtl"
  }
  return "ltr"
}

// Lazy load other languages on demand
const loadedLanguages = new Set<string>(["en"])

export const ensureLanguageLoaded = async (lng: string): Promise<void> => {
  const normalized = normalizeLanguage(lng)
  if (loadedLanguages.has(normalized) || normalized === "en") {
    return
  }

  // Dynamic imports for non-English languages
  try {
    const modules = await Promise.all([
      import(`@tldw/ui/assets/locale/${normalized}/common.json`).catch(
        () => null
      ),
      import(`@tldw/ui/assets/locale/${normalized}/sidepanel.json`).catch(
        () => null
      ),
      import(`@tldw/ui/assets/locale/${normalized}/settings.json`).catch(
        () => null
      ),
      import(`@tldw/ui/assets/locale/${normalized}/playground.json`).catch(
        () => null
      ),
      import(`@tldw/ui/assets/locale/${normalized}/knowledge.json`).catch(
        () => null
      ),
      import(`@tldw/ui/assets/locale/${normalized}/workflows.json`).catch(
        () => null
      ),
    ])

    const namespaces: Namespace[] = [
      "common",
      "sidepanel",
      "settings",
      "playground",
      "knowledge",
      "workflows",
    ]
    modules.forEach((mod, idx) => {
      if (mod) {
        const data = "default" in mod ? mod.default : mod
        i18n.addResourceBundle(normalized, namespaces[idx], data, true, true)
      }
    })

    loadedLanguages.add(normalized)
  } catch {
    // Fallback to English if language files don't exist
    console.warn(`Failed to load translations for ${normalized}`)
  }
}

async function syncStoredLanguage(): Promise<void> {
  try {
    const stored = await browser.storage.local.get("i18nextLng")
    const storedRecord =
      stored && typeof stored === "object" && !Array.isArray(stored)
        ? (stored as Record<string, unknown>)
        : null
    const rawLanguage = storedRecord?.i18nextLng
    const normalized = normalizeLanguage(
      typeof rawLanguage === "string" ? rawLanguage : "en"
    )
    if (normalized === normalizeLanguage(i18n.language || "en")) {
      return
    }
    await ensureLanguageLoaded(normalized)
    await i18n.changeLanguage(normalized)
  } catch (error) {
    console.warn("Failed to load stored language from browser storage.", error)
  }
}

void syncStoredLanguage()

// Listen for language changes
i18n.on("languageChanged", (lng) => {
  void ensureLanguageLoaded(lng)
})

export default i18n
