import i18n from "i18next"
import ICU from "./icu-format"
import { initReactI18next } from "react-i18next"

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
    "sources",
    "evaluations",
    "audiobook",
    "watchlists",
    "tutorials"
] as const

export type Namespace = typeof NAMESPACES[number]

const BASE_NAMESPACES: Namespace[] = ["common"]

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
    de: "de"
}

const getStoredLanguage = () => {
    if (typeof localStorage === "undefined") return "en"
    return localStorage.getItem("i18nextLng") || "en"
}

const normalizeLanguage = (lng: string) => {
    if (!lng) return "en"
    const trimmed = lng.replace("_", "-").trim()
    if (LANGUAGE_ALIASES[trimmed]) return LANGUAGE_ALIASES[trimmed]
    const base = trimmed.split("-")[0]
    if (LANGUAGE_ALIASES[base]) return LANGUAGE_ALIASES[base]
    return "en"
}

// Vite replaces this literal call with the packaged locale loader map.
const localeModules: Record<string, () => Promise<unknown>> =
    import.meta.glob("../assets/locale/*/*.json")
const loadingNamespaces = new Map<string, Promise<void>>()

const resolveLocalePath = (lng: string, ns: Namespace) =>
    `../assets/locale/${lng}/${ns}.json`

const loadNamespaceResource = async (lng: string, ns: Namespace) => {
    const normalized = normalizeLanguage(lng)
    if (i18n.hasResourceBundle(normalized, ns)) {
        if (lng !== normalized) {
            if (!i18n.hasResourceBundle(lng, ns)) {
                const existing = i18n.getResourceBundle(normalized, ns)
                if (existing) {
                    i18n.addResourceBundle(lng, ns, existing, true, true)
                }
            }
        }
        return
    }
    const cacheKey = `${normalized}:${ns}`
    const existing = loadingNamespaces.get(cacheKey)
    if (existing) {
        await existing
        return
    }
    const path = resolveLocalePath(normalized, ns)
    const loader = localeModules[path]
    if (!loader) {
        return
    }
    const loadPromise = loader()
        .then((mod: unknown) => {
            const data =
                typeof mod === "object" && mod && "default" in mod
                    ? (mod as { default: Record<string, unknown> }).default
                    : (mod as Record<string, unknown>)
            i18n.addResourceBundle(normalized, ns, data, true, true)
            if (lng !== normalized) {
                i18n.addResourceBundle(lng, ns, data, true, true)
            }
        })
        .finally(() => {
            loadingNamespaces.delete(cacheKey)
        })
    loadingNamespaces.set(cacheKey, loadPromise)
    await loadPromise
}

export const ensureI18nNamespaces = async (
    namespaces: Namespace[],
    lng: string = i18n.language || "en"
) => {
    const normalized = normalizeLanguage(lng)
    const tasks = namespaces.map((ns) => loadNamespaceResource(normalized, ns))
    await Promise.all(tasks)
}

i18n
    .use(ICU)
    .use(initReactI18next)
    .init({
        fallbackLng: "en",
        lng: normalizeLanguage(getStoredLanguage()),
        ns: [...NAMESPACES],
        defaultNS: "common",
        // React already escapes; avoid double-escaping (e.g., http:// -> http:\/\/)
        interpolation: {
            escapeValue: false,
            defaultVariables: {
                shortcut: commandPaletteShortcut
            }
        }
    })

const initialLanguage = normalizeLanguage(getStoredLanguage())
void ensureI18nNamespaces(BASE_NAMESPACES, "en")
void ensureI18nNamespaces(BASE_NAMESPACES, initialLanguage)

i18n.on("languageChanged", (lng) => {
    void ensureI18nNamespaces(BASE_NAMESPACES, lng)
})

export default i18n;
