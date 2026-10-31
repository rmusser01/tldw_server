import { themes } from "prism-react-renderer"
import type { Language, PrismTheme } from "prism-react-renderer"

export const normalizeLanguage = (language: string): string => {
  const lang = (language || "").toLowerCase()
  if (lang === "js" || lang === "jsx") return "javascript"
  if (lang === "ts" || lang === "tsx") return "typescript"
  if (lang === "sh" || lang === "bash") return "bash"
  if (lang === "py") return "python"
  if (lang === "text" || lang === "txt" || lang === "plain") return "plaintext"
  if (lang === "md" || lang === "markdown") return "markdown"
  if (lang === "yml") return "yaml"
  if (!lang) return "plaintext"
  return lang
}

export const safeLanguage = (language: string): Language =>
  normalizeLanguage(language) as Language

/**
 * Accessible light syntax theme. Every colour reaches 4.5:1 on its own
 * background and on the app's light surfaces, where Markdown code blocks are
 * drawn without a theme background (AX-07, #3124).
 */
export const a11yLightCodeTheme: PrismTheme = {
  plain: { color: "#1f2328", backgroundColor: "#f6f8fa" },
  styles: [
    { types: ["comment", "prolog", "doctype", "cdata"], style: { color: "#5f6877", fontStyle: "italic" } },
    { types: ["punctuation"], style: { color: "#4b5563" } },
    { types: ["keyword", "atrule", "deleted", "important"], style: { color: "#a40e26" } },
    { types: ["string", "char", "attr-value", "regex", "inserted", "tag", "selector"], style: { color: "#116329" } },
    { types: ["number", "boolean", "constant", "symbol", "property", "builtin"], style: { color: "#0550ae" } },
    { types: ["function", "class-name"], style: { color: "#6639ba" } },
    { types: ["operator", "entity", "url", "variable", "attr-name"], style: { color: "#953800" } },
    { types: ["important", "bold"], style: { fontWeight: "bold" } },
    { types: ["italic"], style: { fontStyle: "italic" } },
  ],
}

/**
 * Accessible dark syntax theme, after Eric Bailey's a11y-dark. Every colour
 * reaches 4.5:1 on its own background and on the app's dark surfaces.
 */
export const a11yDarkCodeTheme: PrismTheme = {
  plain: { color: "#f8f8f2", backgroundColor: "#2b2b2b" },
  styles: [
    { types: ["comment", "prolog", "doctype", "cdata"], style: { color: "#d4d0ab", fontStyle: "italic" } },
    { types: ["punctuation"], style: { color: "#fefefe" } },
    { types: ["property", "tag", "constant", "symbol", "deleted"], style: { color: "#ffa07a" } },
    { types: ["boolean", "number", "keyword", "operator", "entity", "url", "variable"], style: { color: "#00e0e0" } },
    { types: ["selector", "attr-name", "string", "char", "builtin", "inserted"], style: { color: "#abe338" } },
    { types: ["atrule", "attr-value", "function", "class-name", "regex", "important"], style: { color: "#ffd700" } },
    { types: ["important", "bold"], style: { fontWeight: "bold" } },
    { types: ["italic"], style: { fontStyle: "italic" } },
  ],
}

export const resolveTheme = (key: string): PrismTheme => {
  if (key === "auto") {
    let isDark = false
    try {
      if (typeof document !== "undefined") {
        const root = document.documentElement
        if (root.classList.contains("dark")) {
          isDark = true
        } else if (root.classList.contains("light")) {
          isDark = false
        } else if (typeof window !== "undefined") {
          isDark = window.matchMedia("(prefers-color-scheme: dark)").matches
        }
      }
    } catch {
      isDark = false
    }
    return isDark ? a11yDarkCodeTheme : a11yLightCodeTheme
  }
  switch (key) {
    case "github":
      return themes.github
    case "nightOwl":
      return themes.nightOwl
    case "nightOwlLight":
      return themes.nightOwlLight
    case "vsDark":
      return themes.vsDark
    case "dracula":
    default:
      return themes.dracula
  }
}
