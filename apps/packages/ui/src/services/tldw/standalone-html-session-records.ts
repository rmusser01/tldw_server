const STANDALONE_HTML_SESSION_PREFIXES = [
  "tldw:presentation-studio:html:draft:v1:",
  "tldw:presentation-studio:html:resume:v1:"
] as const

export const clearStandaloneHtmlSessionRecords = (): void => {
  if (typeof window === "undefined") return
  try {
    for (let index = window.sessionStorage.length - 1; index >= 0; index -= 1) {
      const key = window.sessionStorage.key(index)
      if (key && STANDALONE_HTML_SESSION_PREFIXES.some((prefix) => key.startsWith(prefix))) {
        window.sessionStorage.removeItem(key)
      }
    }
  } catch {
    // Storage may be unavailable; authentication cleanup still completes.
  }
}
