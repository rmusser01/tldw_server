const SETTINGS_RETURN_TO_KEY = "tldw:settingsReturnTo"
export const SETTINGS_NAVIGATION_REQUEST_EVENT =
  "tldw:settings-navigation-request"
export const SETTINGS_HISTORY_ID_PARAM = "settingsHistoryId"
export const SETTINGS_SERVER_CHAT_ID_PARAM = "settingsServerChatId"
export const RESEARCH_RETURN_RUN_ID_PARAM = "researchReturnRunId"

const isSettingsPath = (path: string) => path.startsWith("/settings")
const SETTINGS_RETURN_URL_BASE = "https://tldw.local"

export type SettingsReturnChatContext = {
  historyId?: string | null
  serverChatId?: string | null
}

export type SettingsNavigationRequestDetail = {
  destination: string
}

export const requestSettingsNavigation = (destination: string): boolean => {
  if (typeof window === "undefined") return true
  return window.dispatchEvent(new CustomEvent<SettingsNavigationRequestDetail>(
    SETTINGS_NAVIGATION_REQUEST_EVENT,
    { cancelable: true, detail: { destination } }
  ))
}

export const resolveSettingsNavigationUrl = (
  destination: string,
  currentUrl: string
): string | null => {
  try {
    const current = new URL(currentUrl)
    const route = new URL(destination, current)
    if (route.protocol !== current.protocol || route.host !== current.host) {
      return null
    }
    if (!current.hash.startsWith("#/")) return route.href
    if (route.pathname === current.pathname && route.hash.startsWith("#/")) {
      return route.href
    }
    current.hash = `#${route.pathname}${route.search}${route.hash}`
    return current.href
  } catch {
    return null
  }
}

const toNormalizedId = (value?: string | null): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const addChatContextToPath = (
  path: string,
  chatContext?: SettingsReturnChatContext
): string => {
  if (!chatContext) return path

  try {
    const url = new URL(path, SETTINGS_RETURN_URL_BASE)
    if (url.pathname !== "/chat") {
      return path
    }

    const normalizedHistoryId = toNormalizedId(chatContext.historyId)
    const normalizedServerChatId = toNormalizedId(chatContext.serverChatId)

    if (normalizedHistoryId) {
      url.searchParams.set(SETTINGS_HISTORY_ID_PARAM, normalizedHistoryId)
    } else {
      url.searchParams.delete(SETTINGS_HISTORY_ID_PARAM)
    }

    if (normalizedServerChatId) {
      url.searchParams.set(SETTINGS_SERVER_CHAT_ID_PARAM, normalizedServerChatId)
    } else {
      url.searchParams.delete(SETTINGS_SERVER_CHAT_ID_PARAM)
    }

    const query = url.searchParams.toString()
    return `${url.pathname}${query ? `?${query}` : ""}${url.hash}`
  } catch {
    return path
  }
}

export const setSettingsReturnTo = (
  path: string,
  chatContext?: SettingsReturnChatContext
) => {
  try {
    if (typeof sessionStorage === "undefined") return
    if (!path || isSettingsPath(path)) return
    sessionStorage.setItem(
      SETTINGS_RETURN_TO_KEY,
      addChatContextToPath(path, chatContext)
    )
  } catch {
    // ignore storage errors
  }
}

export const getSettingsReturnTo = (): string | null => {
  try {
    if (typeof sessionStorage === "undefined") return null
    const raw = sessionStorage.getItem(SETTINGS_RETURN_TO_KEY)
    if (!raw || typeof raw !== "string") return null
    if (isSettingsPath(raw)) {
      sessionStorage.removeItem(SETTINGS_RETURN_TO_KEY)
      return null
    }
    return raw
  } catch {
    return null
  }
}

export const clearSettingsReturnTo = () => {
  try {
    if (typeof sessionStorage === "undefined") return
    sessionStorage.removeItem(SETTINGS_RETURN_TO_KEY)
  } catch {
    // ignore storage errors
  }
}
