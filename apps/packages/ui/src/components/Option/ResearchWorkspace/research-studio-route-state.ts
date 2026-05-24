export type ResearchStudioTab = "sources" | "chat" | "studio"

export const RESEARCH_STUDIO_DEFAULT_TAB: ResearchStudioTab = "chat"
export const RESEARCH_STUDIO_LAST_MOBILE_TAB_STORAGE_KEY =
  "tldw:research-studio:last-mobile-tab:v1"

type ResearchStudioTabStorage = Pick<Storage, "getItem" | "setItem">

const getBrowserStorage = (): ResearchStudioTabStorage | null => {
  if (typeof window === "undefined") return null
  try {
    return window.localStorage
  } catch {
    return null
  }
}

export const parseResearchStudioTab = (
  value: unknown
): ResearchStudioTab | null => {
  return value === "sources" || value === "chat" || value === "studio"
    ? value
    : null
}

export const getResearchStudioTabFromSearch = (
  search: string | URLSearchParams | null | undefined
): ResearchStudioTab | null => {
  if (!search) return null
  const params =
    typeof search === "string" ? new URLSearchParams(search) : search

  return parseResearchStudioTab(params.get("tab"))
}

export const getInitialResearchStudioTab = (
  search: string | URLSearchParams | null | undefined,
  fallback: ResearchStudioTab = RESEARCH_STUDIO_DEFAULT_TAB
): ResearchStudioTab => {
  return getResearchStudioTabFromSearch(search) ?? fallback
}

export const readResearchStudioLastMobileTab = (
  storage: ResearchStudioTabStorage | null = getBrowserStorage()
): ResearchStudioTab | null => {
  if (!storage) return null
  try {
    return parseResearchStudioTab(
      storage.getItem(RESEARCH_STUDIO_LAST_MOBILE_TAB_STORAGE_KEY)
    )
  } catch {
    return null
  }
}

export const writeResearchStudioLastMobileTab = (
  tab: ResearchStudioTab,
  storage: ResearchStudioTabStorage | null = getBrowserStorage()
): void => {
  if (!storage) return
  try {
    storage.setItem(RESEARCH_STUDIO_LAST_MOBILE_TAB_STORAGE_KEY, tab)
  } catch {
    // Storage can be unavailable or full; tab persistence is an enhancement.
  }
}
