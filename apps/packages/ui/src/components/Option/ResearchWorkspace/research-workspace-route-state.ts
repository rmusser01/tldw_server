export type ResearchWorkspaceTab = "sources" | "chat" | "studio"

export const RESEARCH_WORKSPACE_DEFAULT_TAB: ResearchWorkspaceTab = "chat"
export const RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY =
  "tldw:research-workspace:last-mobile-tab:v1"

type ResearchWorkspaceTabStorage = Pick<Storage, "getItem" | "setItem">

const getBrowserStorage = (): ResearchWorkspaceTabStorage | null => {
  if (typeof window === "undefined") return null
  try {
    return window.localStorage
  } catch {
    return null
  }
}

export const parseResearchWorkspaceTab = (
  value: unknown
): ResearchWorkspaceTab | null => {
  return value === "sources" || value === "chat" || value === "studio"
    ? value
    : null
}

export const getResearchWorkspaceTabFromSearch = (
  search: string | URLSearchParams | null | undefined
): ResearchWorkspaceTab | null => {
  if (!search) return null
  const params =
    typeof search === "string" ? new URLSearchParams(search) : search

  return parseResearchWorkspaceTab(params.get("tab"))
}

export const getInitialResearchWorkspaceTab = (
  search: string | URLSearchParams | null | undefined,
  fallback: ResearchWorkspaceTab = RESEARCH_WORKSPACE_DEFAULT_TAB
): ResearchWorkspaceTab => {
  return getResearchWorkspaceTabFromSearch(search) ?? fallback
}

export const readResearchWorkspaceLastMobileTab = (
  storage: ResearchWorkspaceTabStorage | null = getBrowserStorage()
): ResearchWorkspaceTab | null => {
  if (!storage) return null
  try {
    return parseResearchWorkspaceTab(
      storage.getItem(RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY)
    )
  } catch {
    return null
  }
}

export const writeResearchWorkspaceLastMobileTab = (
  tab: ResearchWorkspaceTab,
  storage: ResearchWorkspaceTabStorage | null = getBrowserStorage()
): void => {
  if (!storage) return
  try {
    storage.setItem(RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY, tab)
  } catch {
    // Storage can be unavailable or full; tab persistence is an enhancement.
  }
}
