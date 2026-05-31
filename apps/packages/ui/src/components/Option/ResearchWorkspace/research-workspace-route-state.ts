export type ResearchWorkspaceTab = "sources" | "chat" | "studio"

export const RESEARCH_WORKSPACE_DEFAULT_TAB: ResearchWorkspaceTab = "chat"
export const RESEARCH_WORKSPACE_LAST_MOBILE_TAB_STORAGE_KEY =
  "tldw:research-workspace:last-mobile-tab:v1"
const DEEP_RESEARCH_RETURN_ID_MAX_LENGTH = 128
const DEEP_RESEARCH_RETURN_TITLE_MAX_LENGTH = 240

export type ResearchWorkspaceDeepResearchReturnContext = {
  sourceWorkspaceId: string
  sourceArtifactId: string | null
  sourceArtifactTemplate: string | null
  sourceArtifactTitle: string | null
  researchRunId: string
}

type ResearchWorkspaceTabStorage = Pick<Storage, "getItem" | "setItem">
type ResearchWorkspaceLocation = Pick<Location, "hash" | "search">

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

export const getResearchWorkspaceSearchFromLocation = (
  location: ResearchWorkspaceLocation | null | undefined
): string => {
  if (!location) return ""
  if (location.search) return location.search

  const hashQueryIndex = location.hash.indexOf("?")
  if (hashQueryIndex < 0) return ""

  const hashFragmentIndex = location.hash.indexOf("#", hashQueryIndex + 1)
  return hashFragmentIndex < 0
    ? location.hash.slice(hashQueryIndex)
    : location.hash.slice(hashQueryIndex, hashFragmentIndex)
}

export const getInitialResearchWorkspaceTab = (
  search: string | URLSearchParams | null | undefined,
  fallback: ResearchWorkspaceTab = RESEARCH_WORKSPACE_DEFAULT_TAB
): ResearchWorkspaceTab => {
  return getResearchWorkspaceTabFromSearch(search) ?? fallback
}

const readBoundedSearchParam = (
  params: URLSearchParams,
  key: string,
  maxLength: number
): string | null => {
  const value = params.get(key)?.trim()
  if (!value) return null
  return value.slice(0, maxLength)
}

export const getResearchWorkspaceDeepResearchReturnContext = (
  search: string | URLSearchParams | null | undefined
): ResearchWorkspaceDeepResearchReturnContext | null => {
  if (!search) return null
  const params =
    typeof search === "string" ? new URLSearchParams(search) : search

  const sourceWorkspaceId = readBoundedSearchParam(
    params,
    "source_workspace_id",
    DEEP_RESEARCH_RETURN_ID_MAX_LENGTH
  )
  const researchRunId = readBoundedSearchParam(
    params,
    "research_run_id",
    DEEP_RESEARCH_RETURN_ID_MAX_LENGTH
  )

  if (!sourceWorkspaceId || !researchRunId) return null

  return {
    sourceWorkspaceId,
    sourceArtifactId: readBoundedSearchParam(
      params,
      "source_artifact_id",
      DEEP_RESEARCH_RETURN_ID_MAX_LENGTH
    ),
    sourceArtifactTemplate: readBoundedSearchParam(
      params,
      "source_artifact_template",
      DEEP_RESEARCH_RETURN_ID_MAX_LENGTH
    ),
    sourceArtifactTitle: readBoundedSearchParam(
      params,
      "source_artifact_title",
      DEEP_RESEARCH_RETURN_TITLE_MAX_LENGTH
    ),
    researchRunId
  }
}

export const isResearchWorkspaceDeepResearchReturnForWorkspace = (
  context: ResearchWorkspaceDeepResearchReturnContext | null,
  workspaceId: string | null | undefined
): context is ResearchWorkspaceDeepResearchReturnContext => {
  return Boolean(context && workspaceId && context.sourceWorkspaceId === workspaceId)
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
