import {
  RESEARCH_RETURN_RUN_ID_PARAM,
  SETTINGS_SERVER_CHAT_ID_PARAM
} from "@/utils/settings-return"

export const CHAT_PATH = "/chat"
export const RESEARCH_PATH = "/research"
export const RESEARCH_WORKSPACE_PATH = "/research-workspace"
export const DOCUMENT_WORKSPACE_PATH = "/document-workspace"
export const PRESENTATION_STUDIO_PATH = "/presentation-studio"
export const PRESENTATION_STUDIO_NEW_PATH = "/presentation-studio/new"
export const PRESENTATION_STUDIO_DETAIL_PATH = "/presentation-studio/:projectId"
export const PRESENTATION_STUDIO_START_PATH = "/presentation-studio/start"
export const REPO2TXT_PATH = "/repo2txt"
export const SOURCES_PATH = "/sources"
export const SOURCES_NEW_PATH = "/sources/new"
export const SOURCES_DETAIL_PATH = "/sources/:sourceId"
export const ADMIN_SOURCES_PATH = "/admin/sources"

export const VIEWPORT_CONSTRAINED_PATHS = [
  DOCUMENT_WORKSPACE_PATH,
  RESEARCH_WORKSPACE_PATH,
  "/media-multi",
] as const

export const LOREBOOK_DEBUG_FOCUS = "lorebook-debug"

type BuildChatLorebookDebugPathOptions = {
  from?: string | null
}

export const buildChatLorebookDebugPath = (
  options: BuildChatLorebookDebugPathOptions = {}
): string => {
  const params = new URLSearchParams({
    focus: LOREBOOK_DEBUG_FOCUS
  })
  const from = options.from?.trim()
  if (from) {
    params.set("from", from)
  }
  return `${CHAT_PATH}?${params.toString()}`
}

type BuildResearchLaunchPathOptions = {
  query?: string | null
  sourcePolicy?: string | null
  autonomyMode?: string | null
  autorun?: boolean
  from?: string | null
  run?: string | null
  chatId?: string | null
  launchMessageId?: string | null
}

type BuildChatThreadPathOptions = {
  serverChatId?: string | null
  researchReturnRunId?: string | null
}

const setTrimmedSearchParam = (
  params: URLSearchParams,
  key: string,
  value: string | null | undefined
) => {
  const trimmed = value?.trim()
  if (trimmed) {
    params.set(key, trimmed)
  }
}

export const buildResearchLaunchPath = (
  options: BuildResearchLaunchPathOptions = {}
): string => {
  const params = new URLSearchParams()
  setTrimmedSearchParam(params, "query", options.query)
  setTrimmedSearchParam(params, "source_policy", options.sourcePolicy)
  setTrimmedSearchParam(params, "autonomy_mode", options.autonomyMode)
  setTrimmedSearchParam(params, "from", options.from)
  setTrimmedSearchParam(params, "run", options.run)
  setTrimmedSearchParam(params, "chat_id", options.chatId)
  setTrimmedSearchParam(params, "launch_message_id", options.launchMessageId)
  if (options.autorun) {
    params.set("autorun", "1")
  }
  const encoded = params.toString()
  return encoded ? `${RESEARCH_PATH}?${encoded}` : RESEARCH_PATH
}

export const buildChatThreadPath = (
  options: BuildChatThreadPathOptions = {}
): string => {
  const params = new URLSearchParams()
  setTrimmedSearchParam(
    params,
    SETTINGS_SERVER_CHAT_ID_PARAM,
    options.serverChatId
  )
  setTrimmedSearchParam(
    params,
    RESEARCH_RETURN_RUN_ID_PARAM,
    options.researchReturnRunId
  )
  const encoded = params.toString()
  return encoded ? `${CHAT_PATH}?${encoded}` : CHAT_PATH
}
