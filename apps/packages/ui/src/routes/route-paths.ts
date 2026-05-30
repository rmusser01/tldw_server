import {
  RESEARCH_RETURN_RUN_ID_PARAM,
  SETTINGS_SERVER_CHAT_ID_PARAM
} from "@/utils/settings-return"

export const CHAT_PATH = "/chat"
export const CHAT_WORKSPACE_PATH = "/chat-workspace"
export const RESEARCH_PATH = "/research"
export const PROTOTYPE_WORKSPACES_PATH = "/prototype-workspaces"
export const RESEARCH_WORKSPACE_PATH = "/research-workspace"
export const DOCUMENT_WORKSPACE_PATH = "/document-workspace"
export const MODERATION_REVIEW_PATH = "/moderation"
export const MODERATION_RULES_PATH = "/moderation/rules"
export const MODERATION_PLAYGROUND_LEGACY_PATH = "/moderation-playground"
export const PRESENTATION_STUDIO_PATH = "/presentation-studio"
export const PRESENTATION_STUDIO_NEW_PATH = "/presentation-studio/new"
export const PRESENTATION_STUDIO_DETAIL_PATH = "/presentation-studio/:projectId"
export const PRESENTATION_STUDIO_START_PATH = "/presentation-studio/start"
export const REPO2TXT_PATH = "/repo2txt"
export const SOURCES_PATH = "/sources"
export const SOURCES_NEW_PATH = "/sources/new"
export const SOURCES_DETAIL_PATH = "/sources/:sourceId"
export const ADMIN_SOURCES_PATH = "/admin/sources"
export const MEDIA_COLLECTIONS_PATH = "/media-collections"
export const MEDIA_COLLECTION_REVIEW_PATH = `${MEDIA_COLLECTIONS_PATH}/:collectionId`

export type SourcesNewPreset = "notes-folder-sync"

type BuildSourcesNewPathOptions = {
  preset?: SourcesNewPreset
}

export const VIEWPORT_CONSTRAINED_PATHS = [
  CHAT_PATH,
  CHAT_WORKSPACE_PATH,
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
  followUp?: Record<string, unknown> | null
}

type BuildChatThreadPathOptions = {
  serverChatId?: string | null
  researchReturnRunId?: string | null
}

type BuildCharacterChatPathOptions = {
  characterId?: string | number | null
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

const setJsonSearchParam = (
  params: URLSearchParams,
  key: string,
  value: Record<string, unknown> | null | undefined
) => {
  if (!value) {
    return
  }
  const serialized = JSON.stringify(value)
  if (serialized !== "{}") {
    params.set(key, serialized)
  }
}

export const buildSourcesNewPath = (
  options: BuildSourcesNewPathOptions = {}
): string => {
  const params = new URLSearchParams()
  if (options.preset) {
    params.set("preset", options.preset)
  }
  const encoded = params.toString()
  return encoded ? `${SOURCES_NEW_PATH}?${encoded}` : SOURCES_NEW_PATH
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
  setJsonSearchParam(params, "follow_up", options.followUp)
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

export const buildCharacterChatPath = (
  options: BuildCharacterChatPathOptions = {}
): string => {
  const params = new URLSearchParams({ mode: "character" })
  const characterId =
    options.characterId == null ? "" : String(options.characterId).trim()
  if (characterId) {
    params.set("characterId", characterId)
  }
  return `${CHAT_PATH}?${params.toString()}`
}

export const buildMediaCollectionReviewPath = (
  collectionId: string | number
): string => `${MEDIA_COLLECTIONS_PATH}/${encodeURIComponent(String(collectionId))}`
