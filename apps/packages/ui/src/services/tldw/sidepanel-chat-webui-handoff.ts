import { SETTINGS_SERVER_CHAT_ID_PARAM } from "@/utils/settings-return"
import type { ToolChoice } from "@/store/option"

export const SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM = "handoff"
export const SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE = "sidepanel-chat"
const DEFAULT_WEBUI_URL = "http://127.0.0.1:8080"

export type SidepanelChatWebUiConfig = {
  serverUrl?: string | null
  webUiUrl?: string | null
  webuiUrl?: string | null
}

export type SidepanelChatWebUiHandoffPayload = {
  source: typeof SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE
  createdAt: number
  draft?: string
  historyId?: string | null
  serverChatId?: string | null
  chatMode?: "normal" | "rag" | "vision"
  webSearch?: boolean
  toolChoice?: ToolChoice
  selectedModel?: string | null
  selectedSystemPrompt?: string | null
  selectedQuickPrompt?: string | null
  temporaryChat?: boolean
  useOCR?: boolean
  title?: string | null
}

const normalizeOptionalString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const isChatMode = (
  value: unknown
): value is SidepanelChatWebUiHandoffPayload["chatMode"] =>
  value === "normal" || value === "rag" || value === "vision"

const isToolChoice = (
  value: unknown
): value is SidepanelChatWebUiHandoffPayload["toolChoice"] =>
  value === "auto" || value === "none" || value === "required"

const trimTrailingSlash = (value: string) => value.replace(/\/+$/, "")

const normalizeBaseUrl = (value: string | null | undefined): string | null => {
  const trimmed = normalizeOptionalString(value)
  if (!trimmed) return null
  try {
    const url = new URL(trimmed)
    return trimTrailingSlash(url.origin)
  } catch {
    return null
  }
}

export const resolveSidepanelChatWebUiBaseUrl = (
  config: SidepanelChatWebUiConfig = {}
) => {
  const explicitWebUiUrl =
    normalizeBaseUrl(config.webUiUrl) ?? normalizeBaseUrl(config.webuiUrl)
  if (explicitWebUiUrl) return explicitWebUiUrl

  const serverUrl = normalizeOptionalString(config.serverUrl)
  if (serverUrl) {
    try {
      const url = new URL(serverUrl)
      if (url.port === "8000") {
        url.port = "8080"
      }
      return trimTrailingSlash(url.origin)
    } catch {
      // Fall through to the browser origin/default below.
    }
  }

  if (typeof window !== "undefined" && window.location?.origin) {
    const origin = window.location.origin
    if (origin.startsWith("http://") || origin.startsWith("https://")) {
      return trimTrailingSlash(origin)
    }
  }

  return DEFAULT_WEBUI_URL
}

const encodeBase64Url = (value: string) => {
  const bytes = new TextEncoder().encode(value)
  let binary = ""
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte)
  })
  return btoa(binary)
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "")
}

const decodeBase64Url = (value: string) => {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/")
  const padded =
    normalized.length % 4 === 0
      ? normalized
      : `${normalized}${"=".repeat(4 - (normalized.length % 4))}`
  const binary = atob(padded)
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
  return new TextDecoder().decode(bytes)
}

export const encodeSidepanelChatWebUiHandoff = (
  payload: SidepanelChatWebUiHandoffPayload
) => encodeBase64Url(JSON.stringify(payload))

export const decodeSidepanelChatWebUiHandoff = (
  value: string | null | undefined
): SidepanelChatWebUiHandoffPayload | null => {
  const encoded = normalizeOptionalString(value)
  if (!encoded) return null

  try {
    const parsed = JSON.parse(decodeBase64Url(encoded)) as Record<string, unknown>
    if (
      !parsed ||
      parsed.source !== SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE ||
      typeof parsed.createdAt !== "number"
    ) {
      return null
    }

    return {
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      createdAt: parsed.createdAt,
      draft: normalizeOptionalString(parsed.draft) ?? undefined,
      historyId: normalizeOptionalString(parsed.historyId),
      serverChatId: normalizeOptionalString(parsed.serverChatId),
      chatMode: isChatMode(parsed.chatMode) ? parsed.chatMode : undefined,
      webSearch:
        typeof parsed.webSearch === "boolean" ? parsed.webSearch : undefined,
      toolChoice: isToolChoice(parsed.toolChoice)
        ? parsed.toolChoice
        : undefined,
      selectedModel: normalizeOptionalString(parsed.selectedModel),
      selectedSystemPrompt: normalizeOptionalString(parsed.selectedSystemPrompt),
      selectedQuickPrompt: normalizeOptionalString(parsed.selectedQuickPrompt),
      temporaryChat:
        typeof parsed.temporaryChat === "boolean"
          ? parsed.temporaryChat
          : undefined,
      useOCR: typeof parsed.useOCR === "boolean" ? parsed.useOCR : undefined,
      title: normalizeOptionalString(parsed.title)
    }
  } catch {
    return null
  }
}

export const buildSidepanelChatWebUiHandoffUrl = ({
  config,
  payload
}: {
  config?: SidepanelChatWebUiConfig
  payload: SidepanelChatWebUiHandoffPayload
}) => {
  const webUiBaseUrl = resolveSidepanelChatWebUiBaseUrl(config)
  const url = new URL("/chat", `${webUiBaseUrl}/`)
  url.searchParams.set(
    SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM,
    encodeSidepanelChatWebUiHandoff(payload)
  )

  const serverChatId = normalizeOptionalString(payload.serverChatId)
  if (serverChatId) {
    url.searchParams.set(SETTINGS_SERVER_CHAT_ID_PARAM, serverChatId)
  }

  return url.toString()
}
