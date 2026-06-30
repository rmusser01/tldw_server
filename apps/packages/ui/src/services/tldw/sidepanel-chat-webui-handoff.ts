import type { ToolChoice } from "@/store/option"

export const SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM = "handoff"
export const SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE = "sidepanel-chat"
export const SIDEPANEL_CHAT_WEBUI_HANDOFF_MAX_AGE_MS = 10 * 60 * 1000
const SIDEPANEL_CHAT_WEBUI_HANDOFF_MAX_CLOCK_SKEW_MS = 60 * 1000
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

const hasOwn = (record: Record<string, unknown>, key: string) =>
  Object.prototype.hasOwnProperty.call(record, key)

const isHttpUrl = (url: URL) =>
  url.protocol === "http:" || url.protocol === "https:"

const normalizeNullableStringField = (
  record: Record<string, unknown>,
  key: string
): string | null | undefined => {
  if (!hasOwn(record, key)) return undefined
  const raw = record[key]
  if (raw == null) return null
  if (typeof raw !== "string") return undefined
  const trimmed = raw.trim()
  return trimmed.length > 0 ? trimmed : null
}

const normalizeWebUiBaseUrl = (
  value: string | null | undefined
): string | null => {
  const trimmed = normalizeOptionalString(value)
  if (!trimmed) return null
  try {
    const url = new URL(trimmed)
    if (!isHttpUrl(url)) return null
    const pathname = trimTrailingSlash(url.pathname)
    return trimTrailingSlash(`${url.origin}${pathname === "/" ? "" : pathname}`)
  } catch {
    return null
  }
}

const normalizeOriginBaseUrl = (value: string | null | undefined): URL | null => {
  const trimmed = normalizeOptionalString(value)
  if (!trimmed) return null
  try {
    const url = new URL(trimmed)
    return isHttpUrl(url) ? url : null
  } catch {
    return null
  }
}

export const resolveSidepanelChatWebUiBaseUrl = (
  config: SidepanelChatWebUiConfig = {}
) => {
  const explicitWebUiUrl =
    normalizeWebUiBaseUrl(config.webUiUrl) ??
    normalizeWebUiBaseUrl(config.webuiUrl)
  if (explicitWebUiUrl) return explicitWebUiUrl

  const serverUrl = normalizeOriginBaseUrl(config.serverUrl)
  if (serverUrl) {
    if (serverUrl.port === "8000") {
      serverUrl.port = "8080"
    }
    return trimTrailingSlash(serverUrl.origin)
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

const isFreshHandoffCreatedAt = (createdAt: number) => {
  if (!Number.isFinite(createdAt)) return false
  const ageMs = Date.now() - createdAt
  return (
    ageMs <= SIDEPANEL_CHAT_WEBUI_HANDOFF_MAX_AGE_MS &&
    ageMs >= -SIDEPANEL_CHAT_WEBUI_HANDOFF_MAX_CLOCK_SKEW_MS
  )
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
      typeof parsed.createdAt !== "number" ||
      !isFreshHandoffCreatedAt(parsed.createdAt)
    ) {
      return null
    }

    return {
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      createdAt: parsed.createdAt,
      draft: normalizeOptionalString(parsed.draft) ?? undefined,
      historyId: normalizeNullableStringField(parsed, "historyId"),
      serverChatId: normalizeNullableStringField(parsed, "serverChatId"),
      chatMode: isChatMode(parsed.chatMode) ? parsed.chatMode : undefined,
      webSearch:
        typeof parsed.webSearch === "boolean" ? parsed.webSearch : undefined,
      toolChoice: isToolChoice(parsed.toolChoice)
        ? parsed.toolChoice
        : undefined,
      selectedModel: normalizeNullableStringField(parsed, "selectedModel"),
      selectedSystemPrompt: normalizeNullableStringField(
        parsed,
        "selectedSystemPrompt"
      ),
      selectedQuickPrompt: normalizeNullableStringField(
        parsed,
        "selectedQuickPrompt"
      ),
      temporaryChat:
        typeof parsed.temporaryChat === "boolean"
          ? parsed.temporaryChat
          : undefined,
      useOCR: typeof parsed.useOCR === "boolean" ? parsed.useOCR : undefined,
      title: normalizeNullableStringField(parsed, "title")
    }
  } catch {
    return null
  }
}

const buildWebUiChatUrl = (baseUrl: string): URL | null => {
  try {
    const base = new URL(`${baseUrl}/`)
    if (!isHttpUrl(base)) return null
    return new URL("chat", base)
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
  const url =
    buildWebUiChatUrl(webUiBaseUrl) ??
    buildWebUiChatUrl(DEFAULT_WEBUI_URL) ??
    new URL("/chat", DEFAULT_WEBUI_URL)
  const fragmentParams = new URLSearchParams()
  fragmentParams.set(
    SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM,
    encodeSidepanelChatWebUiHandoff(payload)
  )
  url.hash = fragmentParams.toString()

  return url.toString()
}
