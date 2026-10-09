import type { ChatCompletionRequest } from "./TldwApiClient"
import type {
  ChatToolFilterCounts,
  EffectiveChatToolRequestChoice,
  ChatToolOmissionReason
} from "@/utils/chat-tools"

export type ChatRequestDebugMode = "stream" | "non-stream"

export type ChatRequestDebugSnapshot = {
  endpoint: string
  method: string
  mode: ChatRequestDebugMode
  sentAt: string
  body: unknown
  metadata?: ChatRequestDebugMetadata
}

export type ChatToolRequestDebugMetadata = {
  model?: string
  toolChoice?: EffectiveChatToolRequestChoice
  toolOmissionReason?: ChatToolOmissionReason
  toolCounts?: ChatToolFilterCounts
}

export type ChatRequestDebugMetadata = ChatToolRequestDebugMetadata & {
  toolRequests?: ChatToolRequestDebugMetadata[]
}

type CaptureChatRequestDebugSnapshotInput = {
  endpoint: string
  method: string
  mode: ChatRequestDebugMode
  body: unknown
  metadata?: ChatRequestDebugMetadata
}

type PendingChatRequestDebugSnapshot = {
  endpoint: string
  method: string
  mode: ChatRequestDebugMode
  sentAt: string
  // Raw references are cloned lazily on first read (TASK-13511): the capture
  // path used to JSON deep-clone the full conversation payload on every chat
  // request even though the raw preview is the only reader. Cloning at read
  // time removes that synchronous per-request cost entirely.
  rawBody: unknown
  rawMetadata?: ChatRequestDebugMetadata
  clonedBody?: unknown
  clonedMetadata?: ChatRequestDebugMetadata
  cloned: boolean
}

let pendingSnapshot: PendingChatRequestDebugSnapshot | null = null

const clonePayload = (body: unknown): unknown => {
  try {
    return JSON.parse(JSON.stringify(body))
  } catch {
    return body
  }
}

const resolveSnapshot = (): ChatRequestDebugSnapshot | null => {
  const pending = pendingSnapshot
  if (!pending) return null
  if (!pending.cloned) {
    pending.clonedBody = clonePayload(pending.rawBody)
    pending.clonedMetadata = pending.rawMetadata
      ? (clonePayload(pending.rawMetadata) as ChatRequestDebugMetadata)
      : undefined
    pending.cloned = true
  }
  return {
    endpoint: pending.endpoint,
    method: pending.method,
    mode: pending.mode,
    sentAt: pending.sentAt,
    body: pending.clonedBody,
    metadata: pending.clonedMetadata
  }
}

export const captureChatRequestDebugSnapshot = ({
  endpoint,
  method,
  mode,
  body,
  metadata
}: CaptureChatRequestDebugSnapshotInput) => {
  pendingSnapshot = {
    endpoint,
    method,
    mode,
    sentAt: new Date().toISOString(),
    rawBody: body,
    rawMetadata: metadata,
    cloned: false
  }
}

export const getLastChatRequestDebugSnapshot = () => resolveSnapshot()

// Backward-compatible helper for prior /chat/completions-only consumers.
export type ChatCompletionDebugSnapshot = {
  endpoint: "/api/v1/chat/completions"
  mode: ChatRequestDebugMode
  sentAt: string
  request: ChatCompletionRequest
  metadata?: ChatRequestDebugMetadata
}

export const getLastChatCompletionDebugSnapshot =
  (): ChatCompletionDebugSnapshot | null => {
    const snapshot = resolveSnapshot()
    if (!snapshot || snapshot.endpoint !== "/api/v1/chat/completions") {
      return null
    }
    return {
      endpoint: "/api/v1/chat/completions",
      mode: snapshot.mode,
      sentAt: snapshot.sentAt,
      request: (snapshot.body || {}) as ChatCompletionRequest,
      metadata: snapshot.metadata
    }
  }
