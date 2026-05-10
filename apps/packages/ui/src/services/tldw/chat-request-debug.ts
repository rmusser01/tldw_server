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

let lastChatRequestDebugSnapshot: ChatRequestDebugSnapshot | null = null

const clonePayload = (body: unknown): unknown => {
  try {
    return JSON.parse(JSON.stringify(body))
  } catch {
    return body
  }
}

export const captureChatRequestDebugSnapshot = ({
  endpoint,
  method,
  mode,
  body,
  metadata
}: CaptureChatRequestDebugSnapshotInput) => {
  lastChatRequestDebugSnapshot = {
    endpoint,
    method,
    mode,
    sentAt: new Date().toISOString(),
    body: clonePayload(body),
    metadata: metadata
      ? (clonePayload(metadata) as ChatRequestDebugMetadata)
      : undefined
  }
}

export const getLastChatRequestDebugSnapshot = () =>
  lastChatRequestDebugSnapshot

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
    const snapshot = lastChatRequestDebugSnapshot
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
