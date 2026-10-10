import { generateID } from "@/utils/generate-id"

export type QueueStatus = "queued" | "blocked" | "sending"

export type QueuedRequestSnapshot = {
  selectedModel: string | null
  chatMode: "normal" | "rag" | "vision"
  webSearch: boolean
  compareMode: boolean
  compareSelectedModels: string[]
  selectedSystemPrompt: string | null
  selectedQuickPrompt: string | null
  toolChoice: string | null
  useOCR: boolean
}

export type QueuedRequestInput = {
  id?: string
  clientRequestId?: string
  conversationId?: string | null
  promptText?: string
  message?: string
  image?: string
  attachments?: unknown[]
  sourceContext?: Record<string, unknown> | null
  snapshot?: Partial<QueuedRequestSnapshot>
  status?: QueueStatus
  blockedReason?: string | null
  attemptCount?: number
  createdAt?: number
  updatedAt?: number
}

export type QueuedRequest = {
  id: string
  clientRequestId: string
  conversationId: string | null
  promptText: string
  // Transitional alias for existing chat surfaces that still read `message`.
  message: string
  image: string
  attachments: unknown[]
  sourceContext: Record<string, unknown> | null
  snapshot: QueuedRequestSnapshot
  status: QueueStatus
  blockedReason: string | null
  attemptCount: number
  createdAt: number
  updatedAt: number
}

const DEFAULT_SNAPSHOT: QueuedRequestSnapshot = {
  selectedModel: null,
  chatMode: "normal",
  webSearch: false,
  compareMode: false,
  compareSelectedModels: [],
  selectedSystemPrompt: null,
  selectedQuickPrompt: null,
  toolChoice: null,
  useOCR: false
}

export const buildQueuedRequest = (
  input: QueuedRequestInput & { promptText?: string; message?: string }
): QueuedRequest => {
  const now = Date.now()
  const promptText = input.promptText ?? input.message ?? ""

  return {
    id: input.id ?? generateID(),
    clientRequestId: input.clientRequestId ?? generateID(),
    conversationId: input.conversationId ?? null,
    promptText,
    message: promptText,
    image: input.image ?? "",
    attachments: input.attachments ?? [],
    sourceContext: input.sourceContext ?? null,
    snapshot: {
      ...DEFAULT_SNAPSHOT,
      ...(input.snapshot ?? {})
    },
    status: input.status ?? "queued",
    blockedReason: input.blockedReason ?? null,
    attemptCount: input.attemptCount ?? 0,
    createdAt: input.createdAt ?? now,
    updatedAt: input.updatedAt ?? now
  }
}

export const normalizeQueuedRequests = (
  queue: QueuedRequestInput[]
): QueuedRequest[] => queue.map((item) => buildQueuedRequest(item))

export const restoreQueuedRequests = (
  queue: QueuedRequestInput[]
): QueuedRequest[] =>
  normalizeQueuedRequests(queue).map((item) =>
    item.status === "sending"
      ? {
          ...item,
          status: "queued",
          blockedReason: null
        }
      : item
  )

export const moveQueuedRequestToFront = (
  queue: QueuedRequest[],
  requestId: string
): QueuedRequest[] => {
  const target = queue.find((item) => item.id === requestId)
  if (!target) return queue
  return [target, ...queue.filter((item) => item.id !== requestId)]
}

export const blockQueuedRequest = (
  item: QueuedRequest,
  blockedReason: string
): QueuedRequest => ({
  ...item,
  status: "blocked",
  blockedReason,
  updatedAt: Date.now()
})
