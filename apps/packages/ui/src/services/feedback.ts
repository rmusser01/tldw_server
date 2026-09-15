import { bgRequest } from "@/services/background-proxy"
import { tldwClient, type ServerChatSummary } from "@/services/tldw/TldwApiClient"

export type FeedbackType = "helpful" | "relevance" | "report"

export type ExplicitFeedbackRequest = {
  conversation_id?: string
  message_id?: string
  feedback_id?: string
  feedback_type: FeedbackType
  helpful?: boolean
  relevance_score?: number
  document_ids?: string[]
  chunk_ids?: string[]
  corpus?: string
  issues?: string[]
  user_notes?: string
  query?: string
  session_id?: string
  idempotency_key?: string
}

export type ExplicitFeedbackResponse = {
  ok?: boolean
  feedback_id?: string
  [key: string]: any
}

export type ImplicitFeedbackEvent = {
  event_type: "click" | "expand" | "copy" | "dwell_time" | "citation_used"
  query?: string
  feedback_id?: string
  doc_id?: string
  chunk_ids?: string[]
  rank?: number
  impression_list?: string[]
  corpus?: string
  user_id?: string
  session_id?: string
  conversation_id?: string
  message_id?: string
  dwell_ms?: number
}

let cachedSessionId: string | null = null
const FEEDBACK_EXPLICIT_TIMEOUT_MS = 10000
const FEEDBACK_IMPLICIT_TIMEOUT_MS = 8000

export const getFeedbackSessionId = (): string => {
  if (cachedSessionId) return cachedSessionId
  try {
    cachedSessionId = crypto.randomUUID()
  } catch {
    cachedSessionId = `session_${Date.now()}`
  }
  return cachedSessionId
}

export async function submitExplicitFeedback(
  payload: ExplicitFeedbackRequest
): Promise<ExplicitFeedbackResponse> {
  await tldwClient.initialize().catch(() => null)
  return await bgRequest<ExplicitFeedbackResponse>({
    path: "/api/v1/feedback/explicit",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload,
    timeoutMs: FEEDBACK_EXPLICIT_TIMEOUT_MS
  })
}

export async function submitImplicitFeedback(
  payload: ImplicitFeedbackEvent
): Promise<{ ok?: boolean; disabled?: boolean }> {
  await tldwClient.initialize().catch(() => null)
  return await bgRequest<{ ok?: boolean; disabled?: boolean }>({
    path: "/api/v1/rag/feedback/implicit",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload,
    timeoutMs: FEEDBACK_IMPLICIT_TIMEOUT_MS,
    suppressBackendUnavailableEvent: true
  })
}

export async function updateChatRating(
  chatId: string,
  rating: number,
  expectedVersion?: number
): Promise<ServerChatSummary> {
  await tldwClient.initialize().catch(() => null)
  return await tldwClient.updateChat(
    chatId,
    { rating },
    { expectedVersion }
  )
}
