import type { Message } from "@/store/option"

/** These reasons are produced only by selected-source preflight, before Chat dispatch. */
export const isLocalRagDiagnosticInfo = (info: unknown): boolean => {
  if (!info || typeof info !== "object" || Array.isArray(info)) return false
  const value = info as Record<string, unknown>
  return value.mode === "rag" && value.grounded === false &&
    (value.reason === "selected_source_retrieval_failed" ||
      value.reason === "selected_source_evidence_not_found")
}

const isDiagnostic = (message: Message): boolean =>
  message.isBot && !message.serverMessageId &&
  isLocalRagDiagnosticInfo(message.generationInfo)

/** Resolve only an undispatched pair. An answered variant or receipt keeps its user eligible. */
export const getLocalRagDiagnosticUser = (
  messages: Message[], assistant: Message
): Message | undefined => {
  if (!isDiagnostic(assistant) || !assistant.parentMessageId) return undefined
  const users = messages.filter(message => message.id === assistant.parentMessageId)
  if (users.length !== 1) return undefined
  const user = users[0]
  if (user.isBot || user.serverMessageId || !isLocalRagDiagnosticInfo(user.generationInfo)) return undefined
  const replies = [...messages.filter(message => message.isBot && message.parentMessageId === user.id), assistant]
  if (replies.some(reply => !isDiagnostic(reply) || reply.variants?.some(variant =>
    Boolean(variant.serverMessageId) || !isLocalRagDiagnosticInfo(variant.generationInfo)))) return undefined
  return user
}

/** Preserve the visible transcript; project only eligible rows into inference or automatic saving. */
export const excludeLocalRagDiagnostics = (messages: Message[]): Message[] => {
  const excludedUsers = new Set(messages.flatMap(assistant => {
    const user = getLocalRagDiagnosticUser(messages, assistant)
    return user?.id ? [user.id] : []
  }))
  return messages.filter(message => !isDiagnostic(message) && !excludedUsers.has(message.id || ""))
}
