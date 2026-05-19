export const WEB_CLIPPER_ANALYZE_MESSAGE_TYPE = "web-clipper:analyze"

export type WebClipAnalyzeMessageLike = {
  id?: string | number | null
  messageType?: string | null
}

export const collectWebClipAnalyzeMessageIds = (
  messages: WebClipAnalyzeMessageLike[]
): Set<string> =>
  new Set(
    messages
      .map((message) =>
        message.id == null ? "" : String(message.id).trim()
      )
      .filter(Boolean)
  )

export const hasSubmittedWebClipAnalyzeMessage = (
  messages: WebClipAnalyzeMessageLike[],
  baselineMessageIds: Set<string>
): boolean =>
  messages.some((message) => {
    if (message.messageType !== WEB_CLIPPER_ANALYZE_MESSAGE_TYPE) {
      return false
    }

    const messageId = message.id == null ? "" : String(message.id).trim()
    if (!messageId) return false
    return !baselineMessageIds.has(messageId)
  })
