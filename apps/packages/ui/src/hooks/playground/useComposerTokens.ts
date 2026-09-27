import React from "react"
import { useTranslation } from "react-i18next"
import { tldwChat, type ChatMessage } from "@/services/tldw"
import { isImageGenerationMessageType } from "@/utils/image-generation-chat"

export type UseComposerTokensParams = {
  message: string
  messages: any[]
  systemPrompt: string | undefined
  resolvedMaxContext: number | null
  apiModelLabel: string
  isSending: boolean
}

export function useComposerTokens({
  message,
  messages,
  systemPrompt,
  resolvedMaxContext,
  apiModelLabel,
  isSending
}: UseComposerTokensParams) {
  const { t } = useTranslation(["playground", "common"])

  const numberFormatter = React.useMemo(() => new Intl.NumberFormat(), [])
  const formatNumber = React.useCallback(
    (value: number | null) => {
      if (typeof value !== "number" || !Number.isFinite(value)) return "—"
      return numberFormatter.format(Math.round(value))
    },
    [numberFormatter]
  )

  const estimateTokensForText = React.useCallback((text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return 0
    return tldwChat.estimateTokens([
      { role: "user", content: trimmed }
    ])
  }, [])

  const draftTokenCount = React.useMemo(
    () => estimateTokensForText(message || ""),
    [estimateTokensForText, message]
  )

  const estimatedConversationTokenCount = React.useMemo(() => {
    if (isSending) {
      return null
    }
    const convoMessages: ChatMessage[] = []
    const trimmedSystem = systemPrompt?.trim()
    if (trimmedSystem) {
      convoMessages.push({ role: "system", content: trimmedSystem })
    }
    messages.forEach((msg: any) => {
      if (isImageGenerationMessageType(msg?.messageType)) {
        return
      }
      const content = typeof msg.message === "string" ? msg.message.trim() : ""
      if (!content) return
      if (msg.isBot) {
        convoMessages.push({ role: "assistant", content })
      } else {
        convoMessages.push({ role: "user", content })
      }
    })
    if (convoMessages.length === 0) return 0
    return tldwChat.estimateTokens(convoMessages)
  }, [isSending, messages, systemPrompt])
  const [conversationTokenCount, setConversationTokenCount] = React.useState(
    estimatedConversationTokenCount ?? 0
  )
  // Keep the last non-streaming total, including an empty conversation's zero.
  // Adjust state only when the non-streaming estimate changes.
  if (
    estimatedConversationTokenCount !== null &&
    conversationTokenCount !== estimatedConversationTokenCount
  ) {
    setConversationTokenCount(estimatedConversationTokenCount)
  }

  const promptTokenLabel = React.useMemo(
    () =>
      `${t("playground:tokens.prompt", "prompt")} ${formatNumber(draftTokenCount)}`,
    [draftTokenCount, formatNumber, t]
  )
  const convoTokenLabel = React.useMemo(
    () =>
      `${t("playground:tokens.total", "tokens")} ${formatNumber(conversationTokenCount)}`,
    [conversationTokenCount, formatNumber, t]
  )
  const contextTokenLabel = React.useMemo(
    () => `${formatNumber(resolvedMaxContext)} ctx`,
    [formatNumber, resolvedMaxContext]
  )
  const tokenUsageLabel = React.useMemo(
    () => `${promptTokenLabel} · ${convoTokenLabel} / ${contextTokenLabel}`,
    [contextTokenLabel, convoTokenLabel, promptTokenLabel]
  )
  const tokenUsageCompactLabel = React.useMemo(() => {
    const prompt = formatNumber(draftTokenCount)
    const convo = formatNumber(conversationTokenCount)
    const ctx = formatNumber(resolvedMaxContext)
    return `${prompt} · ${convo}/${ctx} ctx`
  }, [conversationTokenCount, draftTokenCount, formatNumber, resolvedMaxContext])

  const contextLabel = React.useMemo(
    () =>
      t(
        "common:modelSettings.form.numCtx.label",
        "Context Window Size (num_ctx)"
      ),
    [t]
  )
  const tokenUsageTooltip = React.useMemo(
    () =>
      `${apiModelLabel} · ${promptTokenLabel} · ${convoTokenLabel} · ${contextLabel} ${formatNumber(resolvedMaxContext)}`,
    [
      apiModelLabel,
      contextLabel,
      convoTokenLabel,
      formatNumber,
      promptTokenLabel,
      resolvedMaxContext
    ]
  )

  return {
    draftTokenCount,
    conversationTokenCount,
    promptTokenLabel,
    convoTokenLabel,
    contextTokenLabel,
    tokenUsageLabel,
    tokenUsageCompactLabel,
    tokenUsageTooltip,
    formatNumber,
    estimateTokensForText
  }
}
