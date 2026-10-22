import React from "react"
import { useTranslation } from "react-i18next"

/**
 * Side panel composer state while a reply is pending or streaming (XS-10).
 *
 * Stop is offered for the whole request, including the wait before the first
 * token. While a reply is in progress the primary button reads "Send next"
 * and queues the message to go out after that reply, in the same words as the
 * full-page composer (CM-02).
 */

type Translate = (key: string, fallback: string, options?: Record<string, unknown>) => string

export type SidepanelPrimaryActionInput = {
  /** A request is pending, processing or streaming. */
  isSending: boolean
  isConnectionReady: boolean
  /** Title for an idle, connected Send button. */
  sendTitle?: string
  /** Why Send is blocked right now, if it is. */
  disabledReason?: string | null
  t: Translate
}

export type SidepanelPrimaryAction = {
  /** Show Stop for the in-flight request. */
  showStop: boolean
  /** Pressing Send queues the message instead of sending it now. */
  queuesAfterReply: boolean
  label: string
  ariaLabel: string
  title?: string
}

export const resolveSidepanelPrimaryAction = ({
  isSending,
  isConnectionReady,
  sendTitle,
  disabledReason,
  t
}: SidepanelPrimaryActionInput): SidepanelPrimaryAction => {
  const queuesAfterReply = isSending && isConnectionReady
  const title =
    disabledReason ??
    (!isConnectionReady
      ? t(
          "playground:composer.connectionRequired",
          "Connect to your tldw server before sending. Your draft has not been queued."
        )
      : queuesAfterReply
        ? t(
            "playground:composer.queue.sendNextTitle",
            "Send this message after the current reply finishes."
          )
        : sendTitle)
  return {
    showStop: isSending,
    queuesAfterReply,
    label: queuesAfterReply
      ? t("playground:composer.queue.sendNext", "Send next")
      : t("common:send", "Send"),
    ariaLabel: queuesAfterReply
      ? t("playground:composer.queue.sendNextAria", "Send next, after the current reply")
      : t("playground:composer.submitAria", "Send message"),
    title
  }
}

type StatusMessage = { isBot?: boolean; message?: string | null }

type StatusModel = { model?: string; name?: string; nickname?: string }

const STREAMING_CURSOR = /▋/g

/** True until the pending reply shows any text. */
export const isAwaitingFirstToken = (
  isSending: boolean,
  messages: readonly StatusMessage[]
): boolean => {
  if (!isSending) return false
  const last = messages[messages.length - 1]
  if (!last?.isBot) return true
  return String(last.message ?? "").replace(STREAMING_CURSOR, "").trim().length === 0
}

/** The selected model's display name, falling back to its id. */
export const resolveResponseModelLabel = (
  selectedModel: string | null | undefined,
  models: readonly StatusModel[] | null | undefined
): string | null => {
  const selected = String(selectedModel ?? "").trim()
  if (!selected) return null
  const match = (models ?? []).find(
    (model) => model.model === selected || model.name === selected
  )
  return match?.nickname || match?.model || match?.name || selected
}

type SidepanelResponseStatusProps = {
  isSending: boolean
  messages: readonly StatusMessage[]
  selectedModel?: string | null
  models?: readonly StatusModel[] | null
  className?: string
}

/** "Waiting for <model>…" while a sent message has no reply text yet. */
export const SidepanelResponseStatus = ({
  isSending,
  messages,
  selectedModel,
  models,
  className
}: SidepanelResponseStatusProps) => {
  const { t } = useTranslation(["sidepanel"])
  if (!isAwaitingFirstToken(isSending, messages)) return null
  const model = resolveResponseModelLabel(selectedModel, models)
  return (
    <p
      role="status"
      aria-live="polite"
      data-testid="chat-waiting-status"
      className={`min-w-0 truncate text-xs text-text-muted ${className ?? ""}`}
    >
      {model
        ? t("sidepanel:composer.waitingForModel", "Waiting for {{model}}…", { model })
        : t("sidepanel:composer.waitingForReply", "Waiting for the reply…")}
    </p>
  )
}
