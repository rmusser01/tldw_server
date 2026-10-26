import React from "react"
import { useTranslation } from "react-i18next"
import {
  resolveModelSelectorSelection,
  type ModelSelectorDescriptor
} from "@/hooks/playground/modelSelectorUtils"
import { formatChatModelLabel } from "@/utils/chat-model-label"

/**
 * Side panel composer state while a reply is pending or streaming (XS-10).
 *
 * Stop is offered for the whole request, including the wait before the first
 * token, and the primary button keeps its "Send" label. While a reply is in
 * progress, pressing Send queues the message to go out after that reply.
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
            "sidepanel:composer.sendAfterReplyTitle",
            "Sends your message after the current reply finishes."
          )
        : sendTitle)
  return {
    showStop: isSending,
    queuesAfterReply,
    label: t("common:send", "Send"),
    ariaLabel: queuesAfterReply
      ? t("sidepanel:composer.sendAfterReplyAria", "Send after this reply")
      : t("playground:composer.submitAria", "Send message"),
    title
  }
}

type StatusMessage = { isBot?: boolean; message?: string | null }

type StatusModel = ModelSelectorDescriptor & { nickname?: string }

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

/**
 * The selected model's display name. The picker stores a provider-qualified
 * key (`provider:id`) while catalog rows carry `tldw:id`, so the row is found
 * the way the picker finds it; without a catalog name the label is the
 * "model (Provider)" form (CC-06).
 */
export const resolveResponseModelLabel = (
  selectedModel: string | null | undefined,
  models: readonly StatusModel[] | null | undefined
): string | null => {
  const selected = String(selectedModel ?? "").trim()
  if (!selected) return null
  const catalog = [...(models ?? [])]
  const match = (resolveModelSelectorSelection(catalog, selected).model ??
    catalog.find((model) => model.model === selected || model.name === selected)) as
    | StatusModel
    | null
    | undefined
  const nickname = typeof match?.nickname === "string" ? match.nickname.trim() : ""
  if (nickname) return nickname
  return formatChatModelLabel(match ? { model: match } : { modelId: selected }) || selected
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
