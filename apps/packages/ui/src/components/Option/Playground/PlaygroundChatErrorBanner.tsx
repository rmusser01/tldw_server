import React from "react"
import { useNavigate } from "react-router-dom"

import { RecoveryCallout } from "@/components/ui/state"
import {
  decodeChatErrorPayload,
  TLDW_ERROR_BUBBLE_PREFIX,
  type ChatErrorPayload
} from "@/utils/chat-error-message"
import {
  openModelSelector,
  openModelSettings
} from "./playground-cockpit-actions"

type ChatErrorMessageCandidate = {
  id?: string | number | null
  serverMessageId?: string | number | null
  server_message_id?: string | number | null
  isBot?: boolean
  role?: string
  message?: unknown
  content?: unknown
}

export type PlaygroundChatErrorBannerEntry = ChatErrorPayload & {
  key: string
}

const isAssistantLikeMessage = (
  entry: ChatErrorMessageCandidate | undefined
) => {
  const role = typeof entry?.role === "string" ? entry.role.toLowerCase() : ""
  return (
    entry?.isBot === true ||
    role === "assistant" ||
    (!role && entry?.isBot !== false)
  )
}

const getMessageIdentifier = (
  entry: ChatErrorMessageCandidate | undefined,
  index: number
) => {
  const identifier =
    entry?.id ?? entry?.serverMessageId ?? entry?.server_message_id ?? index
  return String(identifier)
}

const hashString = (value: string) => {
  let hash = 0x811c9dc5
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(36)
}

const getCandidateMessageText = (
  entry: ChatErrorMessageCandidate | undefined
) => {
  if (typeof entry?.message === "string") return entry.message
  if (typeof entry?.content === "string") return entry.content
  return ""
}

export const getChatErrorBannerScanSignature = (
  messages: readonly ChatErrorMessageCandidate[]
) => {
  const lastIndex = messages.length - 1
  const lastEntry = lastIndex >= 0 ? messages[lastIndex] : undefined
  const lastMessage = getCandidateMessageText(lastEntry)
  const lastEntryIsError =
    isAssistantLikeMessage(lastEntry) &&
    lastMessage.startsWith(TLDW_ERROR_BUBBLE_PREFIX)

  return [
    messages.length,
    getMessageIdentifier(lastEntry, lastIndex),
    lastEntryIsError ? "error" : "not-error",
    lastEntryIsError ? hashString(lastMessage) : ""
  ].join(":")
}

export const getLatestChatErrorBannerEntry = (
  messages: readonly ChatErrorMessageCandidate[]
): PlaygroundChatErrorBannerEntry | null => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const entry = messages[index]
    const role = typeof entry?.role === "string" ? entry.role.toLowerCase() : ""
    if (role === "user" || (!role && entry?.isBot === false)) {
      return null
    }
    if (!isAssistantLikeMessage(entry)) continue

    const message = getCandidateMessageText(entry)
    const payload = decodeChatErrorPayload(message)
    // A newer reply or streaming placeholder supersedes historical failures.
    if (!payload) {
      return null
    }
    return {
      ...payload,
      key: `${getMessageIdentifier(entry, index)}:${hashString(message)}`
    }
  }
  return null
}

/** The error reply rendered in the transcript (Message ErrorBubble). */
export const CHAT_ERROR_BUBBLE_SELECTOR = "[data-chat-error-bubble]"

/**
 * Whether the transcript's latest error reply is on screen (CC-N2). Its own
 * Retry and Switch model actions are then right there, so the composer banner
 * would only repeat the same error; the banner returns once it scrolls away.
 * Without IntersectionObserver, a rendered error reply counts as visible.
 */
export const useChatErrorBubbleInView = (errorKey: string | null) => {
  const [inView, setInView] = React.useState(false)
  React.useEffect(() => {
    if (!errorKey || typeof document === "undefined") {
      setInView(false)
      return
    }
    let observed: Element | null = null
    let observer: IntersectionObserver | null = null
    const attach = () => {
      const bubbles = document.querySelectorAll(CHAT_ERROR_BUBBLE_SELECTOR)
      const bubble = bubbles[bubbles.length - 1] ?? null
      if (bubble === observed) return
      observer?.disconnect()
      observer = null
      observed = bubble
      if (!bubble) {
        setInView(false)
        return
      }
      if (typeof IntersectionObserver === "undefined") {
        setInView(true)
        return
      }
      observer = new IntersectionObserver((entries) => {
        const entry = entries[entries.length - 1]
        if (entry) setInView(entry.isIntersecting)
      })
      observer.observe(bubble)
    }
    attach()
    // The reply may render after the banner, or be replaced (virtualized lists).
    const mutations =
      typeof MutationObserver === "undefined" ? null : new MutationObserver(attach)
    mutations?.observe(document.body, { childList: true, subtree: true })
    return () => {
      mutations?.disconnect()
      observer?.disconnect()
    }
  }, [errorKey])
  return inView
}

export const usePlaygroundChatErrorBanner = (
  messages: readonly ChatErrorMessageCandidate[]
) => {
  const scanSignature = getChatErrorBannerScanSignature(messages)
  const latestError = React.useMemo(
    () => (scanSignature ? getLatestChatErrorBannerEntry(messages) : null),
    [messages, scanSignature]
  )
  const bubbleInView = useChatErrorBubbleInView(latestError?.key ?? null)
  const latestErrorRef = React.useRef<PlaygroundChatErrorBannerEntry | null>(
    latestError
  )
  const [dismissedErrorKey, setDismissedErrorKey] = React.useState<string | null>(
    null
  )

  React.useEffect(() => {
    latestErrorRef.current = latestError
  }, [latestError])

  React.useEffect(() => {
    if (!latestError && dismissedErrorKey !== null) {
      setDismissedErrorKey(null)
    }
  }, [dismissedErrorKey, latestError])

  const dismissError = React.useCallback((key?: string) => {
    const resolvedKey = key ?? latestErrorRef.current?.key ?? null
    if (resolvedKey) {
      setDismissedErrorKey(resolvedKey)
    }
  }, [])

  const dismissAfterSuccessfulSubmit = React.useCallback(
    (key?: string | null) => {
      const resolvedKey =
        key === null ? null : key ?? latestErrorRef.current?.key ?? null
      if (resolvedKey) {
        setDismissedErrorKey(resolvedKey)
      }
    },
    []
  )

  return {
    latestError,
    visibleError:
      latestError && latestError.key !== dismissedErrorKey && !bubbleInView
        ? latestError
        : null,
    dismissError,
    dismissAfterSuccessfulSubmit
  }
}

type PlaygroundChatErrorBannerProps = {
  error: PlaygroundChatErrorBannerEntry | null
  diagnosticsLabel: string
  retryLabel?: string
  editProviderLabel?: string
  switchProviderLabel?: string
  dismissLabel: string
  onRetry?: () => void
  onEditProvider?: () => void
  onSwitchProvider?: () => void
  onDismiss: (key: string) => void
}

export const PlaygroundChatErrorBanner: React.FC<
  PlaygroundChatErrorBannerProps
> = ({
  error,
  diagnosticsLabel,
  retryLabel = "Retry",
  editProviderLabel = "Edit provider",
  switchProviderLabel = "Switch provider",
  dismissLabel,
  onRetry,
  onEditProvider,
  onSwitchProvider,
  onDismiss
}) => {
  const navigate = useNavigate()

  if (!error) {
    return null
  }

  const openDiagnostics = () => navigate("/settings/health")
  const hasInlineRecovery = Boolean(onRetry)

  const primaryAction = hasInlineRecovery
    ? {
        label: retryLabel,
        ariaLabel: retryLabel,
        onClick: onRetry ?? openDiagnostics,
        "data-testid": "playground-chat-error-retry"
      }
    : error.recoveryAction === "open-model-selector"
    ? {
        label: error.recoveryLabel || "Choose another model",
        ariaLabel: error.recoveryLabel || "Choose another model",
        onClick: () =>
          openModelSelector({
            returnFocusSelector: "[data-testid='playground-chat-error-banner']"
          })
      }
    : error.recoveryAction === "open-model-settings"
    ? {
        label: error.recoveryLabel || "Open model settings",
        ariaLabel: error.recoveryLabel || "Open model settings",
        onClick: () => openModelSettings()
      }
    : {
        label: diagnosticsLabel,
        ariaLabel: diagnosticsLabel,
        onClick: openDiagnostics
      }

  return (
    <RecoveryCallout
      state="error"
      data-testid="playground-chat-error-banner"
      title={error.summary}
      message={error.hint}
      role="alert"
      className="mb-2"
      primaryAction={primaryAction}
      secondaryActions={[
        ...(hasInlineRecovery
          ? [
              {
                label: editProviderLabel,
                ariaLabel: editProviderLabel,
                onClick: onEditProvider ?? openDiagnostics,
                "data-testid": "playground-chat-error-edit-provider"
              },
              {
                label: switchProviderLabel,
                ariaLabel: switchProviderLabel,
                onClick: onSwitchProvider ?? onEditProvider ?? openDiagnostics,
                "data-testid": "playground-chat-error-switch-provider"
              },
              {
                label: diagnosticsLabel,
                ariaLabel: diagnosticsLabel,
                onClick: openDiagnostics,
                "data-testid": "playground-chat-error-diagnostics"
              }
            ]
          : []),
        {
          label: dismissLabel,
          ariaLabel: dismissLabel,
          onClick: () => onDismiss(error.key)
        }
      ]}
    />
  )
}
