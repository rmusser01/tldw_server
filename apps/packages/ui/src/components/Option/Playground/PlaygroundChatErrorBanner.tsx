import React from "react"
import { useNavigate } from "react-router-dom"

import { RecoveryCallout } from "@/components/ui/state"
import {
  decodeChatErrorPayload,
  TLDW_ERROR_BUBBLE_PREFIX,
  type ChatErrorPayload
} from "@/utils/chat-error-message"

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
    const message = getCandidateMessageText(entry)
    if (!isAssistantLikeMessage(entry) || !message) {
      continue
    }
    const payload = decodeChatErrorPayload(message)
    if (!payload) {
      continue
    }
    return {
      ...payload,
      key: `${getMessageIdentifier(entry, index)}:${hashString(message)}`
    }
  }
  return null
}

export const usePlaygroundChatErrorBanner = (
  messages: readonly ChatErrorMessageCandidate[]
) => {
  const scanSignature = getChatErrorBannerScanSignature(messages)
  const latestError = React.useMemo(
    () => (scanSignature ? getLatestChatErrorBannerEntry(messages) : null),
    [messages, scanSignature]
  )
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
      latestError && latestError.key !== dismissedErrorKey ? latestError : null,
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

  return (
    <RecoveryCallout
      state="error"
      data-testid="playground-chat-error-banner"
      title={error.summary}
      message={error.hint}
      role="alert"
      className="mb-2"
      primaryAction={
        hasInlineRecovery
          ? {
              label: retryLabel,
              ariaLabel: retryLabel,
              onClick: onRetry,
              "data-testid": "playground-chat-error-retry"
            }
          : {
              label: diagnosticsLabel,
              ariaLabel: diagnosticsLabel,
              onClick: openDiagnostics
            }
      }
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
