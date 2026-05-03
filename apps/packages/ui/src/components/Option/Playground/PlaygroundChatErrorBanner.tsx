import React from "react"
import { X } from "lucide-react"
import { Link } from "react-router-dom"

import {
  decodeChatErrorPayload,
  type ChatErrorPayload
} from "@/utils/chat-error-message"

type ChatErrorMessageCandidate = {
  id?: string | number | null
  isBot?: boolean
  role?: string
  message?: unknown
}

export type PlaygroundChatErrorBannerEntry = ChatErrorPayload & {
  key: string
}

export const getLatestChatErrorBannerEntry = (
  messages: readonly ChatErrorMessageCandidate[]
): PlaygroundChatErrorBannerEntry | null => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const entry = messages[index]
    const role = typeof entry?.role === "string" ? entry.role.toLowerCase() : ""
    const assistantLike =
      entry?.isBot === true ||
      role === "assistant" ||
      (!role && entry?.isBot !== false)
    if (!assistantLike || typeof entry?.message !== "string") {
      continue
    }
    const payload = decodeChatErrorPayload(entry.message)
    if (!payload) {
      continue
    }
    const id = entry.id != null ? String(entry.id) : String(index)
    return {
      ...payload,
      key: `${id}:${entry.message}`
    }
  }
  return null
}

export const usePlaygroundChatErrorBanner = (
  messages: readonly ChatErrorMessageCandidate[]
) => {
  const latestError = React.useMemo(
    () => getLatestChatErrorBannerEntry(messages),
    [messages]
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

  const dismissAfterSuccessfulSubmit = React.useCallback(() => {
    const resolvedKey = latestErrorRef.current?.key ?? null
    if (resolvedKey) {
      setDismissedErrorKey(resolvedKey)
    }
  }, [])

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
  dismissLabel: string
  onDismiss: (key: string) => void
}

export const PlaygroundChatErrorBanner: React.FC<
  PlaygroundChatErrorBannerProps
> = ({ error, diagnosticsLabel, dismissLabel, onDismiss }) => {
  if (!error) {
    return null
  }

  return (
    <div
      role="alert"
      data-testid="playground-chat-error-banner"
      className="mb-2 flex flex-col gap-2 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-text shadow-sm sm:flex-row sm:items-start sm:justify-between"
    >
      <div className="min-w-0">
        <p className="font-medium text-destructive">{error.summary}</p>
        <p className="mt-1 text-xs text-text-muted">{error.hint}</p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <Link
          to="/settings/health"
          className="text-xs font-medium text-destructive underline hover:text-destructive"
        >
          {diagnosticsLabel}
        </Link>
        <button
          type="button"
          onClick={() => onDismiss(error.key)}
          className="inline-flex items-center rounded-full p-1 text-destructive hover:bg-destructive/10"
          aria-label={dismissLabel}
          title={dismissLabel}
        >
          <X className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
      </div>
    </div>
  )
}
