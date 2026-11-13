import React from "react"
import { useTranslation } from "react-i18next"
import { Menu, Minimize2, SquarePen } from "lucide-react"
import { useActiveChatTitle } from "@/hooks/useActiveChatTitle"
import { useStartSavedChat } from "@/hooks/chat/useStartSavedChat"
import { useStoreMessageOption } from "@/store/option"

const actionClass =
  "inline-flex min-h-11 min-w-11 shrink-0 items-center justify-center gap-1.5 rounded-md px-2 text-sm text-text-muted hover:bg-surface2 hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"

type PlaygroundFocusBarProps = {
  exitFocusLabel: string
  onExitFocus: () => void
}

/**
 * Phone Focus mode hides the app header and sidebar. This slim bar keeps the
 * way out: the chat list, the chat's title, New chat and Exit focus (CO-01).
 * It sits above the transcript, so Exit focus no longer covers messages (CO-N1).
 */
export const PlaygroundFocusBar = ({
  exitFocusLabel,
  onExitFocus
}: PlaygroundFocusBarProps) => {
  const { t } = useTranslation(["playground", "option"])
  const { title } = useActiveChatTitle()
  const temporaryChat = useStoreMessageOption((state) => state.temporaryChat)
  const startSavedChat = useStartSavedChat()
  const chatListLabel = String(
    t("playground:focusBar.openChatList", "Open chat list")
  )
  const newChatLabel = String(
    t("playground:header.newSavedChat", "New saved chat")
  )
  const displayTitle = temporaryChat
    ? t("playground:header.newTemporaryChat", "Temporary chat (on this device)")
    : title || t("option:newChat", "New Chat")

  return (
    <header
      data-testid="playground-focus-bar"
      className="flex min-h-12 w-full shrink-0 items-center gap-1 border-b border-border/60 bg-surface px-1"
    >
      <button
        type="button"
        onClick={() => window.dispatchEvent(new Event("tldw:open-chat-sidebar"))}
        aria-label={chatListLabel}
        title={chatListLabel}
        className={actionClass}
      >
        <Menu className="size-4" aria-hidden="true" />
      </button>
      <h2 className="min-w-0 flex-1 truncate text-sm font-semibold text-text">
        {displayTitle}
      </h2>
      <button
        type="button"
        onClick={() => startSavedChat()}
        aria-label={newChatLabel}
        title={newChatLabel}
        className={actionClass}
      >
        <SquarePen className="size-4" aria-hidden="true" />
      </button>
      <button
        type="button"
        data-testid="playground-chat-layout-mode-trigger"
        onClick={onExitFocus}
        title={exitFocusLabel}
        className={`${actionClass} text-xs font-semibold text-text`}
      >
        <Minimize2 className="size-3.5" aria-hidden="true" />
        <span>{exitFocusLabel}</span>
      </button>
    </header>
  )
}
