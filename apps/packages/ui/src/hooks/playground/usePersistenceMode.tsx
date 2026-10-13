import React from "react"
import { useTranslation } from "react-i18next"
import { useServerChatSaveStatus } from "@/hooks/chat/useServerChatSaveStatus"
import {
  getChatPersistenceCopy,
  resolveChatPersistenceKind
} from "@/utils/chat-persistence-status"

export type UsePersistenceModeParams = {
  temporaryChat: boolean
  serverChatId: string | null
}

/**
 * Persistence status for the active chat (CS-03 / XS-05, #3104).
 *
 * The label claims the server only when the chat has a server id and its
 * latest server write was acknowledged. Connectivity alone never counts.
 */
export function usePersistenceMode({
  temporaryChat,
  serverChatId
}: UsePersistenceModeParams) {
  const { t } = useTranslation(["playground", "common"])
  const serverSaveStatus = useServerChatSaveStatus(serverChatId)

  const persistenceKind = resolveChatPersistenceKind({
    temporaryChat,
    serverChatId,
    serverSaveStatus
  })

  const { pill: persistencePillLabel, description: persistenceModeLabel } =
    React.useMemo(
      () => getChatPersistenceCopy(t, persistenceKind),
      [persistenceKind, t]
    )

  const persistenceTooltip = React.useMemo(
    () => (
      <div className="flex flex-col gap-0.5 text-xs">
        <span className="font-medium">{persistencePillLabel}</span>
        <span className="text-text-subtle">{persistenceModeLabel}</span>
      </div>
    ),
    [persistenceModeLabel, persistencePillLabel]
  )

  const focusConnectionCard = React.useCallback(() => {
    try {
      const card = document.getElementById("server-connection-card")
      if (card) {
        card.scrollIntoView({ block: "nearest", behavior: "smooth" })
        ;(card as HTMLElement).focus()
        return
      }
    } catch {
      // ignore DOM errors and fall through to hash navigation
    }
    try {
      const base =
        window.location.href.replace(/#.*$/, "") || "/options.html"
      const target = `${base}#/settings/tldw`
      window.location.href = target
    } catch {
      // ignore navigation failures
    }
  }, [])

  return {
    persistenceKind,
    persistenceModeLabel,
    persistencePillLabel,
    persistenceTooltip,
    focusConnectionCard
  }
}
