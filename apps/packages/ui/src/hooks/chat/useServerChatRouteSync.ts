import React from "react"
import { useLocation, useNavigate } from "react-router-dom"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { useStoreMessageOption } from "@/store/option"
import {
  historyHandoffState,
  isCharacterChatRouteSearch,
  isServerChatNavigationPending,
  readServerChatRouteId,
  reportMountedChatRoute,
  settleServerChatNavigation,
  withoutHistoryHandoff,
  withServerChatRouteId
} from "@/hooks/chat/server-chat-route"

/**
 * Keep the chat page's address and the open saved chat in step (XP-05).
 *
 * - When the address changes (Back, Forward, a pasted `/chat?chat=<id>` link)
 *   and names another chat, open that chat.
 * - When the open chat changes without a navigation (a restored session, a new
 *   chat's first saved turn, New chat, a chat the server no longer has),
 *   rewrite the address in place so reloading reopens what is on screen.
 *
 * Character chat routes (`?mode=character`) keep their own address handling.
 */
export const useServerChatRouteSync = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const selectServerChat = useSelectServerChat()
  const serverChatId = useStoreMessageOption((state) => state.serverChatId)
  const temporaryChat = useStoreMessageOption((state) => state.temporaryChat)
  const search = location.search ?? ""
  const routeChatId = readServerChatRouteId(search)
  const characterRoute = isCharacterChatRouteSearch(search)
  const handledLocationKey = React.useRef<string | null>(null)

  React.useEffect(() => {
    reportMountedChatRoute({ namesChat: characterRoute || routeChatId !== null })
  }, [characterRoute, routeChatId])
  React.useEffect(() => () => reportMountedChatRoute(null), [])

  React.useEffect(() => {
    const locationChanged = handledLocationKey.current !== location.key
    handledLocationKey.current = location.key
    if (characterRoute) return
    const state = useStoreMessageOption.getState()
    if (locationChanged && routeChatId) {
      settleServerChatNavigation(routeChatId)
      if (routeChatId === state.serverChatId) return
      const opened = selectServerChat(
        { id: routeChatId } as ServerChatSummary,
        { fromAddress: true }
      )
      if (opened !== false) return
      // The user kept the chat on screen (for example a temporary chat they
      // chose not to discard), so the address goes back to naming that chat.
    }
    const current = useStoreMessageOption.getState()
    // No saved chat open (null or not yet set) means the address names none.
    const activeChatId = current.temporaryChat ? null : current.serverChatId ?? null
    if (activeChatId === routeChatId) return
    // A navigation that names this chat is on its way; let it land.
    if (isServerChatNavigationPending(activeChatId)) return
    // A history handoff link is consumed by the Playground first; writing the
    // address meanwhile would drop it, and writing it afterwards from the
    // router's stale query would bring it back.
    const handoff = historyHandoffState(search)
    if (handoff === "pending") return
    const baseSearch = handoff === "consumed" ? withoutHistoryHandoff(search) : search
    navigate(
      {
        pathname: location.pathname,
        search: withServerChatRouteId(baseSearch, activeChatId),
        hash: location.hash
      },
      { replace: true }
    )
  }, [
    characterRoute,
    location.hash,
    location.key,
    location.pathname,
    navigate,
    routeChatId,
    search,
    selectServerChat,
    serverChatId,
    temporaryChat
  ])
}
