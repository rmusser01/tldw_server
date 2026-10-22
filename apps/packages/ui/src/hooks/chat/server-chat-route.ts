/**
 * The chat page's address names the open saved chat: `/chat?chat=<server chat
 * id>` (XP-05). The server conversation id is the only chat identity (D7), so
 * local-only and temporary chats have no address of their own.
 *
 * Opening a chat from the sidebar, a note or the command palette navigates to
 * that address. Switching from one saved chat to another adds a history entry
 * so Back returns to the previous chat; filling in the address of a chat that
 * had none (a restored session, a new chat's first saved turn) replaces the
 * current entry instead.
 */
import { normalizeCharacterChatSessionId } from "@/utils/character-chat-mode-intent"

export const SERVER_CHAT_ROUTE_PARAM = "chat"

const toSearchParams = (search: string | null | undefined) =>
  new URLSearchParams((search ?? "").replace(/^\?/, ""))

/** The saved chat a location's query names, or null. */
export const readServerChatRouteId = (search: string | null | undefined): string | null =>
  normalizeCharacterChatSessionId(toSearchParams(search).get(SERVER_CHAT_ROUTE_PARAM))

/** Character chat routes (`?mode=character&chatId=...`) are owned by the character intent. */
export const isCharacterChatRouteSearch = (search: string | null | undefined): boolean =>
  toSearchParams(search).get("mode")?.trim().toLowerCase() === "character"

/** `search` with the chat parameter set to `chatId`, or removed when it is null. */
export const withServerChatRouteId = (
  search: string | null | undefined,
  chatId: string | null
): string => {
  const params = toSearchParams(search)
  if (chatId) params.set(SERVER_CHAT_ROUTE_PARAM, chatId)
  else params.delete(SERVER_CHAT_ROUTE_PARAM)
  const next = params.toString()
  return next ? `?${next}` : ""
}

export const buildServerChatPath = (chatId: string, chatPath = "/chat"): string =>
  `${chatPath}${withServerChatRouteId("", chatId)}`

/** Whether the mounted chat page's address already names a chat. */
type MountedChatRoute = { namesChat: boolean }

let mountedChatRoute: MountedChatRoute | null = null
let pendingRouteChatId: string | null = null

/** The chat page reports the chat its current address names while it is mounted. */
export const reportMountedChatRoute = (route: MountedChatRoute | null) => {
  mountedChatRoute = route
}

/**
 * Plan the navigation that opens `chatId` on the chat page. The chat page's
 * address sync leaves the URL alone until this navigation lands, so a slow
 * router (Next.js) does not record the chat twice.
 */
export const planServerChatNavigation = (
  chatId: string,
  chatPath = "/chat"
): { to: string; replace: boolean } => {
  pendingRouteChatId = chatId
  return {
    to: buildServerChatPath(chatId, chatPath),
    // From a chat page with no chat in its address there is nothing to go Back to.
    replace: mountedChatRoute !== null && !mountedChatRoute.namesChat
  }
}

/** True while a navigation to `chatId` planned above has not landed yet. */
export const isServerChatNavigationPending = (chatId: string | null): boolean =>
  chatId !== null && pendingRouteChatId === chatId

export const settleServerChatNavigation = (routeChatId: string | null) => {
  if (routeChatId !== null && pendingRouteChatId === routeChatId) pendingRouteChatId = null
}

/** Test seam: forget module state between tests. */
export const resetServerChatRouteState = () => {
  mountedChatRoute = null
  pendingRouteChatId = null
}
