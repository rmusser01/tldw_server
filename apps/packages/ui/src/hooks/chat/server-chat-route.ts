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
import {
  RESEARCH_RETURN_RUN_ID_PARAM,
  SETTINGS_HISTORY_ID_PARAM,
  SETTINGS_SERVER_CHAT_ID_PARAM
} from "@/utils/settings-return-params"

export const SERVER_CHAT_ROUTE_PARAM = "chat"

const toSearchParams = (search: string | null | undefined) =>
  new URLSearchParams((search ?? "").replace(/^\?/, ""))

/** The saved chat a location's query names, or null. */
export const readServerChatRouteId = (search: string | null | undefined): string | null =>
  normalizeCharacterChatSessionId(toSearchParams(search).get(SERVER_CHAT_ROUTE_PARAM))

/** Character chat routes (`?mode=character&chatId=...`) are owned by the character intent. */
export const isCharacterChatRouteSearch = (search: string | null | undefined): boolean =>
  toSearchParams(search).get("mode")?.trim().toLowerCase() === "character"

/** The one-shot history handoff a link may carry; the Playground consumes it. */
export const HISTORY_SELECTION_HANDOFF_PARAM = "historySelection"

/** The query string of a browser address, from either the URL or a hash route. */
const browserQueries = (href: string): URLSearchParams[] => {
  try {
    const url = new URL(href)
    const hashQueryStart = url.hash.indexOf("?")
    return [
      url.searchParams,
      new URLSearchParams(hashQueryStart < 0 ? "" : url.hash.slice(hashQueryStart + 1))
    ]
  } catch {
    return []
  }
}

/**
 * Where a router query that carries the history handoff stands against the
 * real browser address. The Playground removes the handoff with
 * `history.replaceState`, which the router does not see, so the router's
 * query can still carry a handoff the page has already consumed.
 *
 * - `"none"`: the router query carries no handoff.
 * - `"pending"`: the browser address still carries it, so the page has not
 *   consumed it yet and the address must be left alone.
 * - `"consumed"`: only the router's stale query carries it.
 */
export const historyHandoffState = (
  routerSearch: string | null | undefined,
  browserHref: string | null | undefined = typeof window === "undefined"
    ? null
    : window.location.href
): "none" | "pending" | "consumed" => {
  if (!toSearchParams(routerSearch).has(HISTORY_SELECTION_HANDOFF_PARAM)) return "none"
  if (!browserHref) return "pending"
  return browserQueries(browserHref).some((params) =>
    params.has(HISTORY_SELECTION_HANDOFF_PARAM)
  )
    ? "pending"
    : "consumed"
}

/**
 * One-shot parameters the Playground reads on arrival and then removes with
 * `history.replaceState`: the history handoff and the Settings or research
 * return targets. The router does not see that removal.
 */
const CONSUMED_ON_ARRIVAL_PARAMS = [
  HISTORY_SELECTION_HANDOFF_PARAM,
  SETTINGS_HISTORY_ID_PARAM,
  SETTINGS_SERVER_CHAT_ID_PARAM,
  RESEARCH_RETURN_RUN_ID_PARAM
]

/**
 * `search` without the one-shot parameters the real browser address no
 * longer carries, so writing the chat address from the router's stale query
 * never brings a consumed one back. Other parameters are always kept.
 */
export const withoutConsumedParams = (
  search: string | null | undefined,
  browserHref: string | null | undefined = typeof window === "undefined"
    ? null
    : window.location.href
): string => {
  const params = toSearchParams(search)
  if (browserHref) {
    const queries = browserQueries(browserHref)
    for (const name of CONSUMED_ON_ARRIVAL_PARAMS) {
      if (params.has(name) && !queries.some((query) => query.has(name))) {
        params.delete(name)
      }
    }
  }
  const next = params.toString()
  return next ? `?${next}` : ""
}

/**
 * True when the address can name this chat: the id survives a write and a
 * read of the chat parameter. Other ids get no address, rather than an
 * endless rewrite of one that reads back as "no chat".
 */
export const isAddressableServerChatId = (chatId: string): boolean =>
  readServerChatRouteId(withServerChatRouteId("", chatId)) === chatId

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
