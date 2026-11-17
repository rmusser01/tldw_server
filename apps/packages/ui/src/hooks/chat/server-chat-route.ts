/**
 * The chat page's address names the open saved chat: `/chat?chat=<server chat
 * id>` (XP-05). The server conversation id is the only chat identity (D7), so
 * local-only and temporary chats have no address of their own.
 *
 * Opening a chat from the sidebar, a note or the command palette navigates to
 * that address, and New chat navigates to the bare chat page. Leaving a saved
 * chat this way adds a history entry so Back returns to it; filling in the
 * address of a chat that had none (a restored session, a new chat's first
 * saved turn) replaces the current entry instead.
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

/**
 * A character chat route in the location's query, or in a hash route outside
 * a hash router (`/options.html#/chat?mode=character&...`), which the
 * Playground's character intent also reads.
 */
export const isCharacterChatRouteLocation = (
  search: string | null | undefined,
  hash: string | null | undefined
): boolean => {
  const hashQueryStart = (hash ?? "").indexOf("?")
  return (
    isCharacterChatRouteSearch(search) ||
    (hashQueryStart >= 0 && isCharacterChatRouteSearch((hash ?? "").slice(hashQueryStart)))
  )
}

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

/** The history handoff the mounted chat page has finished with, loaded or not. */
let settledHistoryHandoff: string | null = null

/**
 * The Playground has finished with the history handoff link in the browser
 * address. A link it loaded is already gone from the address; one it could
 * not load stays there, and the next address write drops it instead of
 * waiting for it for as long as the page stays open (#3119).
 */
export const settleHistoryHandoff = (
  browserHref: string | null | undefined = typeof window === "undefined"
    ? null
    : window.location.href
) => {
  settledHistoryHandoff = browserHref
    ? browserQueries(browserHref)
        .map((params) => params.get(HISTORY_SELECTION_HANDOFF_PARAM))
        .find((value) => value !== null) ?? null
    : null
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
 * - `"consumed"`: only the router's stale query carries it, or the page has
 *   settled it (see `settleHistoryHandoff`).
 */
export const historyHandoffState = (
  routerSearch: string | null | undefined,
  browserHref: string | null | undefined = typeof window === "undefined"
    ? null
    : window.location.href
): "none" | "pending" | "consumed" => {
  const handoff = toSearchParams(routerSearch).get(HISTORY_SELECTION_HANDOFF_PARAM)
  if (handoff === null) return "none"
  if (handoff === settledHistoryHandoff) return "consumed"
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
 * longer carries, or (for the history handoff) that the page has settled, so
 * writing the chat address from the router's stale query never brings a
 * consumed one back. Other parameters are always kept.
 */
export const withoutConsumedParams = (
  search: string | null | undefined,
  browserHref: string | null | undefined = typeof window === "undefined"
    ? null
    : window.location.href
): string => {
  const params = toSearchParams(search)
  const queries = browserHref ? browserQueries(browserHref) : null
  for (const name of CONSUMED_ON_ARRIVAL_PARAMS) {
    const value = params.get(name)
    if (value === null) continue
    const settled = name === HISTORY_SELECTION_HANDOFF_PARAM && value === settledHistoryHandoff
    if (settled || (queries && !queries.some((query) => query.has(name)))) {
      params.delete(name)
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
/** The chat a planned navigation opens; `chatId: null` is New chat's bare chat page. */
let pendingRoute: { chatId: string | null; plannedAt: number } | null = null

/**
 * How long the address sync waits for a planned navigation. A same-page
 * navigation lands well within this; one that never lands (cancelled by a
 * leave-page prompt, say) must not hold the address for good (#3119).
 */
export const PLANNED_CHAT_NAVIGATION_TIMEOUT_MS = 3_000

/**
 * The chat page reports the chat its current address names while it is
 * mounted, and null when it unmounts. A navigation it planned can no longer
 * land on it then, so the plan is dropped.
 */
export const reportMountedChatRoute = (route: MountedChatRoute | null) => {
  mountedChatRoute = route
  if (route === null) {
    pendingRoute = null
    settledHistoryHandoff = null
  }
}

/**
 * Plan the navigation that opens `chatId` on the chat page, or the bare chat
 * page for New chat (`null`). The chat page's address sync leaves the URL
 * alone until this navigation lands, so a slow router (Next.js) does not
 * record the chat twice, and its in-place rewrite cannot cancel New chat's
 * history entry. A later plan replaces this one.
 */
export const planServerChatNavigation = (
  chatId: string | null,
  chatPath = "/chat"
): { to: string; replace: boolean } => {
  pendingRoute = { chatId, plannedAt: Date.now() }
  return {
    to: chatId === null ? chatPath : buildServerChatPath(chatId, chatPath),
    // From a chat page with no chat in its address there is nothing to go Back to.
    replace: mountedChatRoute !== null && !mountedChatRoute.namesChat
  }
}

/** Milliseconds the address sync may still wait for the planned navigation, or 0. */
export const serverChatNavigationWaitMs = (): number => {
  if (pendingRoute === null) return 0
  const left = pendingRoute.plannedAt + PLANNED_CHAT_NAVIGATION_TIMEOUT_MS - Date.now()
  if (left > 0) return left
  pendingRoute = null
  return 0
}

/** True while a navigation to `chatId` (null: the bare chat page) planned above has not landed yet. */
export const isServerChatNavigationPending = (chatId: string | null): boolean =>
  serverChatNavigationWaitMs() > 0 && pendingRoute?.chatId === chatId

/**
 * A new location arrived: the planned navigation landed, or something else
 * replaced it and it will not. Either way the address sync stops waiting.
 */
export const settleServerChatNavigation = () => {
  pendingRoute = null
}

/** Test seam: forget module state between tests. */
export const resetServerChatRouteState = () => {
  mountedChatRoute = null
  pendingRoute = null
  settledHistoryHandoff = null
}
