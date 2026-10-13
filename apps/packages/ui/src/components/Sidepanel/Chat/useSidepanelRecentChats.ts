import React from "react"

import { PageAssistDatabase } from "@/db/dexie/chat"
import type { HistoryInfo } from "@/db/dexie/types"
import type { SidepanelChatOwner } from "@/hooks/useSidepanelChatOwner"
import {
  useServerChatHistory,
  type ServerChatHistoryItem
} from "@/hooks/useServerChatHistory"

/** How many recent chats the side panel lists without a search (XS-06). */
export const SIDEPANEL_RECENT_CHAT_LIMIT = 8
/** Read a few extra of each, since open tabs and local copies are left out. */
const RECENT_SERVER_FETCH = 10
const RECENT_LOCAL_FETCH = 20

export type SidepanelRecentChat =
  | { kind: "server"; key: string; title: string; at: number; chat: ServerChatHistoryItem }
  | { kind: "local"; key: string; title: string; at: number; history: HistoryInfo }

const finiteOrZero = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value) ? value : 0

/**
 * Merge recent server chats and this device's chat histories, newest first.
 *
 * Chats that already have a tab are left out, and so is a local copy of a
 * server chat the list already shows (or that has a tab).
 */
export const mergeRecentChats = ({
  server,
  local,
  openHistoryIds,
  openServerChatIds,
  limit = SIDEPANEL_RECENT_CHAT_LIMIT
}: {
  server: readonly ServerChatHistoryItem[]
  local: readonly HistoryInfo[]
  openHistoryIds: ReadonlySet<string>
  openServerChatIds: ReadonlySet<string>
  limit?: number
}): SidepanelRecentChat[] => {
  const listedServerIds = new Set(server.map((chat) => String(chat.id)))
  const serverItems: SidepanelRecentChat[] = server
    .filter((chat) => !openServerChatIds.has(String(chat.id)))
    .map((chat) => ({
      kind: "server",
      key: `server-${chat.id}`,
      title: chat.title,
      at: finiteOrZero(chat.updatedAtMs) || finiteOrZero(chat.createdAtMs),
      chat
    }))
  const localItems: SidepanelRecentChat[] = local
    .filter(
      (history) =>
        !openHistoryIds.has(history.id) &&
        !(
          history.server_chat_id &&
          (openServerChatIds.has(history.server_chat_id) ||
            listedServerIds.has(history.server_chat_id))
        )
    )
    .map((history) => ({
      kind: "local",
      key: `local-${history.id}`,
      title: history.title,
      at: finiteOrZero(history.createdAt),
      history
    }))
  return [...serverItems, ...localItems]
    .sort((a, b) => b.at - a.at)
    .slice(0, limit)
}

type RecentChatsOwner = Pick<SidepanelChatOwner, "ownerKey" | "isCurrent"> & {
  snapshot: Pick<SidepanelChatOwner["snapshot"], "requestScope">
}

/**
 * The side panel's recent chats: the full page's server history query (first
 * page, most recently updated) plus this account's histories on this device.
 */
export const useSidepanelRecentChats = ({
  owner,
  enabled,
  openHistoryIds,
  openServerChatIds
}: {
  owner?: RecentChatsOwner
  enabled: boolean
  openHistoryIds: ReadonlySet<string>
  openServerChatIds: ReadonlySet<string>
}): SidepanelRecentChat[] => {
  const active = enabled && Boolean(owner?.isCurrent())
  const server = useServerChatHistory("", {
    enabled: active,
    owner: owner
      ? { key: owner.ownerKey, requestScope: owner.snapshot.requestScope, isCurrent: owner.isCurrent }
      : undefined,
    mode: "overview",
    page: 1,
    limit: RECENT_SERVER_FETCH
  })
  const [local, setLocal] = React.useState<{
    owner?: RecentChatsOwner
    items: HistoryInfo[]
  }>({ items: [] })
  // Re-read when tabs open or close: a closed tab's chat becomes recent.
  const openKey = React.useMemo(
    () => [...openHistoryIds].sort().join("\n"),
    [openHistoryIds]
  )

  React.useEffect(() => {
    if (!active || !owner) return
    let live = true
    const ownerKey = owner.ownerKey
    const belongsToOwner = (history: HistoryInfo) =>
      history.server_scope_key === ownerKey
    void (async () => {
      try {
        const items = await new PageAssistDatabase().getRecentChatHistories(
          RECENT_LOCAL_FETCH,
          belongsToOwner
        )
        if (live && owner.isCurrent()) {
          setLocal({ owner, items: items.filter(belongsToOwner) })
        }
      } catch (error) {
        console.warn("[sidepanel] Could not read recent chats on this device", error)
        if (live) setLocal({ owner, items: [] })
      }
    })()
    return () => {
      live = false
    }
  }, [active, owner, openKey])

  const serverChats = server.data
  const localChats = local.owner === owner ? local.items : null
  return React.useMemo(
    () =>
      active
        ? mergeRecentChats({
            server: serverChats ?? [],
            local: localChats ?? [],
            openHistoryIds,
            openServerChatIds
          })
        : [],
    [active, localChats, openHistoryIds, openServerChatIds, serverChats]
  )
}
