import { browser } from "wxt/browser"

import {
  getRecentChatFromCopilot,
  getFullChatData
} from "@/db/dexie/helpers"
import { copilotResumeLastChat } from "@/services/app"
import {
  getChatSettingsStorageKey,
  normalizeChatSettingsRecord,
  resolveChatSettingsKey
} from "@/services/chat-settings"
import { useSidepanelChatTabsStore, type SidepanelChatSnapshot, type SidepanelChatTab } from "@/store/sidepanel-chat-tabs"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { createSafeStorage } from "@/utils/safe-storage"
import {
  getSidepanelDraftStorageKey,
  getSidepanelOverlayResumeMarkerKey
} from "@/utils/sidepanel-overlay-resume"

export type SidepanelTabsState = {
  version: 2
  ownerKey: string
  tabs: SidepanelChatTab[]
  activeTabId: string | null
  snapshotsById: Record<string, SidepanelChatSnapshot>
}

export const getTabsStorageKey = (id: number | null | undefined, ownerKey: string) =>
  `sidepanelChatTabsState:v2:${encodeURIComponent(ownerKey)}:${id != null ? `tab-${id}` : "global"}`

/** Recover a completed durable reply when its tab snapshot still has a stream cursor. */
const recoverCompletedSnapshotReplies = async (
  snapshot: SidepanelChatSnapshot,
  ownerKey: string,
  isCurrent: () => boolean
): Promise<SidepanelChatSnapshot> => {
  const pending = snapshot.messages?.filter(message => message.isBot && message.id &&
    !message.serverMessageId && message.message.endsWith("▋")) || []
  if (!snapshot.historyId || pending.length === 0 || !isCurrent()) return snapshot
  try {
    const saved = await getFullChatData(snapshot.historyId)
    if (!isCurrent() || !saved || saved.historyInfo.id !== snapshot.historyId ||
        saved.historyInfo.server_scope_key !== ownerKey ||
        (saved.historyInfo.server_chat_id ?? null) !== snapshot.serverChatId) return snapshot
    const messages = snapshot.messages.map(message => {
      if (!pending.includes(message)) return message
      const matches = saved.messages.filter(row => row.id === message.id &&
        row.history_id === snapshot.historyId && row.role === "assistant" &&
        (row.parent_message_id ?? null) === (message.parentMessageId ?? null))
      const completed = matches.length === 1 ? matches[0] : undefined
      if (!completed || !completed.content.trim() || completed.content.endsWith("▋") ||
          !completed.content.startsWith(message.message.slice(0, -1))) return message
      return { ...message, message: completed.content,
        reasoning_time_taken: completed.reasoning_time_taken ?? message.reasoning_time_taken,
        serverMessageId: completed.serverMessageId ?? message.serverMessageId,
        serverMessageVersion: completed.serverMessageVersion ?? message.serverMessageVersion }
    })
    return { ...snapshot, messages }
  } catch {
    // The recoverable tab snapshot remains usable if the local mirror is unavailable.
    return snapshot
  }
}

/** Ownerless legacy keys are deliberately left untouched and never adopted. */
export const readSidepanelTabs = async (
  storage: Pick<ReturnType<typeof createSafeStorage>, "get">,
  tabId: number | null,
  ownerKey: string,
  isCurrent: () => boolean
): Promise<SidepanelTabsState | null> => {
  const keys = [getTabsStorageKey(tabId, ownerKey)]
  if (tabId !== null) keys.push(getTabsStorageKey(null, ownerKey))
  for (const key of keys) {
    if (!isCurrent()) return null
    const candidate = await storage.get<SidepanelTabsState>(key)
    if (!isCurrent()) return null
    if (candidate?.version === 2 && candidate.ownerKey === ownerKey &&
        Array.isArray(candidate.tabs) && candidate.snapshotsById &&
        typeof candidate.snapshotsById === "object" &&
        candidate.tabs.every(tab => typeof tab?.id === "string")) {
      const snapshotsById = { ...candidate.snapshotsById }
      for (const tab of candidate.tabs) {
        if (!isCurrent()) return null
        const snapshot = snapshotsById[tab.id]
        if (snapshot) snapshotsById[tab.id] = await recoverCompletedSnapshotReplies(snapshot, ownerKey, isCurrent)
      }
      return isCurrent() ? { ...candidate, snapshotsById } : null
    }
  }
  return null
}

export const readSidepanelRuntimeTabId = async (): Promise<number | null> => {
  try {
    if (!browser?.runtime?.sendMessage) {
      return null
    }

    const resp = await browser.runtime.sendMessage({
      type: "tldw:get-tab-id"
    })

    return typeof resp?.tabId === "number" ? resp.tabId : null
  } catch {
    return null
  }
}

const hasOverlayDraftSettings = async (
  storage: ReturnType<typeof createSafeStorage>,
  snapshot: SidepanelChatSnapshot | undefined,
  tab: SidepanelChatTab | undefined
): Promise<boolean> => {
  const chatKey = resolveChatSettingsKey({
    historyId: snapshot?.historyId ?? tab?.historyId ?? null,
    serverChatId: snapshot?.serverChatId ?? tab?.serverChatId ?? null
  })
  if (chatKey === "scratch") {
    const draftKey = getSidepanelDraftStorageKey(tab?.id)
    const resumeMarkerKey = getSidepanelOverlayResumeMarkerKey(draftKey)
    if (!resumeMarkerKey) return false
    return Boolean(await storage.get(resumeMarkerKey))
  }
  const settingsKey = getChatSettingsStorageKey(chatKey)
  const storedSettings = await storage.get(settingsKey)
  return Boolean(normalizeChatSettingsRecord(storedSettings)?.assistantOverlay)
}

const hasRestorableSnapshot = async (
  snapshot: SidepanelChatSnapshot | undefined,
  tab: SidepanelChatTab | undefined,
  storage: ReturnType<typeof createSafeStorage>
): Promise<boolean> => {
  if (tab?.historyId || tab?.serverChatId || tab?.serverChatTopic) {
    return true
  }

  if (!snapshot) {
    return false
  }

  if (
    snapshot.history.length > 0 ||
    snapshot.messages.length > 0 ||
    snapshot.historyId ||
    snapshot.serverChatId ||
    snapshot.serverChatTopic ||
    snapshot.serverChatClusterId ||
    snapshot.serverChatExternalRef ||
    snapshot.queuedMessages.length > 0
  ) {
    return true
  }

  return hasOverlayDraftSettings(storage, snapshot, tab)
}

export const hasResumableSidepanelChat = async (): Promise<boolean> => {
  const revision = useSidepanelChatTabsStore.getState().revision
  let snapshot: ServicePromptSnapshot | undefined
  try {
    snapshot = await loadServicePromptSnapshot([])
    const lease = snapshot
    const current = () => !lease.scopeSignal.aborted && !lease.scopeInvalidatedSignal.aborted &&
      useSidepanelChatTabsStore.getState().revision === revision
    if (!current()) return false
    const tabId = await readSidepanelRuntimeTabId()
    const storage = createSafeStorage({ area: "local" })
    const tabs = await readSidepanelTabs(storage, tabId, serverChatMirrorOwnerKey(snapshot), current)
    if (!current()) return false
    if (tabs) {
      const restorable = await Promise.all(tabs.tabs.map(tab =>
        hasRestorableSnapshot(tabs.snapshotsById[tab.id], tab, storage)))
      if (!current()) return false
      if (restorable.some(Boolean)) return true
    }
    const enabled = await copilotResumeLastChat()
    if (!current() || !enabled) return false
    const recentChat = await getRecentChatFromCopilot(snapshot)
    return current() && Boolean(recentChat)
  } catch {
    return false
  } finally {
    snapshot?.release()
  }
}
