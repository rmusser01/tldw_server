import { browser } from "wxt/browser"

import {
  getRecentChatFromCopilot
} from "@/db/dexie/helpers"
import { copilotResumeLastChat } from "@/services/app"
import {
  getChatSettingsStorageKey,
  normalizeChatSettingsRecord,
  resolveChatSettingsKey
} from "@/services/chat-settings"
import type { SidepanelChatSnapshot, SidepanelChatTab } from "@/store/sidepanel-chat-tabs"
import { createSafeStorage } from "@/utils/safe-storage"
import {
  getSidepanelDraftStorageKey,
  getSidepanelOverlayResumeMarkerKey
} from "@/utils/sidepanel-overlay-resume"
import type { ChatHistory, Message as ChatMessage } from "~/store/option"

export type LegacySidepanelChatSnapshot = {
  history: ChatHistory
  messages: ChatMessage[]
  chatMode: "normal" | "rag" | "vision"
  historyId: string | null
}

type SidepanelTabsState = {
  tabs: SidepanelChatTab[]
  activeTabId: string | null
  snapshotsById: Record<string, SidepanelChatSnapshot>
}

export const getTabsStorageKey = (id: number | null | undefined) =>
  id != null ? `sidepanelChatTabsState:tab-${id}` : "sidepanelChatTabsState"

export const getLegacyStorageKey = (id: number | null | undefined) =>
  id != null ? `sidepanelChatState:tab-${id}` : "sidepanelChatState"

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
  try {
    const tabId = await readSidepanelRuntimeTabId()
    const storage = createSafeStorage({
      area: "local"
    })

    const keysToTry: string[] = [getTabsStorageKey(tabId)]
    if (tabId != null) {
      keysToTry.push(getTabsStorageKey(null))
    }

    for (const key of keysToTry) {
      // eslint-disable-next-line no-await-in-loop
      const candidate = (await storage.get(key)) as SidepanelTabsState | null
      if (
        candidate &&
        Array.isArray(candidate.tabs) &&
        (await Promise.all(
          candidate.tabs.map((tab) =>
            hasRestorableSnapshot(candidate.snapshotsById?.[tab.id], tab, storage)
          )
        )).some(Boolean)
      ) {
        return true
      }
    }

    const legacyKeysToTry: string[] = [getLegacyStorageKey(tabId)]
    if (tabId != null) {
      legacyKeysToTry.push(getLegacyStorageKey(null))
    }

    for (const key of legacyKeysToTry) {
      // eslint-disable-next-line no-await-in-loop
      const candidate = (await storage.get(key)) as
        | LegacySidepanelChatSnapshot
        | null
      if (candidate && Array.isArray(candidate.messages)) {
        return true
      }
    }

    const isEnabled = await copilotResumeLastChat()
    if (!isEnabled) return false

    const recentChat = await getRecentChatFromCopilot()
    return Boolean(recentChat)
  } catch {
    return false
  }
}
