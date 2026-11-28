import type { Storage } from "@plasmohq/storage"
import { browser } from "wxt/browser"
import {
  ACTION_ICON_CLICK_SETTING,
  CONTEXT_MENU_CLICK_SETTING,
  getInitialConfig
} from "@/services/action"
import { getStorageForSetting } from "@/services/settings/registry"
import { getServerCapabilities } from "@/services/tldw/server-capabilities"

export type BackgroundCapabilities = {
  sendToTldw: boolean
  processLocal: boolean
  transcribe: boolean
}

export type BackgroundInitOptions = {
  storage: Storage
  contextMenuId: { webui: string; sidePanel: string }
  saveToClipperMenuId: string
  saveToCompanionMenuId: string
  saveToNotesMenuId: string
  narrateSelectionMenuId: string
  transcribeMenuId: { transcribe: string; transcribeAndSummarize: string }
  warmModels: (force?: boolean, throwOnError?: boolean) => Promise<any[] | null>
  capabilities: BackgroundCapabilities
  onActionIconClickChange: (value: string) => void
  onContextMenuClickChange: (value: string) => void
}

export type BackgroundInitResult = {
  modelWarmAlarmName: string
}

export const MODEL_WARM_ALARM_NAME = "tldw:model-warm"
const MODEL_WARM_INTERVAL_MINUTES = 360

const getServerUrl = (cfg: any): string => {
  if (!cfg || typeof cfg !== "object") return ""
  return String(cfg.serverUrl || "").trim()
}

const scheduleModelWarmAlarm = async (enabled: boolean) => {
  try {
    if (!browser?.alarms) return
    if (!enabled) {
      await browser.alarms.clear(MODEL_WARM_ALARM_NAME)
      return
    }
    await browser.alarms.clear(MODEL_WARM_ALARM_NAME)
    await browser.alarms.create(MODEL_WARM_ALARM_NAME, {
      periodInMinutes: MODEL_WARM_INTERVAL_MINUTES
    })
  } catch (error) {
    console.debug(
      "[tldw] model warm alarm setup failed:",
      (error as any)?.message || error
    )
  }
}

const syncWebClipperContextMenu = async (menuId: string) => {
  try {
    await browser.contextMenus.remove(menuId)
  } catch {
    // best effort removal before re-creating the menu item
  }

  let hasWebClipper = false
  try {
    // Cold start must not bypass the persisted capability cache; the cache
    // key is scoped to the connection, so config changes still miss.
    const capabilities = await getServerCapabilities()
    hasWebClipper = Boolean(capabilities.hasWebClipper)
  } catch {
    hasWebClipper = false
  }

  if (!hasWebClipper) {
    return
  }

  browser.contextMenus.create({
    id: menuId,
    title: browser.i18n.getMessage("contextSaveToClipper"),
    contexts: ["page", "selection"]
  })
}

export const initBackground = async (
  options: BackgroundInitOptions
): Promise<BackgroundInitResult> => {
  const {
    storage,
    contextMenuId,
    saveToClipperMenuId,
    saveToCompanionMenuId,
    saveToNotesMenuId,
    narrateSelectionMenuId,
    transcribeMenuId,
    warmModels,
    capabilities,
    onActionIconClickChange,
    onContextMenuClickChange
  } = options

  try {
    await browser.contextMenus.removeAll()
  } catch (error) {
    console.debug(
      "[tldw] contextMenus.removeAll failed:",
      (error as any)?.message || error
    )
  }

  const contextMenuTitle = {
    webui: browser.i18n.getMessage("openOptionToChat"),
    sidePanel: browser.i18n.getMessage("openSidePanelToChat")
  }

  getStorageForSetting(ACTION_ICON_CLICK_SETTING).watch({
    actionIconClick: (value) => {
      const oldValue = value?.oldValue || "webui"
      const newValue = value?.newValue || "webui"
      if (oldValue !== newValue) {
        onActionIconClickChange(newValue)
      }
    }
  })
  getStorageForSetting(CONTEXT_MENU_CLICK_SETTING).watch({
    contextMenuClick: (value) => {
      const oldValue = value?.oldValue || "sidePanel"
      const newValue = value?.newValue || "sidePanel"
      if (oldValue !== newValue) {
        onContextMenuClickChange(newValue)
        browser.contextMenus.remove(contextMenuId[oldValue])
        browser.contextMenus.create({
          id: contextMenuId[newValue],
          title: contextMenuTitle[newValue],
          contexts: ["page", "selection"]
        })
      }
    }
  })
  storage.watch({
    tldwConfig: (value) => {
      const nextUrl = getServerUrl(value?.newValue)
      const prevUrl = getServerUrl(value?.oldValue)
      const hasServer = nextUrl.length > 0
      void scheduleModelWarmAlarm(hasServer)
      // WP-62: the clipper context menu only depends on the connected
      // server, so re-sync it (and re-warm models) only when the URL actually
      // changed — unrelated config writes (e.g. credential refreshes) used to
      // rebuild the menu and re-resolve capabilities on every write.
      if (nextUrl !== prevUrl) {
        void syncWebClipperContextMenu(saveToClipperMenuId)
        if (hasServer) {
          void warmModels(true)
        }
      }
    }
  })

  const data = await getInitialConfig()
  onContextMenuClickChange(data.contextMenuClick)
  onActionIconClickChange(data.actionIconClick)

  // Top-level: Open Sidebar / Open Web UI (dynamic)
  browser.contextMenus.create({
    id: contextMenuId[data.contextMenuClick],
    title: contextMenuTitle[data.contextMenuClick],
    contexts: ["page", "selection"]
  })

  // Parent: AI Actions
  browser.contextMenus.create({
    id: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextMenuAIActions") || "AI Actions",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "summarize-pa",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextSummarize") || "Summarize",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "explain-pa",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextExplain") || "Explain",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "rephrase-pa",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextRephrase") || "Rephrase",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "translate-pg",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextTranslate") || "Translate",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "custom-pg",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextCustom") || "Custom",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: "contextual-popup-pa",
    parentId: "tldw-ai-actions",
    title: browser.i18n.getMessage("contextCopilotPopup") || "AI Popup",
    contexts: ["selection"]
  })

  // Parent: Save
  browser.contextMenus.create({
    id: "tldw-save",
    title: browser.i18n.getMessage("contextMenuSave") || "Save",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: saveToNotesMenuId,
    parentId: "tldw-save",
    title: browser.i18n.getMessage("contextSaveToNotes") || "Save to Notes",
    contexts: ["selection"]
  })
  void syncWebClipperContextMenu(saveToClipperMenuId)
  browser.contextMenus.create({
    id: saveToCompanionMenuId,
    parentId: "tldw-save",
    title:
      browser.i18n.getMessage("contextSaveToCompanion") || "Save to Companion",
    contexts: ["selection"]
  })
  browser.contextMenus.create({
    id: narrateSelectionMenuId,
    parentId: "tldw-save",
    title: browser.i18n.getMessage("contextNarrateSelection") || "Read Aloud",
    contexts: ["selection"]
  })

  // Parent: Process (only created if at least one capability is enabled)
  if (capabilities.sendToTldw || capabilities.processLocal || capabilities.transcribe) {
    browser.contextMenus.create({
      id: "tldw-process",
      title: browser.i18n.getMessage("contextMenuProcess") || "Process",
      contexts: ["page", "link"]
    })

    if (capabilities.sendToTldw) {
      browser.contextMenus.create({
        id: "send-to-tldw",
        parentId: "tldw-process",
        title: browser.i18n.getMessage("contextSendToTldw") || "Save to Library",
        contexts: ["page", "link"]
      })
    }

    if (capabilities.processLocal) {
      browser.contextMenus.create({
        id: "process-local-tldw",
        parentId: "tldw-process",
        title: browser.i18n.getMessage("contextProcessLocalTldw") || "Analyze without Saving",
        contexts: ["page", "link"]
      })
    }

    if (capabilities.transcribe) {
      browser.contextMenus.create({
        id: transcribeMenuId.transcribe,
        parentId: "tldw-process",
        title: browser.i18n.getMessage("contextTranscribeMedia") || "Transcribe Video/Audio",
        contexts: ["page", "link"]
      })
      browser.contextMenus.create({
        id: transcribeMenuId.transcribeAndSummarize,
        parentId: "tldw-process",
        title: browser.i18n.getMessage("contextTranscribeAndSummarizeMedia") || "Transcribe + Summarize",
        contexts: ["page", "link"]
      })
    }
  }

  let hasServer = false
  try {
    const cfg = await storage.get<any>("tldwConfig")
    hasServer = getServerUrl(cfg).length > 0
  } catch {
    hasServer = false
  }

  if (hasServer) {
    // Cold start warms from the persisted catalog (15-min TTL) instead of
    // forcing a refresh_openrouter=true round trip on every worker wake.
    void warmModels(false)
  }
  await scheduleModelWarmAlarm(hasServer)

  return { modelWarmAlarmName: MODEL_WARM_ALARM_NAME }
}
